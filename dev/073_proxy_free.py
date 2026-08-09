"""H4 — a classifier with no prototype matrix: class centroids maintained by EMA.

This is the last open candidate in Group H ([[2026-08-05-scaling-the-head]]). The problem it attacks
is memory, not accuracy: at ~1 M species a 1280-wide prototype matrix is **5.1 GB of parameters plus
10.2 GB of Adam state**, and the optimiser state is the part that does not fit. Every other option
has already been measured and rejected --

    low-rank factorisation   dead     rank 1035/1280; the margin *spends* dimensions
    fixed / taxonomy codes   weakened by the same spectrum
    uniform sampled softmax  dead     smooth, no plateau; 1024/1M is 0.1 % coverage
    hard-negative sampling   weakened taxonomy-aware negatives recovered only 26 % (H3)

-- while **inference** is solved: replacing the trained matrix with train-set class centroids costs
0.29 pt (H1). H4 asks the obvious follow-up. If centroids are good enough to *predict* with, are they
good enough to *train against*?

**The mechanism.** The finest level's `nn.Linear` is replaced by a `register_buffer`. Each class
prototype is an exponential moving average of the embeddings of its own training examples, updated
under `no_grad` after each backward. There is no gradient with respect to the prototypes and therefore
no optimiser state for them: **15.3 GB becomes 5.1 GB at 1 M classes, a 3x reduction**, and the saved
component is the one that scales worst.

The backbone still learns. The loss gradient flows into the embedding, which is pulled toward its own
class centroid and away from the others -- the same signal a trained matrix provides, minus the
matrix's freedom to move independently of the data. Whether that freedom was load-bearing is exactly
what this measures.

**Two details that decide whether it trains at all.**

*Cold start.* A class whose centroid is initialised randomly starts orthogonal to its own examples,
and an EMA at momentum 0.9 needs tens of updates to escape. A species with 50 images would spend most
of an epoch pointing at noise. So a class's **first** observation replaces its centroid outright
rather than blending, and the EMA only engages from the second onward.

*Update after backward, not merely after scoring.* Two separate reasons, and only the first is
obvious. (a) If the centroids absorbed the current batch before the logits were computed, every
example would be scored against a prototype that already contains it -- training accuracy inflates
and validation is the only place the problem shows. (b) Less obvious and caught by the selftest:
`F.linear` **saves the centroid tensor** to compute the gradient w.r.t. the embedding, so mutating
the buffer any time before `backward()` raises "one of the variables needed for gradient computation
has been modified by an inplace operation". Cloning to dodge that would allocate a second 5.1 GB
matrix per forward at 1 M classes and defeat the point of the head. The embedding is therefore
stashed at forward (B x d) and the callback applies the update in `after_backward`.

Coarser levels keep their ordinary trained layers: 4,333 genera and 102 families are not the memory
problem, and changing them too would confound the measurement.

    python dev/073_proxy_free.py train configs/<cfg>.yaml
    python dev/073_proxy_free.py test  --model ... --parquet ...   # registers the head, then delegates
    python dev/073_proxy_free.py selftest          # CPU, no data

`test` exists only so the head is in HEAD_REGISTRY before the checkpoint is rebuilt. `lepinet test`
on a proxy-free checkpoint dies with "Unknown head 'proxy_free'" -- three scripts in this project
have already died on that, and it is listed in PLAN.md's queue discipline.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lepinet.heads import HEAD_REGISTRY, IndependentHead, cosine_to_zscore


class ProxyFreeHead(IndependentHead):
    """Finest level scored against EMA class centroids held in a buffer; no prototype parameters.

    Args:
        ema_momentum: blend factor for a class already seen. 0.9 keeps ~10 batches of history.
        levels_proxy_free: how many levels, fine->coarse, use centroids. 1 = species only.
    """

    def __init__(
        self,
        in_features: int,
        n_classes,
        hidden: bool | int = True,
        droprate: float = 0.1,
        ema_momentum: float = 0.9,
        levels_proxy_free: int = 1,
    ):
        super().__init__(in_features, n_classes, hidden, droprate)
        self.ema_momentum = float(ema_momentum)
        self.n_proxy_free = int(levels_proxy_free)
        d = self.preclass_size

        for i in range(self.n_proxy_free):
            n = self.n_classes[i]
            self.register_buffer(f"centroids_{i}", F.normalize(torch.randn(n, d), dim=1))
            self.register_buffer(f"seen_{i}", torch.zeros(n, dtype=torch.long))
            # Drop the trained layer entirely. Leaving it in place would keep 5.1 GB of unused
            # parameters at 1 M classes and make the memory claim false while the accuracy claim
            # stayed true -- the worst combination to discover later.
            self.layers[i] = nn.Identity()

        self._labels: list[torch.Tensor] | None = None
        self._pending_emb: torch.Tensor | None = None

    def centroids(self, i: int) -> torch.Tensor:
        return getattr(self, f"centroids_{i}")

    @torch.no_grad()
    def update_centroids(self, emb: torch.Tensor, labels) -> None:
        emb = emb.detach().float()
        for i in range(self.n_proxy_free):
            y = labels[i].long().view(-1)
            C, seen = self.centroids(i), getattr(self, f"seen_{i}")
            summed = torch.zeros_like(C).index_add_(0, y, emb)
            counts = torch.zeros(C.shape[0], device=y.device, dtype=emb.dtype).index_add_(
                0, y, torch.ones(y.numel(), device=y.device, dtype=emb.dtype))
            hit = counts > 0
            mean = summed[hit] / counts[hit].unsqueeze(1)
            # First sighting replaces; afterwards, blend. See the cold-start note in the docstring.
            first = (seen[hit] == 0).unsqueeze(1)
            m = torch.where(first, torch.zeros_like(mean), torch.full_like(mean, self.ema_momentum))
            C[hit] = F.normalize(m * C[hit] + (1.0 - m) * mean, dim=1)
            seen[hit] += counts[hit].long()

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        emb = self.preclassification(x)
        out = []
        for i, layer in enumerate(self.layers):
            if i < self.n_proxy_free:
                out.append(cosine_to_zscore(F.linear(emb, self.centroids(i)), self.preclass_size))
            else:
                out.append(cosine_to_zscore(F.linear(emb, layer.weight), self.preclass_size)
                           + layer.bias)
        # The update CANNOT happen here. `F.linear` saves the centroid tensor to compute the
        # gradient w.r.t. `emb`, so mutating the buffer in place before backward raises
        # "one of the variables needed for gradient computation has been modified by an inplace
        # operation". Cloning instead would allocate a second 5.1 GB matrix per forward at 1 M
        # classes and destroy the only reason this head exists. So the embedding is stashed here
        # (B x d, negligible) and the callback applies the update after backward.
        if self.training and self._labels is not None:
            self._pending_emb = emb.detach()
        return out

    def apply_pending_update(self) -> None:
        """Called after backward. No-op unless a training forward stashed an embedding."""
        if getattr(self, "_pending_emb", None) is not None and self._labels is not None:
            self.update_centroids(self._pending_emb, self._labels)
            self._pending_emb = None


HEAD_REGISTRY["proxy_free"] = ProxyFreeHead


class CentroidContextCallback:
    """Hands the batch labels to the head so it can update its centroids.

    Built lazily so importing this module does not require fastai (the selftest runs on CPU with
    neither fastai nor a GPU). Mirrors `dev/050`'s MarginContextCallback -- and carries the same
    warning: registering the head without this callback is silent. The model trains, the centroids
    never move from their random init, and the run reports a plausible-looking bad number.
    """

    def __new__(cls):
        from fastai.callback.core import Callback

        class _CentroidContext(Callback):
            order = -5

            def before_batch(self):
                head = self.learn.model[1].head
                head._labels = list(self.learn.yb)

            def after_backward(self):
                # After backward, before the optimiser step: the graph that saved the centroid
                # tensor is already consumed, so mutating the buffer is safe here and nowhere
                # earlier.
                self.learn.model[1].head.apply_pending_update()

            def after_validate(self):
                head = self.learn.model[1].head
                head._labels = None
                head._pending_emb = None

        return _CentroidContext()


def memory_report(n_classes: int, d: int = 1280) -> str:
    matrix = n_classes * d * 4 / 1e9
    return (f"{n_classes:,} classes x {d}: trained matrix {matrix:.2f} GB params + "
            f"{2 * matrix:.2f} GB Adam state = {3 * matrix:.2f} GB; "
            f"proxy-free {matrix:.2f} GB buffer only ({3:.0f}x less)")


def selftest() -> None:
    """CPU, no data, no fastai. Checks the three things that would silently produce a bad run."""
    torch.manual_seed(0)
    d, n_sp, n_gn = 16, 7, 3
    head = ProxyFreeHead(d, [n_sp, n_gn], hidden=False, droprate=0.0, ema_momentum=0.9)

    # 1. No parameters for the proxy-free level, and the buffer is unit-norm.
    names = [n for n, _ in head.named_parameters()]
    assert not any(n.startswith("layers.0") for n in names), f"level 0 still has parameters: {names}"
    assert torch.allclose(head.centroids(0).norm(dim=1), torch.ones(n_sp), atol=1e-5)
    print(f"  no level-0 parameters; buffer unit-norm.  params={names}")

    # 2. First sighting replaces, so one step puts the centroid exactly on the example.
    head.train()
    x = torch.randn(4, d)
    y_sp = torch.tensor([0, 0, 1, 2])
    head._labels = [y_sp, torch.tensor([0, 0, 1, 1])]
    emb = head.preclassification(x)
    snap0 = head.centroids(0).clone()
    head(x)
    assert torch.equal(head.centroids(0), snap0), "forward must not mutate the buffer (see backward)"
    head.apply_pending_update()
    got, want = head.centroids(0)[1], F.normalize(emb[2], dim=0)
    assert torch.allclose(got, want, atol=1e-5), (got - want).abs().max()
    c0_want = F.normalize(emb[:2].mean(0), dim=0)
    assert torch.allclose(head.centroids(0)[0], c0_want, atol=1e-5)
    assert head.seen_0.tolist() == [2, 1, 1, 0, 0, 0, 0]
    print("  first sighting replaces (single and averaged); `seen` counts correctly.")

    # 3. Scoring happens BEFORE the update: logits must not reflect this batch's absorption.
    head2 = ProxyFreeHead(d, [n_sp, n_gn], hidden=False, droprate=0.0)
    head2.train()
    before = head2.centroids(0).clone()
    head2._labels = [y_sp, torch.tensor([0, 0, 1, 1])]
    logits = head2(x)[0]
    emb2 = head2.preclassification(x)
    expected = cosine_to_zscore(F.linear(emb2, before), head2.preclass_size)
    assert torch.allclose(logits, expected, atol=1e-5), "logits used post-update centroids"
    head2.apply_pending_update()
    assert not torch.allclose(head2.centroids(0), before), "centroids did not update at all"
    print("  logits computed against pre-update centroids, and the update did happen.")

    # 4. Eval mode must not touch the buffers.
    head2.eval()
    snap = head2.centroids(0).clone()
    head2(x)
    head2.apply_pending_update()
    assert torch.equal(head2.centroids(0), snap), "centroids moved during eval"
    print("  eval mode leaves centroids untouched.")

    # 5. The gradient reaches the backbone, and no prototype parameter exists to receive one.
    #
    # This check is written with `hidden=True` on purpose. With `hidden=False` the level-0 logits
    # have NO trainable parameter anywhere in their path -- `preclassification` is just a normalize,
    # and the prototypes are a buffer -- so the species loss can only train the backbone. That is
    # fine when the backbone is unfrozen and fatal when it is not: **a proxy-free head cannot be
    # used for a frozen-trunk stage (cRT, adaptation) unless the bottleneck is kept.** Both of this
    # project's cheap classifier stages freeze the body, so this is a real constraint on where H4
    # can be deployed, not a detail of the test.
    head3 = ProxyFreeHead(d, [n_sp, n_gn], hidden=True, droprate=0.0)
    head3.train()
    xg = torch.randn(4, d, requires_grad=True)
    head3._labels = [y_sp, torch.tensor([0, 0, 1, 1])]
    head3(xg)[0].sum().backward()
    head3.apply_pending_update()          # the real order: backward first, then update
    assert xg.grad is not None and xg.grad.abs().sum() > 0, "no gradient reached the backbone"
    assert head3.hidden.weight.grad is not None, "the bottleneck received no gradient"
    assert not hasattr(head3.centroids(0), "grad") or head3.centroids(0).grad is None
    print("  gradient reaches backbone and bottleneck; centroids receive none.")

    head4 = ProxyFreeHead(d, [n_sp, n_gn], hidden=False, droprate=0.0)
    assert not any(p.requires_grad for n, p in head4.named_parameters()
                   if n.startswith("layers.0") or n.startswith("hidden")), \
        "hidden=False must leave no trainable parameter on the species path"
    print("  hidden=False leaves the species path with no trainable parameter (frozen-trunk trap).")

    print("  " + memory_report(1_000_000))
    print("selftest: OK")


def main(argv):
    if argv and argv[0] == "selftest":
        return selftest()
    if argv and argv[0] == "test":
        # The head is registered by importing this module; delegate everything else to the CLI.
        from lepinet.cli import app
        print("H4 eval: proxy_free registered, delegating to `lepinet test`")
        return app(["test", *argv[1:]], standalone_mode=False)
    if len(argv) < 2 or argv[0] != "train":
        raise SystemExit(__doc__)
    from lepinet.config import load_config
    from lepinet.train import train_from_config

    cfg, _ = load_config(argv[1])
    if cfg.head != "proxy_free":
        raise SystemExit(f"config head is {cfg.head!r}, expected 'proxy_free'")
    print("H4: proxy-free head (+CentroidContextCallback)")
    print("  " + memory_report(1_000_000))
    train_from_config(argv[1], extra_cbs=[CentroidContextCallback()])


if __name__ == "__main__":
    main(sys.argv[1:])
