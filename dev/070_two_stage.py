"""Two-stage training: cRT (L4) and target-domain fine-tuning (T2), from one runner.

Both are the same mechanic — start from a trained checkpoint, optionally freeze the backbone, train
on a different distribution — and both answer questions the project has open.

**L4 / cRT** (Kang et al.). The long-tail 2x2 showed tail-reweighting trades robustness for accuracy
monotonically ([[2026-08-01-imbalance-methods-bench]]). cRT asks *where* the damage is: train the
representation on the natural distribution, then retrain **only the classifier** with rebalancing.
If oversampling harms the *representation*, cRT escapes the trade entirely; if it harms only the
classifier, cRT should recover the in-distribution gain without the shifted cost. Either answer is
worth having, and stage 2 is cheap because the backbone is frozen.

**T2 / integration strategy.** The owner's Group T question: given target labels, is it better to mix
them into training or to fine-tune on them afterwards? Fine-tuning is the arm the package could not
express until now.

    python dev/070_two_stage.py train configs/<stage2>.yaml --init-from '...*.pt' [--freeze-body]
"""
from __future__ import annotations

import sys
from pathlib import Path


def _register_dev_heads():
    """dev/-registered heads are invisible to the package unless dev/050 is imported. A stage-2 run
    whose *source checkpoint* used one (e.g. `marginal_arcface`, inherited by every config derived
    from F1) otherwise dies in `build_head` with "Unknown head"."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "dev050_heads", Path(__file__).with_name("050_hierarchical_heads.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(argv):
    if len(argv) < 2 or argv[0] != "train":
        raise SystemExit(__doc__)
    cfg_path = argv[1]
    if "--init-from" not in argv:
        raise SystemExit("--init-from is required; without it this is just `lepinet train`.")
    init_from = argv[argv.index("--init-from") + 1]
    freeze = "--freeze-body" in argv

    heads = _register_dev_heads()

    from lepinet.config import load_config
    from lepinet.train import train_from_config

    cfg, _ = load_config(cfg_path)
    # `marginal_arcface` applies its margin inside forward and needs the batch labels, supplied by
    # MarginContextCallback. Without it the head silently degrades to a plain MarginalHead -- no
    # crash, no warning, and a stage that quietly does not match its own config.
    cbs = [heads.MarginContextCallback()] if cfg.head == "marginal_arcface" else None
    print(f"stage 2: init_from={init_from}, freeze_body={freeze}, head={cfg.head}"
          + (" (+MarginContextCallback)" if cbs else ""))
    train_from_config(cfg_path, init_from=init_from, freeze_body=freeze, extra_cbs=cbs)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main(sys.argv[1:])
