"""The calibration core, tested on synthetic data — no model, no images, no GPU.

These four functions decide what confidence a user is shown and when a name is greyed, so they are
worth pinning independently of the pipeline that feeds them. Each test states the property rather
than a golden number, so the tests survive a reimplementation.
"""
import numpy as np
import torch

from lepinet.calibrate import (
    TEMPERATURE_GRID,
    calibrated_confidence,
    choose_threshold,
    fit_temperature,
    report,
)


def _stats_from_logits(z: torch.Tensor, y: torch.Tensor) -> dict:
    """Build the streaming accumulator's output directly, so the pure functions can be tested
    without running a model."""
    grid = torch.tensor(TEMPERATURE_GRID, dtype=torch.float32)
    top_logit, top_idx = z.max(dim=1)
    return {
        "top1_idx": top_idx,
        "top1_logit": top_logit,
        "lse_grid": torch.logsumexp(z.unsqueeze(-1) / grid.view(1, 1, -1), dim=1),
        "true_logit": z.gather(1, y.unsqueeze(1)).squeeze(1),
    }


def test_temperature_fit_recovers_a_known_overconfidence():
    """An overconfident model should be corrected by a T near the factor it was inflated by.

    The construction matters and is easy to get wrong. Labels must be **sampled from** softmax(z),
    so the model's stated probabilities are truthful at T=1; presenting it as emitting 3*z then makes
    it exactly 3x overconfident and the NLL-optimal temperature is 3.

    Setting ``y = z.argmax(1)`` instead — the obvious thing — makes the model 100 % accurate, and the
    NLL-optimal temperature collapses to the grid floor, because a *perfect* classifier should be
    sharpened rather than softened. That is correct behaviour, not a bug, and it is the reason
    calibration is defined against realised accuracy rather than against the logits alone.
    """
    torch.manual_seed(0)
    base = torch.randn(4000, 40)
    y = torch.multinomial(torch.softmax(base, dim=1), 1).squeeze(1)   # truthful at T=1
    stats = _stats_from_logits(base * 3.0, y)                         # ...now 3x overconfident
    t, _, nll_t, nll_1 = fit_temperature(stats)
    assert 2.0 < t < 4.5, f"expected T near 3, got {t:.2f}"
    assert nll_t < nll_1, "calibration must not make NLL worse than T=1"


def test_a_perfectly_accurate_model_is_sharpened_not_softened():
    """The counterpart, pinned because it looks like a bug the first time you see it."""
    torch.manual_seed(0)
    z = torch.randn(2000, 40)
    stats = _stats_from_logits(z * 3.0, z.argmax(1))     # accuracy is 100 %
    t, _, _, _ = fit_temperature(stats)
    assert t < 1.0, f"a perfect classifier should be sharpened (T<1), got {t:.2f}"


def test_temperature_scaling_never_changes_the_prediction():
    """The central guarantee: calibration makes confidence honest, it does not touch accuracy."""
    torch.manual_seed(1)
    z = torch.randn(500, 30) * 4
    for t in (0.3, 1.0, 2.5, 7.0):
        assert torch.equal((z / t).argmax(1), z.argmax(1))


def test_choose_threshold_picks_the_lowest_cut_meeting_the_target():
    """Lowest, not any: a higher cut greys names for precision the target did not ask for."""
    conf = np.array([0.95, 0.90, 0.85, 0.80, 0.20])
    correct = np.array([1, 1, 1, 1, 0], dtype=bool)   # everything above 0.8 is right
    thr = choose_threshold(conf, correct, 0.95)
    assert thr == 0.80
    r = report(conf, correct, thr)
    assert r["coverage"] == 0.8 and r["precision_among_shown"] == 1.0


def test_choose_threshold_returns_none_when_the_claim_cannot_be_supported():
    """An unreachable target is a real answer -- the UI must not make the claim."""
    conf = np.array([0.9, 0.8, 0.7, 0.6])
    correct = np.array([0, 1, 0, 1], dtype=bool)      # 50 % at best, at any cut
    assert choose_threshold(conf, correct, 0.95) is None
    assert report(conf, correct, None)["coverage"] == 0.0


def test_calibrated_confidence_is_a_probability_and_tracks_temperature():
    """Confidence must be in [0,1], and a larger T must make the model less confident."""
    torch.manual_seed(2)
    z = torch.randn(300, 25) * 5
    stats = _stats_from_logits(z, z.argmax(1))
    lo = calibrated_confidence(stats, int(np.abs(TEMPERATURE_GRID - 0.5).argmin()))
    hi = calibrated_confidence(stats, int(np.abs(TEMPERATURE_GRID - 4.0).argmin()))
    assert ((0 <= lo) & (lo <= 1)).all() and ((0 <= hi) & (hi <= 1)).all()
    assert hi.mean() < lo.mean(), "higher temperature must reduce confidence"


def test_grid_accumulation_matches_recomputing_from_full_logits():
    """The streaming trick must be exact, not approximate -- it is the reason the 12,041-wide
    logit matrix is never retained."""
    torch.manual_seed(3)
    z = torch.randn(64, 200)
    stats = _stats_from_logits(z, z.argmax(1))
    for t_index in (0, 40, 95):
        t = float(TEMPERATURE_GRID[t_index])
        assert torch.allclose(stats["lse_grid"][:, t_index],
                              torch.logsumexp(z / t, dim=1), atol=1e-4)
