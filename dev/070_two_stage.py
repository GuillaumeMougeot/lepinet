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


def main(argv):
    if len(argv) < 2 or argv[0] != "train":
        raise SystemExit(__doc__)
    cfg_path = argv[1]
    if "--init-from" not in argv:
        raise SystemExit("--init-from is required; without it this is just `lepinet train`.")
    init_from = argv[argv.index("--init-from") + 1]
    freeze = "--freeze-body" in argv

    from lepinet.train import train_from_config

    print(f"stage 2: init_from={init_from}, freeze_body={freeze}")
    train_from_config(cfg_path, init_from=init_from, freeze_body=freeze)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main(sys.argv[1:])
