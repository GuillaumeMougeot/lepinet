# archive/bash — the multilabel era (2025-10 → 2026-06)

Launchers for the pre-hierarchical-heads line (`dev/005`, `011`, `012`, `014`, `017`, `022`,
`024`). Superseded by the `dev/028`/`dev/030` heads work.

Most carry `#SBATCH` headers targeting the GPU24 cluster with `/home/george/...` paths — they
were never local scripts and will not run on this workstation.

**Their `--config configs/<name>.yaml` paths are historical, not live.** Those configs now sit
in [`../configs/`](../configs/) (i.e. `archive/configs/`). The paths were deliberately left
unrewritten — see [`../README.md`](../README.md) for why.

Current launchers: [`../../bash/README.md`](../../bash/README.md).
