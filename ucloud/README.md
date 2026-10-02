# ucloud/ — cluster job specs

One TOML per UCloud job, submitted with [`ucloud-api`](https://github.com/GuillaumeMougeot/ucloud-api)
(`ucloud q submit <spec>`). Only the live specs stay here: the shipped models (B8, P5) as templates
to clone, the I/O probe, and the TreeOfLife jobs. The ~300 specs of finished experiments are in
[`../archive/ucloud/`](../archive/ucloud/); copy one back here to re-run it (paths inside are
relative to `ucloud/`).

| file | what it is |
|---|---|
| `lepinet-B8*.toml`, `lepinet-P5*.toml` | train / eval / probe / probe-HO / open-set rules / abstention for the two shipped models. **Clone these; never hand-write a `run =` line** |
| `lepinet-ioprobe3.toml` | one-minute cold-read throughput probe: tells starvation from a hang |
| `lepinet-tol*.toml` | TreeOfLife plan, split, probe, I/O test and fetch (1-4 vCPU) |
| `setup-lepinet.sh`, `setup-crawler.sh`, `setup.sh`, `setup-staged.sh`, `stage.py` | environment setup run inside the job |
| `budget_check.py` | the CPU budget guard (see below) |

## Rules, each learned by breaking it

- **Cost a CPU job before submitting it.** CPU core-hours are scarcer than GPU hours. Worst case =
  vCPU x `max_time`. Over 8 vCPU or 300 core-hours needs `# budget-approved: <who/when/why>` in the
  spec. `budget_check.py` enforces it as a pre-submit hook, a CI test and at runtime.
- **The queue only advances when a tick succeeds.** `auto_extend` and `--after` do nothing without
  one; cron runs it. Check it is *working*: `tail ~/.ucloud-tick.log`.
- **Read job logs, not job status.** A job reports SUCCESS while the script inside exits 1.
- **Inside a job, `os.cpu_count()` says 256.** Size thread pools from `/sys/fs/cgroup/cpu.max`.
- **Run one image-heavy job at a time**: concurrent jobs collapsed `/work` read throughput once.
- **Mount `flemming` for any self-training job**: the pseudo-label parquet reaches trap images
  through `../../flemming/images/`.

Mounts: `/work/lepinet` is this repo (pushed by `[sync]`), `/work/global_lepi` and
`/work/flemming_helsing` come from the `datasets` drive (`/12383016/...`), read-only.
