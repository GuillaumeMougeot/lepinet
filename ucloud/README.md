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
  `ucloud jobs logs` takes the spec file, not the job id.
- **A spec with `[setup] run` needs a `[sync]` block**: the generated script is stored in the synced folder.
- **Inside a job, `os.cpu_count()` says 256.** Size thread pools from `/sys/fs/cgroup/cpu.max`.
- **`/work` reads scale with the number of readers, so give image jobs 256 workers or stage.** Two
  training jobs at 256 workers read ~1,000 img/s each at once (V0, 2026-11-03); their evaluations at
  32 workers read 80-94 img/s, two hours for the test fold. Concurrent jobs collapsed throughput once
  (July), so watch the first epoch.
- **`ucloud files download` copies a folder's contents, not the folder**: give each folder its own
  destination, or same-named run folders inside merge and overwrite each other's `predictions.csv`.
- **Mount `flemming` for any self-training job**: the pseudo-label parquet reaches trap images
  through `../../flemming/images/`.
- **A submission that fails with "POST /api/files/upload failed with 502" is not transient while the
  backend is slow.** The `[sync]` step asks for upload sessions in batches of 64 files, under a ~5 s
  gateway limit. On 2026-11-05, with many jobs staging from `/work`, 64 files overran it on every
  retry while 16 passed in 1.4 s. The queue marks the job FAILED and does not retry it, so resubmit
  with a smaller batch (the `mt` session used a 16-file wrapper).
- **An 8-GPU job waits for a whole free node; 3- and 4-GPU jobs start at once.** Split large jobs
  into 4-GPU ones.
- **Single-GPU jobs are scheduled first, but they share a node's local disk.** On 2026-11-06, seven
  single-GPU Lepidoptera jobs failed with "No space left on device": each stages the 452 GB image
  set to the node's `/tmp`. Small datasets (Pl@ntNet, 0.9 min) run fine one per job; Lepidoptera runs
  go 4 per job, staged once.

Mounts: `/work/lepinet` is this repo (pushed by `[sync]`), `/work/global_lepi` and
`/work/flemming_helsing` come from the `datasets` drive (`/12383016/...`), read-only.
