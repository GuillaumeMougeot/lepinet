# What killed the training box overnight on 2026-07-16?

**Status:** RESOLVED as far as the evidence allows — **almost certainly hardware, not software.**
Root cause not provable; the decisive evidence is an absence (see below).

## What happened

The 10-epoch one-cycle run (`20260715-073321`, started 07-15 07:33) died at **~03:34** on
07-16, roughly 20h into continuous full-power CUDA training. Discovered ~10:00. The machine
(HP Z2 Tower G1i workstation, RTX 5090) was:

- answering **ping**, but refusing SSH
- showing **"no signal"** on the monitor — not a frozen image, no signal at all
- totally unresponsive to keyboard and mouse

Recovered by holding the power button. System came back up at 10:19:49.

## Ruled out

| suspect | evidence against |
|---|---|
| training-side fault (OOM/NaN) | Log stops mid-batch with **no traceback**. 8 epochs healthy, f1_species 0.7152 → 0.8815 monotonic, ~2:13/epoch. |
| the reboot itself | Red herring — the run should have *finished* by ~10:00. The 10:19 reboot was a consequence (the manual power-cycle), not the cause. The real event was the ~03:34 hang. |
| unattended-upgrades auto-reboot | All `Automatic-Reboot` directives commented out in `/etc/apt/apt.conf.d/50unattended-upgrades`. |
| kernel / NVIDIA driver upgrade | Last package upgrade was **07-02 06:43**, two weeks earlier — only picked up *by* this forced reboot. The box had been running the old loaded kernel throughout. |
| a systemd timer | None align with 03:33:56. |
| the one `Call Trace` in kern.log | Dated **07-13**, three days prior, thermald-related. Unrelated. |
| another user's job | Only inference was running, mid-night. No load spike. |

## The decisive finding: total silence

`/var/log/kern.log` and `/var/log/syslog` contain **zero lines of any kind** between
~03:33:50 (a clean `fwupd-refresh.service` completion, 6 seconds before the training log's
last write) and the 10:20:01 boot. No Xid, no NVRM error, no soft-lockup, no NMI watchdog, no
panic.

**That absence is the evidence.** Any recoverable fault — a GPU Xid reset, an OOM kill, a
kernel panic — logs something first. Total silence means whatever failed was abrupt enough to
take out the logging path with it. That is not how driver bugs fail; it is how power delivery
and thermal events fail. Corroborated by the physical symptoms: "no signal" rather than a
frozen desktop, dead input, ping answering (NIC firmware alive) while everything above it was
gone.

**Leading hypothesis:** a power-delivery or thermal event under sustained ~320–350W, ~20h into
a continuous full-load run. The RTX 5090 is a consumer card, not validated for 24/7 sustained
compute the way an A100/H100 is; long unattended training is precisely the stress pattern that
exposes this.

## Mitigations — proposed, NOT applied

Both were offered and never actioned; recording so the option is not lost:

- **Cap sustained GPU power**: `sudo nvidia-smi -pl <watts>` for long unattended runs. Trades
  throughput for headroom.
- **Health logger** alongside multi-hour jobs:
  `nvidia-smi --query-gpu=temperature.gpu,power.draw --format=csv -l 60 >> gpu_health.log`.
  dmesg will not catch the next one either, so a thermal/power *trend* is the only forewarning
  available.

## Diagnostic access limitation

Kernel logs are not readable without root here: `sudo` needs a password, `dmesg` is blocked by
`dmesg_restrict=1`, and `kern.log`/`syslog` are `syslog:adm` mode 640 while the user is not in
`adm`. Every log quoted above had to be relayed by hand.

**Fix for next time:** `sudo usermod -aG adm $USER`, then re-login. Not yet done.

## What came out of it

dev/030 gained `resume_checkpoint` / `resume_epochs_done` + `fit_resume`, which rebuild the
original LR-schedule function and continue it from the exact fractional position an interrupted
run left off, instead of restarting the anneal. Resumed `20260715-073321` from its epoch-7
checkpoint into `20260716-105029`, which finished epochs 8–9 correctly and scored the project's
best result (see [2026-07-why-was-fastai-behind-mini-trainer.md](2026-07-why-was-fastai-behind-mini-trainer.md)).

Requires the original run to have used `warmup_epochs > 0` or `schedule: front_loaded` — the
built-in `fit_one_cycle`/`fit_flat_cos` paths expose no schedule function to resume from.
