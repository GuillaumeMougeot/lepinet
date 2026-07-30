# What killed the 12-epoch run: nothing was ticking the queue

**Kind:** incident · **Status:** RESOLVED (2026-07-30) — `ucloud q` is a state file, not a service.
No daemon was running, so nothing extended running jobs or launched queued ones.

## Symptom

Two things went wrong at once and looked unrelated:

1. The owner's **12-epoch DINOv3-ConvNeXt-L run expired mid-training**, despite its TOML carrying
   `auto_extend = "1h"` and `max_time = "12h"`.
2. `lepi-marginal` had **finished successfully**, yet its chained `lepi-marginal-eval` was still
   sitting `QUEUED` — with no dependency left to wait for.

`ucloud q ls` looked entirely healthy. That is the trap.

## Root cause

`ucloud q` keeps its queue as **state on disk**. It has no resident process. Every state transition
— reconciling a finished job, extending a low-time one, submitting a dependent whose `--after` is
satisfied — happens inside a **tick**, and a tick only happens when something calls it:

```
ucloud q tick     # advance the queue once (documented as cron-safe)
ucloud q daemon   # tick in a loop
```

Neither was running. `ps aux | grep ucloud` returned nothing and `crontab -l` reported no crontab.

So `auto_extend` was never a promise the system could keep on its own: it is a *policy* that a tick
enforces. With no ticker, it is inert, and a job simply runs until its `time_allocation` expires.
Likewise a `QUEUED` job is not waiting on a scheduler — it is waiting on a tick that never comes.

**The states shown by `ucloud q ls` are the last recorded ones, not live ones.** That is why the
listing looked fine while nothing was progressing: `RUNNING` meant "was running when last ticked".

## Fix

A single `ucloud q tick` immediately reconciled `lepi-marginal` to DONE and submitted its eval as
job `12362002`. A daemon was then started at a 2-minute interval:

```bash
ucloud q daemon --interval 120
```

Stopping the daemon never harms running jobs — they are ordinary UCloud jobs; queued ones simply
wait on disk until something ticks again. That property is what makes the failure mode so quiet, and
it is also what makes the fix free.

**The durable fix is a cron entry** (`ucloud q tick`), because a daemon started inside a shell
session dies with the session — which is almost certainly how this happened in the first place.

## The lesson

**Check that the thing enforcing your policy is actually alive.** `auto_extend` in a config file
reads like a guarantee, but it is only a request that some process must act on. Before trusting any
queue, scheduler or watchdog written as configuration, confirm the component that *executes* it is
running — the config being correct and the config being enforced are different facts.

Corollary for this project: **before diagnosing a lost run as a cluster or hardware fault, check the
queue daemon.** That is now the first question, ahead of node health
([[2026-07-16-gpu-hang]]) and ahead of reading logs for an exit code.

## Consequence for the plan

The expired 12-epoch run was **not** restarted as-is. It trained the *old* multi-head architecture,
which the single-head + marginalisation result has superseded, so its number would have landed on a
baseline no longer in use — precisely the hygiene failure [`PLAN.md`](PLAN.md) exists to prevent. The
longer-schedule question (B0) is better asked once A2 lands, as 12 epochs of **A2's** configuration,
where the comparison against A2-at-6-epochs isolates exactly one factor.
