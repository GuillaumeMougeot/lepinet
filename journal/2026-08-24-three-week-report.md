# Three-week report: 2 August to 24 August

**Kind:** infrastructure · **Status:** **RESOLVED (2026-08-24).** The report the owner asked for
before leaving. Covers everything run, every committed prediction and whether it held, every claim
corrected, and what is left.

**Scope note.** The owner was away 2–24 August. I ran autonomously 2–10 August and then stopped; the
last 14 days are idle, not silent-failing. Two runs completed after I stopped and **their numbers
have not been read yet** — see §6.

---

## 1. What the project asserts now that it did not on 2 August

Six things, in rough order of how much they change what we would write.

1. **Domain adaptation is a classifier problem, not a representation problem.** Two epochs on a
   frozen trunk recover **83 % of the probe gain and 89 % of the held-out gain** of full
   self-training. The control (T2b) then showed a trunk that never saw the target domain, even in
   imitation, adapts just as well — so hand-authored domain augmentation and classifier adaptation
   are **substitutes**, not complements (`domain_aug` is worth +4.75 pt alone and +0.57 once the
   classifier is adapted). [[2026-08-06-adaptation-is-mostly-a-classifier-problem]]

2. **The whole recipe is a classifier-stage story.** Coarse supervision, long-tail rebalancing,
   domain adaptation and the prototype matrix itself all resolved toward the classifier. F2 is the
   constructive proof: one clean representation plus 2 + 2 frozen-trunk epochs gives in-distribution
   0.9081 and probe 0.7541, better in-distribution than any end-to-end 20 M model.
   [[2026-08-06-f2-capstone]]

3. **Self-training's confidence gate was never doing what it was for, and is now gone.** The gate
   was introduced against a circularity risk that never materialised. Four runs varying only the
   gate showed: a quantile gate loses coverage as the labeller improves (R2, −3.80 pt); a per-species
   cap recovers it (R3, +4.24 pt) *while label accuracy falls 24.6 points*; and the cap's real
   function is **class balancing**, not filtering. The recommended setting is now **no gate at all,
   with balanced replication**. [[2026-08-08-self-training-does-not-iterate]]

4. **Balanced replication is resampling, and it inherits resampling's scale profile.** Free at 20 M
   (+1.51 probe / +1.87 held-out), a trade at 198 M (+0.92 / −0.82), and actively harmful to
   end-to-end training at any scale (−0.71 / −3.62). Only a frozen trunk is protected, because
   balancing concentrates replication on the classes with fewest unique images — which are also
   those with least reliable pseudo-labels.
   [[2026-08-10-balance-is-oversampling-and-it-does-not-scale]]

5. **Novelty detection is monotone in taxonomic distance, and it is not a rarity artefact.** The
   original result took its novel taxa for free (everything below the 50-image training floor), so
   "unseen" was confounded with "rare". Retraining with **231 common taxa deliberately withheld**
   gives the same ordering with every stratum slightly *better*: near 0.8717, mid 0.9463, far 0.9726.
   Two novel populations chosen by opposite criteria, same ordering.
   [[2026-08-08-is-novelty-monotone-or-just-rare]]

6. **An open-set scoring rule does not transfer across scale.** `max-logit` is best at 20 M and among
   the worst at 198 M, where MSP beats it by 6.1–7.6 points. This one invalidated an earlier
   conclusion of our own rather than adding to it.
   [[2026-08-01-the-scoring-rule-was-the-bug]]

---

## 2. The prediction scorecard

Every archival entry in this window states a committed prediction before the result lands. That is
the practice worth keeping, and it only means anything if the misses are counted too.

| run | predicted | landed | verdict |
|---|---|---|---|
| T2 | probe 0.720–0.745 | **0.7572** | **wrong**, above range; falsification line missed by 0.3 pt |
| T2b | probe 0.700–0.730 | **0.7515** | **wrong**, and the better of the two readings won |
| F2 | in-dist 0.900–0.910, probe 0.740–0.765 | 0.9081 / 0.7541 | **both inside** |
| G1 | in-dist 0.912–0.920 | 0.9112 | near-miss, 0.0008 below |
| G2 | probe 0.780–0.800 | 0.7648 | **below range** |
| F3 | in-dist 0.902–0.910, probe 0.745–0.760 | 0.9061 / 0.7479 | inside range, but tripped its own falsification line by 0.21 |
| R2 | probe 0.758–0.772 | **0.7161** | **falsified by 4.2 pt — the largest miss of the project** |
| R3 | probe 0.755–0.775 | 0.7585 | **inside** |
| R4 | probe 0.760–0.780 | 0.7674 | **inside** (but held-out was not predicted, and fell 2.24) |
| R5 | probe 0.762–0.780, held-out 0.765–0.785 | 0.7692 / 0.7781 | **both inside** |
| B9 | probe 0.775–0.790 | **0.7635** | **falsified** |
| C3b | monotone, near 0.78–0.85 | 0.8717 / 0.9463 / 0.9726 | ordering **right**, magnitude direction **wrong** |
| G3 | probe 0.775–0.795, held-out 0.770–0.790 | 0.7740 / 0.7518 | near-miss (0.10 below) / **miss** (1.8 below) |
| H4 | species 0.900–0.912 | **not yet read** | pending (§6) |
| B10 | probe 0.770–0.785, expected to lose | **not yet read** | pending (§6) |

**Five clean hits, three near-misses, four clear misses, two falsifications, two pending.** The two
falsifications (R2, B9) were the most useful runs of the fortnight, which is the argument for
writing the prediction down.

---

## 3. Claims corrected or retracted

Eight, and they are the real output of the period.

1. **The 31-point ArcFace open-set advantage → 0.78 points.** It compared ArcFace's best scoring rule
   against the plain head's worst. Best-vs-best is 0.9068 vs 0.8990, and the plain head is 1 pt
   better on accuracy. [[2026-08-06-the-arcface-open-set-claim-was-a-rule-comparison]]
2. **"Marginalisation is consistent by construction" is false.** `max` and `Σ` do not commute over a
   partition. It looked like a theorem, so nobody tested it.
   [[2026-08-01-marginalisation-is-not-argmax-consistent]]
3. **Self-training's mechanism is not "target-domain pixels".** I asserted a representation-learning
   story on 3 August; T2 showed 83 % of it is available at a frozen classifier.
4. **The cosine head's rows are not unit-norm** (mean 1.08 / 1.77). Downgraded to a documentation
   problem — no accuracy number is affected — but the head's docstring was wrong.
5. **`dev/059` was scoring pre-clamp values**, monotone-safe for `max` and wrong for `entropy`. Caught
   because a "corrected" number exactly matched a stale one from a different script.
6. **C3b's −0.38 pt "cost of holding out data" does not exist.** Different species denominators;
   macro-F1 does not decompose over subsets. The matched control put it at **0.04 pt**.
7. **The staged-vs-end-to-end trade is capacity-dependent, not a property of the method** — and I got
   this wrong in *both* directions on consecutive days. On 9 August B9 made me write that the trade
   dissolves; on 10 August G3 showed it dissolves only at 20 M and holds at 198 M.
8. **START-HERE finding 7 was narrowed three times in three days** as coverage → coverage + balance →
   balance only on a frozen trunk → and only at small scale.

The pattern across 5, 6 and 7 is one thing: **a claim measured in one configuration, or at one
capacity, asserted generally.** `docs/design-decisions.md` gained a companion rule to "change one
factor per run": *measure at two capacities before claiming a method property.*

---

## 4. The current recipe and the numbers behind it

**20 M, each regime at its own best configuration:**

| | in-dist | probe | held-out |
|---|---|---|---|
| staged (R5: clean repr + cRT + adaptation, balanced, frozen trunk) | **0.9074** | 0.7692 | **0.7781** |
| end-to-end (B3rep5x, natural pseudo-labels) | 0.9003 | **0.7706** | 0.7704 |

**198 M:**

| | in-dist | probe | held-out |
|---|---|---|---|
| staged (G3, balanced) | **0.9138** | 0.7740 | 0.7518 |
| staged (G2, natural) | **0.9150** | 0.7648 | 0.7600 |
| end-to-end (B8, natural) | 0.9060 | **0.7798** | **0.7816** |

**Read it as:** the staged recipe reliably buys **+0.7 to +0.9 pt in-distribution** at a fraction of
the training cost and is re-runnable per deployment without labels. Whether it *also* matches
end-to-end under shift depends on capacity — at 20 M it does, at 198 M it does not. That is the
honest headline and it is weaker than what I wrote on 9 August.

Noise floors, for reading any of the above: species in-distribution 0.0000, probe 0.0041, held-out
0.0052.

---

## 5. What was built

| | what |
|---|---|
| `dev/065`, `dev/066` | pseudo-labelling and merging; `--balance` added for R5 |
| `dev/072_holdout_common.py` | the C3b split: withhold **common** taxa at three ranks |
| `dev/073_proxy_free.py` | H4's head — species prototypes as an EMA buffer, no trained matrix |
| `dev/074_figures.py` | four paper figures, numbers transcribed from journal entries with sources cited |
| `dev/060_doc_health.py` | doc hygiene, wired into CI |
| `CLAUDE.md` | the agent operating manual |

`paper/DRAFT.md` gained §1b related work (citations marked `[VERIFY]` — written from memory, needs
your fact-check), §3.1 the baseline recipe, and rewrites of §4.3, §4.4 and §4.13.

---

## 6. What is blocked, and it needs you

**The UCloud refresh token expired during the 14 idle days.** `ucloud q ls` still reports from the
local queue database — all jobs show DONE — but `ucloud q logs` fails with:

```
AuthError: UCloud rejected the refresh token (expired or invalid).
Log in again in the browser and run `ucloud login` with a fresh token.
```

So **two completed runs have never been read**:

- **H4** (proxy-free head — Group H's last training candidate). Trained cleanly for 5 epochs;
  validation f1_species was 0.8689, but that is the *validation* fold and the 0.9148 baseline is the
  *test* fold, so it is not a score. Predicted 0.900–0.912, falsified below 0.885.
- **B10** (198 M end-to-end, balanced). Predicted to **lose** to B8's 0.7798 — the prediction was
  rewritten on evidence from B9 *before* the run started, and the reason is in the config header.

Neither changes §4's conclusions — B10 can only raise end-to-end's best, and H4 is a memory result,
not an accuracy one — but both are paid-for numbers sitting unread.

**Also:** the cron entry that ticks `ucloud q` every 5 minutes is still installed and did keep the
queue advancing after I stopped. No daemon process is running, which is expected — the cron is the
mechanism.

---

## 7. Backlog, ordered

| # | work | cost | why |
|---|---|---|---|
| 1 | `ucloud login`, then read H4 and B10 | minutes | two paid-for results unread |
| 2 | **decide the staged-vs-end-to-end question** (see §8) | — | it has flipped twice; it is a framing call |
| 3 | seed-repeat of G3 | ~1 h at 198 M | the held-out drop at 198 M is n = 1, and two conclusions have already moved on single measurements |
| 4 | **P1 — BioCLIP-2 as a frozen trunk** | build + ~1 h | you deferred it; reviewers will ask. T2b is what makes it plausible: if adaptation needs nothing from the representation, it does not need *ours* |
| 5 | paper: fold the four figures into `DRAFT.md`, fact-check `[VERIFY]` citations | — | the citations are written from memory and are not reliable |
| 6 | C3b at 198 M | ~6 h | the novelty result is currently 20 M only |

**Do not do** — closed for a reason: LDAM and background suppression (T2b showed adaptation subsumes
that family); re-tuning the ArcFace margin (two cheap proxies failed for principled reasons, and the
margin is worth ~0.8 pt); the autoregressive head (lost by 20 pt); uniform sampled softmax at scale
(no plateau); more in-distribution accuracy (saturated).

---

## 8. The one open decision

I asked this on 10 August and there was nobody to answer, so it is still open.

The staged-vs-end-to-end story has consumed five runs and flipped twice in three days. It is a
cost-and-deployability question, not the project's stated subject (reliable prediction that knows
what it doesn't know).

**Freeze it as "capacity-dependent" and move remaining compute to the open-set/abstention side, or
keep pushing it?** My recommendation is **freeze**: it is adequately characterised, the pivot says
the interesting axis is elsewhere, and item 3 above (one cheap seed repeat) is enough to make the
198 M claim safe.
