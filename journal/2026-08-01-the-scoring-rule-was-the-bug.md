# Most of "scale hurts open-set" was the scoring rule (E2)

**Kind:** research · **Status:** **RESOLVED (2026-08-01) — and it overturns a claim this project
made confidently two days ago.** Switching from `max-logit` to `max-softmax-probability` recovers
**+6.05 pt** of AUROC on A2 and **+7.61 pt** on B4, while *costing* ~1 pt on the two small models.
The capacity penalty on novelty detection shrinks from **8.8 pt to 1.6 pt**. About four fifths of
what was reported as a property of the embedding was a property of the readout.

## The claim being corrected

[[2026-07-31-best-model-is-not-the-best-model]] stated that scale *hurts* open-set detection
(0.9068 at 20 M → 0.8298 at 198 M), proposed a mechanism (higher capacity fits known taxa more
tightly and so projects novel inputs onto known prototypes with more confidence), and treated B4's
0.8132 as confirmation because it fell *further* from the 0.88 falsification line rather than above
it. [[2026-08-01-capacity-x-augmentation]] then built on it: "every intervention that buys accuracy
costs novelty detection."

Every one of those AUROCs used a single scoring rule — `-max_logit` — inherited from `dev/052` and
never questioned. **The falsification criterion was set on the wrong axis.** It asked whether the
effect was real; it should also have asked whether the *measurement* was.

## Result

`dev/061`, five rules computed in one forward pass over the same images and the same embeddings.

| model | params | `max` | `energy` | `entropy` | `margin` | **`msp`** | best − max |
|---|---|---|---|---|---|---|---|
| A1 | 20 M | **0.9068** | 0.9064 | 0.9047 | 0.8979 | 0.8953 | — |
| B1 | 20 M | **0.9010** | 0.9005 | 0.9008 | 0.8945 | 0.8917 | — |
| A2 | 198 M | 0.8298 | 0.8287 | 0.8813 | 0.8807 | **0.8904** | **+0.0605** |
| B4 | 198 M | 0.8132 | 0.8118 | 0.8802 | 0.8789 | **0.8893** | **+0.0761** |

**The best rule is not the same at both scales.** At 20 M, `max-logit` wins and `msp` is ~1 pt worse.
At 198 M the ranking flips completely and `msp` wins by 6–7.6 pt.

### The capacity penalty, re-measured with each model's best rule

| | A1 (20 M) | A2 (198 M) | penalty |
|---|---|---|---|
| as reported on 07-31 (`max` everywhere) | 0.9068 | 0.8298 | **−7.70** |
| best rule per model | 0.9068 | 0.8904 | **−1.64** |

The effect does not vanish — bigger is still slightly worse at novelty — but it is **4.7× smaller
than published**, and small enough to sit near the region where an unmeasured AUROC noise floor
matters. "Scale hurts open-set" should be downgraded from a headline finding to a weak trend.

## Why the rule flips with scale

`msp` is `max` normalised by the log-partition function: to within the softmax,
$\mathrm{msp} \approx z_{\max} - \operatorname{logsumexp}(z)$. So `max` asks *"how strongly does the
best prototype match?"* and `msp` asks *"how much better than the alternatives?"*

Those come apart exactly when absolute scores stop being informative. The 198 M model has a
better-fitted embedding, so **everything** — known and novel alike — attains a high cosine to
*some* prototype; the absolute maximum saturates and stops discriminating. What still separates them
is *shape*: for a known species one prototype dominates, while for a novel one several score
similarly. Only the normalised rules see that. The 20 M model has not saturated, so magnitude still
carries the signal there and normalising away throws it out.

That `energy` tracks `max` almost exactly at both scales (within 0.15 pt everywhere) is the
consistency check: `energy` = `logsumexp` is dominated by its largest term for peaked logits, so it
is nearly a monotone function of `max` and cannot rank differently. The rules that gain — `msp`,
`entropy`, `margin` — are precisely the three that use the vector's *shape*.

**Generalisable form:** as a classifier's fit improves, open-set signal migrates from the
*magnitude* of the top score to its *dominance* over the rest. A scoring rule chosen on a small
model does not transfer to a large one.

## What this changes

**Revised, with each model's best rule:**

| model | params | in-distribution | shifted | open-set |
|---|---|---|---|---|
| A1 | 20 M | 0.9035 | 0.6437 | **0.9068** |
| B1 | 20 M | 0.8999 | 0.6836 | 0.9010 |
| A2 | 198 M | **0.9216** | 0.6616 | 0.8904 |
| B4 | 198 M | **0.9216** | **0.7101** | 0.8893 |

**The "ranking inversion" is much weaker than claimed.** B4 now ties the best in-distribution score,
wins the shifted benchmark by 2.65 pt over the next model, and gives up only **1.75 pt** of AUROC to
the best. It is not a model that "wins the headline and loses deployment" — it is close to the best
model on all three axes, and the sharp trade-off reported on 07-31 was substantially manufactured by
the readout.

What *survives* from that entry: the axes still disagree in ordering (A1 leads open-set, B4 leads the
other two), in-distribution macro-F1 still should not be the sole selection criterion, and B1 remains
the right choice when size matters. What does **not** survive is the strength of the claim — "the
best in-distribution model is the *worst* deployable one" is no longer supported.

**E1 is cancelled.** The plan was ~36 GPU-hours re-tuning the ArcFace margin at 198 M to recover an
8.8 pt loss. Two thirds of that loss was never in the model, and a 5-minute rescoring found it. This
is the concrete payoff of the ordering rule written into `PLAN.md`: **check the measurement before
buying the experiment.**

## The methodological lesson, which is the durable part

This project already had a rule for this — *"suspect the measurement before the model"*
(`docs/design-decisions.md` §4), learned three times over from the eval-set filter, the metric
mismatch, and fastai's `num_workers`. It was not applied here, because `-max_logit` did not look like
a measurement choice. It looked like *the definition of the score*.

That is the generalisable trap: **a default that arrives with the tooling stops being visible as a
decision.** `dev/052` chose max-logit on day one for a 20 M model and every subsequent open-set
number in this project inherited it silently, across a 10× change in model scale, without anyone
recording that a choice had been made.

**Adopted:** open-set results are reported **with the rule named**, and `dev/061` (all five rules,
one pass) replaces `dev/052` as the default open-set tool. Reporting one number without its rule is
now the same category of error as reporting macro-F1 without saying which level.

---

## Addendum: two rules are *degenerate* for log-probability heads (2026-08-01)

Rescoring A4 (`marginal_arcface`) produced a table that looked alarming and was actually empty:

```
entropy  0.8950   <- best
margin   0.8871
max      0.8591
msp      0.8591          <- identical to max, to four decimals
energy   0.4399          <- apparently catastrophic
```

Neither oddity is a result. The `marginal*` heads emit **log-probabilities**, not raw z-scores,
because the marginals are computed by `logsumexp` inside `forward` and must live on a normalised
scale for that sum to mean anything. On log-probabilities:

- $\mathrm{energy} = \operatorname{logsumexp}(\log p) = \log \sum_j p_j = \log 1 = \mathbf{0}$ for
  every image. A **constant**, so its "AUROC" is whatever the tie-breaking in the rank sort produces.
  0.4399 is noise, not a failure.
- $\mathrm{msp} = \operatorname{softmax}(\log p) = p$, so it is a strictly monotone function of
  $\max_j \log p_j$ and **must** give bit-identical AUROC to `max`. It did.

Both were predictable from the head's output convention, and neither was predicted — the script
applied five rules without asking whether all five were defined for the model in front of it.

**Fixed in `dev/061`:** degeneracy is now detected *from the data* (`logsumexp` identically zero
implies normalised rows) rather than from the head's name, so a future head with the same convention
is caught too. Degenerate rules are printed with an explanation and excluded from the "best rule"
ranking.

**A4's real open-set numbers**, with the degenerate rules removed: **entropy 0.8950**, margin 0.8871,
max/msp 0.8591. So the best rule differs *again* — `max` for the small raw-z-score models, `msp` for
the large ones, `entropy` for the log-probability heads. That strengthens rather than weakens this
entry's conclusion: **the scoring rule is a per-model choice, not a project-wide constant**, and it
now depends on the head's output convention as well as on capacity.

**The recurring shape.** This is the third time this week that something arriving as a default
escaped scrutiny: `-max_logit` as "the" score, "consistent by construction" as a definition, and now
five rules applied without checking they were all defined. Each was caught by taking the claim
literally and testing it.
