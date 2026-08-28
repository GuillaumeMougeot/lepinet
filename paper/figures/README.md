# Figures

Regenerate with the `dev/` scripts; the JSON beside each PNG holds the numbers the caption quotes.

| file | source | what it shows |
|---|---|---|
| `fig3_embedding_tsne.png` + `fig3_geometry.json` | `dev/053_embedding_viz.py` | t-SNE of the embedding space, plain cosine vs ArcFace×z-score, 12 species × 40 held-out images. **Deliberately unconvincing**: the panels look alike and silhouette barely moves (0.617→0.641), because separability was never the problem. Included to justify *not* using a projection as the main figure. |
| `fig4_openset_scores.png` + `fig4_openset.json` | `dev/054_openset_viz.py` | Top: novelty score (max cos θ) for known vs novel species — the overlap *is* the error. Bottom: cos to own prototype vs nearest wrong. Plain cosine piles everything near 0 (AUROC 0.607); ArcFace×z-score separates known 0.704 from novel 0.453 (AUROC 0.898). **Read the caveat below before using this as the open-set figure.** |

Both were computed on 1,500 known + 1,500 novel images (566 unseen species) from the *same* image
domain, so they isolate novelty from domain shift. The independently measured AUROCs (0.607 / 0.898)
reproduce the evaluation pipeline's numbers (0.601 / 0.9115) by a separate code path.

> **Caveat (added 2026-08-28).** `fig4` scores **both** heads by max cos θ, which is the plain cosine
> head's *worst* rule by 27 points (paper §4.9). The 0.607-vs-0.898 gap it draws is therefore the
> best-vs-worst comparison that paper §4.3 retracts: best-rule against best-rule the two heads are
> 0.8990 and 0.9068. The figure is a correct picture of *what max-logit sees*, and it is a misleading
> picture of *what the margin is worth*. It should not carry the open-set claim without being
> redrawn per-head-best-rule, or relabelled as a diagnostic of the score's magnitude.
>
> Undocumented here: `fig1_capacity`, `fig2_dose`, `fig5_rank_abstention` (added 2026-08-24).
