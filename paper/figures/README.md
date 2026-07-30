# Figures

Regenerate with the `dev/` scripts; the JSON beside each PNG holds the numbers the caption quotes.

| file | source | what it shows |
|---|---|---|
| `fig3_embedding_tsne.png` + `fig3_geometry.json` | `dev/053_embedding_viz.py` | t-SNE of the embedding space, plain cosine vs ArcFace×z-score, 12 species × 40 held-out images. **Deliberately unconvincing**: the panels look alike and silhouette barely moves (0.617→0.641), because separability was never the problem. Included to justify *not* using a projection as the main figure. |
| `fig4_openset_scores.png` + `fig4_openset.json` | `dev/054_openset_viz.py` | **The main open-set figure.** Top: novelty score (max cos θ) for known vs novel species — the overlap *is* the error. Bottom: cos to own prototype vs nearest wrong. Plain cosine piles everything near 0 (AUROC 0.607); ArcFace×z-score separates known 0.704 from novel 0.453 (AUROC 0.898). |

Both were computed on 1,500 known + 1,500 novel images (566 unseen species) from the *same* image
domain, so they isolate novelty from domain shift. The independently measured AUROCs (0.607 / 0.898)
reproduce the evaluation pipeline's numbers (0.601 / 0.9115) by a separate code path.
