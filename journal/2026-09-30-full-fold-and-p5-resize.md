# Full-fold scores for the public releases, and a preprocessing bug in P5's release

**Kind:** subproject · **Status:** **RESOLVED (2026-09-30).** With the GPU back, every published
file was scored on the **full GBIF test fold** (629,742 images, 12,041 species) instead of a 10,000
image sample. That exposed a real flaw in the first P5 release, now fixed: **its documented
preprocessing did not match training, and cost 0.3-0.4 pt on photos and 1.1 pt on trap images.**
The P5 files on the Hub now carry the missing step inside the graph. Follow-up to
[[2026-09-29-public-hf-release]].

## Result: full fold, published fp32 files

| | full fold | top-1 | ToL-clean subset (217,856) | lepinet pipeline |
|---|---|---|---|---|
| P5, first release (224 px bicubic) | 0.9175 | 94.3 % | 0.9072 | clean: 0.9113 |
| **P5, fixed (256 px bilinear, resample in graph)** | **0.9210** | **94.6 %** | **0.9104** | clean: 0.9113 |
| B8 | 0.9059 | 94.5 % | 0.8988 | full: **0.9060** |
| B3rep5x | 0.8985 | 92.5 % | 0.8885 | never measured |

B8's file reproduces its training-pipeline score to 0.0001, so on photos an ONNX export plus the card's
preprocessing is exact. P5's was 0.41 pt low on the same 217,856 images, with 97.75 % per-image
agreement with lepinet's own predictions. That is a real difference, not noise.

## Cause

P5 was trained and evaluated with `aug_img_size 256, img_size 224`. fastai resizes the item to 256
(PIL, bilinear) and then **resamples it to 224 on the GPU, bilinear, without antialiasing**. The card
told users to resize straight to 224 with bicubic, which gives different pixels. Tested on 20,000 clean
photos: straight to 224 gave 0.9183 (97.8 % agreement with lepinet), and the two-stage resize gave
0.9213 (**99.2 %**, lepinet 0.9207). B8 does not need it: its card preprocessing already agrees 99.1 %
with lepinet. B3rep5x gains +0.2 pt from it, not enough to justify asking users for 460 px input.

**Fix:** `dev/083 export --input-size 256 --resample bilinear` adds that resample to the graph
(`F.interpolate(..., antialias=False)` before normalisation; the dynamic H/W input). The transformers
model does the same in `forward`, and the processor now produces 256 px bilinear crops. **224 px input
still works unchanged:** a same-size bilinear resample is the identity, and the new graph matched the
old one to max |d| 0.0 at 224. `predict.py` and `calibrate`/`eval` read `input_size`/`resample` from
`config.json`.

## The fixed P5, re-evaluated end to end

| | first release | fixed | lepinet |
|---|---|---|---|
| probe | 0.7723 | **0.7829** | 0.7810 |
| probe held-out species | 0.7897 | 0.7852 | 0.7806 |
| per-image agreement with lepinet (probe / clean) | 96.4 % / 97.8 % | **98.7 % / 99.1 %** | |
| useful-answer rate, 95 % policy, held-out nights | 83.3 % | **86.4 %** | |
| unseen species: wrongly commits to a species | 13.7 % | 16.9 % | |
| novelty AUROC (entropy) | 0.915 | 0.917 | |

int8 and fp16 track fp32 within 0.2 / 0.1 pt on all three sets. Speed is unchanged (GPU fp16 771
img/s, CPU int8 66 ms per image).

**A correction to [[2026-09-29-public-hf-release]].** That entry explained P5's probe gap (-0.87)
as the up-sampling of tiny trap crops, with the gap "inside the noise floor". For P5 that was wrong:
the gap was this preprocessing mismatch, and the fix closes it (0.7829 vs 0.7810, 98.7 % agreement).
The up-sampling account still stands for B3rep5x, whose two-stage test recovered only half its gap.
**The fix also changes the ranking among the releases:** P5 now leads B8 on every set, by 1.5 pt on
photos, which is clear; on trap images the lead is within noise. The cards say so.

**Trade-off, reported as found:** the better-preprocessed model is more confident. It answers at
species more often on known photos (84.0 % vs 80.6 %) and also commits more often on unseen species
(16.9 % vs 13.7 %). Precision among answered stays at 96 %, and P5 still commits least of the three
(B8 20.9 %, B3rep5x 25.2 %).

**Lesson for any future release:** reproduce the *validation* transform exactly, including a
resample that happens on the GPU inside the framework. The test that caught it was per-image
agreement with the training pipeline on the full fold, which is cheap and should be run for every
file before publishing: macro-F1 alone looked fine (0.4 pt from the reference).
