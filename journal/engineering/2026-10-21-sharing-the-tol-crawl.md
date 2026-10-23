# Sharing the ToL crawl with the workstation's other users

**Kind:** infrastructure · **Status:** **RESOLVED (2026-10-23).** `/data/datasets/tol256` holds
65,203,076 images (2.0 TB) in gbifxdl style: one 29-column `metadata.parquet` and
`images/<Genus_epithet>/<uuid>.jpg`. Its `README.md` is the user-facing doc.

**How.** `dev/090` runs the steps in order:
1. `snapshot` merges the retry metas into the main ones; the latest `ok` wins.
2. The `dev/085` audit.
3. `plan` and `apply` rename the slug folders to exact species keys and split the 148 shared
   ones. Every move is journalled in `tol256s/share_build/applied.tsv`.
4. `recover`: 52,495 images on disk had no meta, because `dev/082` writes a meta only when a part
   completes, and these parts were unfinished when `retry1`/`retry3` stopped. A new
   `082 fetch --offline` (no requests) recorded 51,881 of them; the other 614 stay in
   `tol256s/orphans/`.
5. `enrich` joins in every field ToL publishes per image (catalog: kingdom to class, names,
   publisher, basis of record; provenance: licence and rights holder). Rows from our own GBIF
   download take the same fields from `global_lepi`'s parquet. All 65.2 M rows matched one
   source. Occurrence-level data (date, place, recorder) is not in ToL and was not fetched.

**Numbers.** 202,269 species folders. 567,116 rows carry `exclude`: 548,584 test fold, 15,237
pixel duplicates, 2,856 duplicate URLs, 396 same occurrence, 43 placeholders. 96 % have a licence
name.

**What went wrong.**
- **The pandas audit needed ~130 GB at 94 M rows.** Three attempts set off systemd-oomd, which
  killed the owner's tmux sessions; a `MemoryMax` cap did not help (CLAUDE.md §3). The fix was to
  shard by key hash (`dev/085`, `dev/090`). Peak memory is now under 3 GB, and the output is
  identical to the in-memory version on 7.35 M real rows and on synthetic cases.
- **Two pyarrow traps.** Open `ParquetWriter`s hold their data until closed, and
  `ParquetFile.iter_batches` retained ~95 MB per million rows on one 6 GB file. Both are replaced:
  one file per flush, and row-group reads.
- **Page cache.** Reading 27 GB of ToL metadata filled the page cache, and reclaiming it showed up
  as pressure. Files are now released after use (`drop_cache`).

**Upload.** The 28 slug-path shards on UCloud are in the drive's Trash. `dev/087` is re-planned
from the new paths: 1,053 shards, not yet uploaded (UCloud storage).
