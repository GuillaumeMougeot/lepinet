# Global Lepidoptera on the datasets drive: move the cut tail out of global_lepi/images so that images/ holds only
# what global_lepi_shards-v1 already has (owner, 2026-11-25: one copy per dataset).
#
# The shards hold every row of the quality-filtered parquet (12,632 species, 6,329,994 images, original JPEG bytes).
# The cut tail, the 38,316 species of the postprocessed parquet that the quality-filtered set dropped (441,994 rows),
# exists only in images/; J1's open-set unknowns are read from it. `mv` inside one mount is a rename:
# no copy, no extra space, resumable (a species already moved is skipped). Lists (written from the local parquets,
# uploaded next to the data): cut_tail_species.txt, kept_species.txt. Deleting images/ afterwards is a separate,
# manual step, after the counts below are checked.
set -euo pipefail
G=/work/global_lepi
mkdir -p "$G/cut_tail"
moved=0; already=0; missing=0
while read -r k; do
  if [ -d "$G/images/$k" ]; then
    mv "$G/images/$k" "$G/cut_tail/$k"; moved=$((moved + 1))
  elif [ -d "$G/cut_tail/$k" ]; then
    already=$((already + 1))
  else
    missing=$((missing + 1))
  fi
done < "$G/cut_tail_species.txt"
echo "tail species: moved $moved, already moved $already, no folder $missing"
echo "cut_tail: $(find "$G/cut_tail" -mindepth 1 -maxdepth 1 -type d | wc -l) folders, $(find "$G/cut_tail" -type f | wc -l) files"
find "$G/images" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort > /tmp/left.txt
sort "$G/kept_species.txt" > /tmp/kept.txt
echo "images/ now: $(wc -l < /tmp/left.txt) folders; not a kept species: $(comm -23 /tmp/left.txt /tmp/kept.txt | wc -l)"
comm -23 /tmp/left.txt /tmp/kept.txt | head -20
echo "images/ files: $(find "$G/images" -type f | wc -l)"
