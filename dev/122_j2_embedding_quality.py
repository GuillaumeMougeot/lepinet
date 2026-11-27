"""J2: the quality of the embedding space itself, head set aside (CPU, from e-j2-os's saved embeddings).

    CUDA_VISIBLE_DEVICES="" taskset -c 0-3 nice -n 10 .venv/bin/python -I dev/122_j2_embedding_quality.py [--reps 200]
        [--center | --transform none,preact,whiten,preact_whiten]

--center subtracts the mean embedding of the reference (training) images and renormalises before every measure:
mini_trainer's embedding is a leaky ReLU's output, nearly confined to the positive orthant, so a dense code shares one
large common direction (the owner's question about the learned head's cone, 2026-11-26). Output then goes to
`embedding_quality_centered.json`.

--transform (owner's question whether the leaky ReLU sandbagged the models, 2026-11-26) scores other views of the same
saved embeddings, each fitted on the reference images only and paired against `none` (the embedding as saved):
- `preact` undoes the leaky ReLU. It is positively homogeneous and invertible, so dividing the negative coordinates by
  the slope (0.01) gives exactly the direction of the hidden layer's output before the activation (h = W f + c):
  the signed vector the model computed, of which normalisation kept the negative half at a hundredth.
- `whiten` is within-class whitening: subtract the reference mean, multiply by Sw^(-1/2), Sw the covariance of the
  reference images around their own species' mean (a floor of 1e-4 times the mean eigenvalue), renormalise. Nearest
  class mean then becomes LDA's rule, and nearest neighbour Mahalanobis'; it is the first (within-class) step of the
  learned whitening used on non-negative retrieval descriptors (Mikolajczyk and Matas 2007; Radenovic et al. 2018).
  It asks whether one model's lead is geometry a linear map can undo, or information the other model lacks.
- `preact_whiten` is both.
Output then goes to `embedding_quality_<transforms>.json`, with transform-minus-none differences per arm.

The owner's question (2026-11-26): which of J2's three models (learned head, frozen head, one learned global scale)
has the best embedding space, whatever its head does? Open-set AUROC (dev/119, mt's openset_auroc.py) is one answer.
This adds three more, all computed from the L2-normalised embeddings alone, with prototypes = class means of 10
training images per species (the `ref` set, 126,310 images, every species):

1. Known species (the `known_bal` test-fold images, 12,630 species): nearest class mean and nearest training image
   (1-NN) as classifiers, scored at species, genus and family (the genus or family of the nearest species).
2. Species never trained on (Global Lepidoptera's cut tail): `near` (new species of known genera; 4,000 species) scored
   on the genus of the nearest class mean, `mid` (new genera of known families; 4,000 species) on its family. This asks whether a
   new species lands next to its relatives, which is what an embedding model offers beyond its label set.
3. Taxonomic structure of the class means: mean cosine between two species' means in the same genus, in the same
   family but another genus, and in different families; and the share of species (in genera with two or more trained
   species) whose nearest other class mean is a congener.

For reference, the head's own argmax at each rank on the same images (mt's columns argmax_species/genus/family,
mapped through J2's class_spec, checksum 0bb683a6). All accuracies are macro over species (mean of per-species
accuracy); intervals are a paired bootstrap over species. Output: `<dir>/embedding_quality.json`.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

D = Path("/data/au761367/lepinet/j2/openset")
SPEC = Path("/data/au761367/global_lepi_shards/class_spec.json")
ARMS = ("ctl", "frz", "gs")
# paired comparisons, each made when both arms are scored; j4a/j4b are frz-os/ctl-os without the leaky ReLU (J4)
PAIRS = (("frz", "ctl"), ("gs", "ctl"), ("frz", "gs"), ("j4a", "frz"), ("j4a", "ctl"), ("j4b", "ctl"), ("j4a", "j4b"), ("j4b", "frz"))
NO_RELU = ("j4a", "j4b")
CHUNK = 4096
SLOPE = 0.01        # F.leaky_relu's default negative slope, as in mini_trainer's Classifier.preclassification
FLOOR = 1e-4        # eigenvalue floor of the within-class whitening, relative to the mean eigenvalue; set a priori


def load_embeddings(path):
    x = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(x, dict):
        keys = [k for k, v in x.items() if torch.is_tensor(v) and v.ndim == 2]
        assert len(keys) == 1, f"{path}: expected one 2-D tensor, got {list(x)}"
        x = x[keys[0]]
    return x


def nearest(Q, M):
    """Index of the row of M (float32) with the largest cosine, for every row of Q (float16); both normalised."""
    out = []
    for s in range(0, len(Q), CHUNK):
        out.append((Q[s:s + CHUNK].float() @ M.T).argmax(1))
    return torch.cat(out).numpy()


def transform(E, ref_rows, sp_codes, kind):
    """The saved embedding under `kind`, unit length, float16 like the saved one; anything fitted uses `ref_rows` only."""
    if kind == "none":
        return E
    X = E.float()
    if kind.startswith("preact"):
        X = F.normalize(torch.where(X < 0, X / SLOPE, X), dim=1)
    if kind == "center" or kind.endswith("whiten"):
        X -= X[ref_rows].mean(0)
        if kind.endswith("whiten"):
            R = X[ref_rows]
            mu = torch.zeros(int(sp_codes.max()) + 1, R.shape[1]).index_add_(0, torch.from_numpy(sp_codes), R)
            mu /= torch.from_numpy(np.bincount(sp_codes)).float()[:, None]
            W = R - mu[sp_codes]
            lam, U = torch.linalg.eigh(W.T @ W / (len(W) - len(mu)))
            del R, mu, W
            X = X @ (U / lam.clamp(min=FLOOR * lam.mean()).sqrt())
        X = F.normalize(X, dim=1)
    return X.half()


def geometry(R):
    """Spread of unit vectors R (float32): mean pairwise cosine, effective dimensions, and for a non-negative code its
    sparsity. Mean pairwise cosine is |mean(R)|^2 (exact up to the n self-pairs). Effective dimensions is the
    participation ratio of the centred covariance's eigenvalues, (sum l)^2 / sum l^2: d if the variance is spread
    evenly over d directions, 1 if it is all in one."""
    m = R.mean(0)
    lam = torch.linalg.eigvalsh((R - m).T @ (R - m) / len(R))
    on = R > 1e-3
    use = on.float().mean(0)
    return {"mean_pairwise_cos": float(m @ m), "effective_dims": float(lam.sum() ** 2 / (lam ** 2).sum()),
            "coords_on_per_image": float(on.sum(1).float().mean()), "coords_negative": float((R < 0).float().mean()),
            "energy_in_negative_coords": float((R.clamp(max=0) ** 2).sum(1).mean()),
            "coords_never_on": int((use == 0).sum()), "coords_on_in_1pct_or_more": int((use >= 0.01).sum()),
            "median_share_of_images_per_coord": float(use.median())}


def macro(ok, species):
    """Per-species accuracy (a Series indexed by species), whose mean is the macro accuracy."""
    return pd.Series(ok, dtype=float).groupby(np.asarray(species)).mean()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reps", type=int, default=200)
    p.add_argument("--dir", type=Path, default=D)
    p.add_argument("--arms", default=",".join(ARMS))
    p.add_argument("--center", action="store_true", help="subtract the reference images' mean embedding, renormalise")
    p.add_argument("--transform", default="none", help="comma list of none, center, preact, whiten, preact_whiten")
    p.add_argument("--out", default=None, help="output name in --dir (default: from the transforms and layer)")
    p.add_argument("--layer", default="embeddings", help="emb/<arm>-<layer>.pt; 'pooled' is the pooled trunk feature "
                   "before the hidden layer (J4f), L2-normalised on load")
    a = p.parse_args()
    arms = a.arms.split(",")
    assert not ({"preact", "preact_whiten"} & set(a.transform.split(",")) and set(arms) & set(NO_RELU)), \
        "preact undoes the leaky ReLU; J4a/J4b have none"
    transforms = ["center"] if a.center else a.transform.split(",")
    assert set(transforms) <= {"none", "center", "preact", "whiten", "preact_whiten"}, transforms
    torch.set_num_threads(4)
    rows = pd.read_parquet(a.dir / "rows.parquet")
    uniq = rows.path.drop_duplicates()
    emb_row = pd.Series(np.arange(len(uniq)), index=uniq.to_numpy())[rows.path.to_numpy()].to_numpy()
    spec = json.loads(SPEC.read_text())
    idx2key = [{v: str(k) for k, v in spec["cls2idx"][lvl].items()} for lvl in ("0", "1", "2")]  # keys are strings in mt's tables

    base = pd.read_parquet(a.dir / f"{arms[0]}.parquet", columns=["set", "key", "speciesKey", "genusKey", "familyKey"])
    assert (base.key.to_numpy() == rows.key.to_numpy()).all()
    ref = (base.set == "ref").to_numpy()
    sp_codes, species = pd.factorize(base.speciesKey[ref])
    tax = base[ref].drop_duplicates("speciesKey").set_index("speciesKey").loc[species, ["genusKey", "familyKey"]]
    m_genus, m_family = tax.genusKey.to_numpy(), tax.familyKey.to_numpy()
    sets = {s: (base.set == s).to_numpy() for s in ("known_bal", "near", "mid")}

    per = {}   # (arm, measure) -> per-species accuracy Series
    out = {"n": {s: {"images": int(m.sum()), "species": int(base.speciesKey[m].nunique())} for s, m in sets.items()}}
    variants = []
    # 'none' keeps the bare arm name; so does center when it is the only transform (the older --center output)
    sfx = lambda tr: "" if tr == "none" or transforms == ["center"] else f":{tr}"
    for arm_, tr in [(x, y) for x in arms for y in transforms]:
        arm = arm_ + sfx(tr)
        variants.append(arm)
        if tr == transforms[0]:
            t = pd.read_parquet(a.dir / f"{arm_}.parquet")
            assert (t.key.to_numpy() == rows.key.to_numpy()).all()
            E0 = load_embeddings(a.dir / "emb" / f"{arm_}-{a.layer}.pt")
            assert E0.shape[0] == len(uniq)
            if a.layer != "embeddings":
                assert "preact" not in a.transform, "preact undoes the hidden layer's ReLU; it has no meaning before it"
                E0 = F.normalize(E0.float(), dim=1).half()
        E = transform(E0, emb_row[ref], sp_codes, tr)
        R = E[emb_row[ref]].float()
        M = torch.zeros(len(species), R.shape[1]).index_add_(0, torch.from_numpy(sp_codes), R)
        M = torch.nn.functional.normalize(M, dim=1)
        res = {}
        assert t.speciesKey.map(type).eq(str).all()

        # 1. known species
        k = sets["known_bal"]
        q = E[emb_row[k]]
        cm = nearest(q, M)
        nn = nearest(q, R)
        kt = t[k]
        truth = kt.speciesKey.to_numpy()
        for name, ok in {
            "known_cm_species": species[cm] == truth,
            "known_cm_genus": m_genus[cm] == kt.genusKey.to_numpy(),
            "known_cm_family": m_family[cm] == kt.familyKey.to_numpy(),
            "known_1nn_species": species[sp_codes[nn]] == truth,
            "known_head_species": np.array([idx2key[0][i] for i in kt.argmax_species]) == truth,
            "known_head_genus": np.array([idx2key[1][i] for i in kt.argmax_genus]) == kt.genusKey.to_numpy(),
            "known_head_family": np.array([idx2key[2][i] for i in kt.argmax_family]) == kt.familyKey.to_numpy(),
        }.items():
            per[(arm, name)] = macro(ok, truth)

        # 2. species never trained on
        for s, lvl, col in (("near", 1, "genusKey"), ("near", 2, "familyKey"), ("mid", 2, "familyKey")):
            m = sets[s]
            st = t[m]
            cm = nearest(E[emb_row[m]], M)
            emb_pred = (m_genus if lvl == 1 else m_family)[cm]
            head_pred = np.array([idx2key[lvl][i] for i in st[f"argmax_{'genus' if lvl == 1 else 'family'}"]])
            rank = "genus" if lvl == 1 else "family"
            per[(arm, f"{s}_cm_{rank}")] = macro(emb_pred == st[col].to_numpy(), st.speciesKey.to_numpy())
            per[(arm, f"{s}_head_{rank}")] = macro(head_pred == st[col].to_numpy(), st.speciesKey.to_numpy())

        # 3. taxonomic structure of the class means
        C = (M @ M.T).numpy()
        same_g = m_genus[:, None] == m_genus[None, :]
        same_f = m_family[:, None] == m_family[None, :]
        off = ~np.eye(len(species), dtype=bool)
        res["cos_same_genus"] = float(C[same_g & off].mean())
        res["cos_same_family_other_genus"] = float(C[same_f & ~same_g].mean())
        res["cos_other_family"] = float(C[~same_f].mean())
        np.fill_diagonal(C, -np.inf)
        nb = C.argmax(1)
        multi = pd.Series(m_genus).map(pd.Series(m_genus).value_counts()).to_numpy() >= 2
        res["nearest_mean_is_congener"] = float((m_genus[nb] == m_genus)[multi].mean())
        res["species_in_multi_species_genera"] = int(multi.sum())
        # how tight a species is around its own mean, against the nearest other mean
        own = (R * M[sp_codes]).sum(1).numpy()
        res["cos_image_to_own_mean"] = float(pd.Series(own).groupby(sp_codes).mean().mean())
        res["cos_mean_to_nearest_other_mean"] = float(C.max(1).mean())
        del C
        res.update(geometry(R))
        out[arm] = res
        print(arm, {n: round(float(v.mean()), 4) for (aa, n), v in per.items() if aa == arm}, flush=True)
        print(arm, {n: round(v, 4) if isinstance(v, float) else v for n, v in res.items()}, flush=True)

    names = sorted({n for _, n in per})
    rng = np.random.default_rng(0)
    pairs = [(x + sfx(tr), y + sfx(tr)) for tr in transforms for x, y in PAIRS]
    if "none" in transforms:
        pairs += [(f"{arm}{sfx(tr)}", arm) for tr in transforms if tr != "none" for arm in arms]
    for n in names:
        idx = per[(variants[0], n)].index
        vals = {arm: per[(arm, n)].loc[idx].to_numpy() for arm in variants}
        draws = [rng.integers(0, len(idx), len(idx)) for _ in range(a.reps)]
        out.setdefault("macro", {})[n] = {arm: float(v.mean()) for arm, v in vals.items()}
        for x, y in pairs:
            if x in vals and y in vals:
                d = vals[x] - vals[y]
                boot = [d[i].mean() for i in draws]
                out["macro"][n][f"{x}-{y}"] = [float(d.mean()), *map(float, np.percentile(boot, [2.5, 97.5]))]
    for n in names:
        r = out["macro"][n]
        print(f"{n:22s} " + " / ".join(f"{r[arm]:.4f}" for arm in variants) + "   " + "  ".join(
            f"{k} {v[0]:+.4f} [{v[1]:+.4f}, {v[2]:+.4f}]" for k, v in r.items() if "-" in k), flush=True)
    name = ("embedding_quality_centered.json" if a.center else "embedding_quality.json" if transforms == ["none"]
            else f"embedding_quality_{'_'.join(transforms)}.json")
    if a.layer != "embeddings":
        name = name.replace("embedding_quality", f"embedding_quality_{a.layer}")
    name = a.out or name
    (a.dir / name).write_text(json.dumps(out, indent=1))
    print("wrote", a.dir / name)


if __name__ == "__main__":
    main()
