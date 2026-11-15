"""Train the whole-sign classifier.

Learns from data/gislr_clips.npz (made by build_dataset.py) plus any signs you
recorded yourself with record_signs.py, and writes models/sign_clf.joblib.

  python train.py            # honest accuracy check, then train on everything
  python train.py --no-eval  # skip the check (faster)

The accuracy check holds out whole signers: the model is scored on people it
has never seen, which is the situation it faces with you. (Scoring on clips
from signers it trained on gives numbers 20+ points too high.)
"""

from __future__ import annotations

import argparse
import time

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

import config
import signfeat
from recognizer import MODEL_KIND

# Signers held out for the accuracy check (4 of the dataset's 21).
HOLDOUT_SIGNERS = [2044, 27610, 37055, 53618]
AUG_PER_CLIP = config.AUG_PER_CLIP
# Your own recordings are few, so each one is varied many more times.
AUG_PER_CUSTOM = 40


def load_gislr():
    if not config.GISLR_CLIPS.exists():
        raise SystemExit(
            f"{config.GISLR_CLIPS} not found. Run `python build_dataset.py` first "
            "(needs the unpacked Kaggle data in data/asl-signs/)."
        )
    d = np.load(config.GISLR_CLIPS)
    off = np.concatenate([[0], np.cumsum(d["lengths"])])
    lh, rh, pose = (d[k].astype(np.float32) for k in ("lh", "rh", "pose"))
    clips = [
        (lh[a:b], rh[a:b], pose[a:b], signfeat.GISLR_ASPECT)
        for a, b in zip(off[:-1], off[1:])
    ]
    labels = d["labels"].astype(str)
    keep = [i for i, label in enumerate(labels) if label not in config.EXCLUDED_SIGNS]
    return [clips[i] for i in keep], labels[keep], d["participants"][keep]


def load_custom():
    clips, labels = [], []
    if config.CUSTOM_SIGNS_DIR.is_dir():
        for label_dir in sorted(p for p in config.CUSTOM_SIGNS_DIR.iterdir() if p.is_dir()):
            for f in sorted(label_dir.glob("*.npz")):
                d = np.load(f)
                clips.append((d["lh"], d["rh"], d["pose"], float(d["aspect"])))
                labels.append(label_dir.name)
    return clips, np.array(labels, dtype=str)


def featurize(clips, labels, n_aug: int, seed: int = 0):
    """One plain + n_aug varied feature vectors per clip."""
    rng = np.random.default_rng(seed)
    X, y = [], []
    for clip, label in zip(clips, labels):
        base = signfeat.segment_features(*clip)
        if base is None:
            continue
        X.append(base)
        y.append(label)
        for _ in range(n_aug):
            varied = signfeat.segment_features(*clip, rng=rng)
            if varied is not None:
                X.append(varied)
                y.append(label)
    return np.stack(X), np.array(y)


def make_model() -> Pipeline:
    """Shared scaling + PCA, then a small committee of MLPs voting together."""
    members = [
        (
            f"mlp{seed}",
            MLPClassifier(
                hidden_layer_sizes=config.MLP_HIDDEN,
                alpha=config.MLP_ALPHA,
                early_stopping=True,
                max_iter=config.MLP_MAX_ITER,
                random_state=seed,
            ),
        )
        for seed in range(config.ENSEMBLE_SIZE)
    ]
    return make_pipeline(
        StandardScaler(),
        PCA(config.PCA_COMPONENTS, random_state=0),
        StandardScaler(),
        VotingClassifier(members, voting="soft", n_jobs=config.ENSEMBLE_SIZE),
    )


def evaluate(clips, labels, signers) -> None:
    test = np.isin(signers, HOLDOUT_SIGNERS)
    tr = np.where(~test)[0]
    te = np.array([i for i in np.where(test)[0] if labels[i] != "_other"])
    t = time.time()
    Xtr, ytr = featurize([clips[i] for i in tr], labels[tr], AUG_PER_CLIP)
    Xte, yte = featurize([clips[i] for i in te], labels[te], 0)
    model = make_model().fit(Xtr, ytr)
    proba = model.predict_proba(Xte)
    classes = model.classes_
    pred = classes[proba.argmax(1)]
    top3 = np.mean([yte[i] in classes[np.argsort(-proba[i])[:3]] for i in range(len(yte))])
    sure = (proba.max(1) >= config.SIGN_CONFIDENCE) & (pred != "_other")
    print(f"\nAccuracy on {len(HOLDOUT_SIGNERS)} signers the model never saw ({time.time() - t:.0f}s):")
    print(f"  right first time:           {(pred == yte).mean():.1%}")
    print(f"  right answer in top 3:      {top3:.1%}")
    print(
        f"  when confident enough to add a sign ({sure.mean():.0%} of signs): "
        f"{(pred[sure] == yte[sure]).mean():.1%} right"
    )
    per = sorted((float(np.mean(pred[yte == c] == c)), c) for c in np.unique(yte))
    print("  hardest signs: " + ", ".join(f"{c} {a:.0%}" for a, c in per[:8]))
    print("  easiest signs: " + ", ".join(f"{c} {a:.0%}" for a, c in per[-8:]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--no-eval", action="store_true", help="skip the held-out accuracy check")
    args = ap.parse_args()

    clips, labels, signers = load_gislr()
    print(f"Dataset: {len(clips)} clips, {len(set(labels)) - 1} signs (+ '_other')")
    if not args.no_eval:
        evaluate(clips, labels, signers)

    t = time.time()
    X, y = featurize(clips, labels, AUG_PER_CLIP)
    custom_clips, custom_labels = load_custom()
    if len(custom_clips):
        counts = {str(c): int((custom_labels == c).sum()) for c in np.unique(custom_labels)}
        print(f"Your recordings: {counts}")
        Xc, yc = featurize(custom_clips, custom_labels, AUG_PER_CUSTOM, seed=1)
        X, y = np.concatenate([X, Xc]), np.concatenate([y, yc])
    print(f"\nTraining on everything ({len(X)} examples, {len(set(y))} classes) ...")
    model = make_model().fit(X, y)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"kind": MODEL_KIND, "pipeline": model}, config.CLASSIFIER_PATH)
    print(f"Saved classifier → {config.CLASSIFIER_PATH} ({time.time() - t:.0f}s)")


if __name__ == "__main__":
    main()
