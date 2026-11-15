"""Train a tiny synthetic classifier so call-mode can start before real samples exist.

Not for demo accuracy — replace with `python train.py` after collect/GISLR data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402


def main() -> None:
    rng = np.random.default_rng(0)
    labels = list(config.VOCABULARY)
    feat_dim = config.FEATURE_DIM * 4  # window features: last/mean/delta/std
    xs: list[np.ndarray] = []
    ys: list[str] = []
    for i, label in enumerate(labels):
        base = np.zeros(feat_dim, dtype=np.float32)
        base[i % feat_dim] = 1.0
        for _ in range(12):
            noise = rng.normal(0.0, 0.05, size=feat_dim).astype(np.float32)
            xs.append(base + noise)
            ys.append(label)

    X = np.stack(xs)
    y = np.array(ys)
    encoder = LabelEncoder()
    y_enc = encoder.fit_transform(y)
    clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(64,),
                    max_iter=200,
                    random_state=0,
                ),
            ),
        ]
    )
    clf.fit(X, y_enc)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": clf, "label_encoder": encoder}, config.CLASSIFIER_PATH)
    print(f"Wrote smoke classifier → {config.CLASSIFIER_PATH}")
    print("Replace with a real model via collect/train or GISLR when ready.")


if __name__ == "__main__":
    main()
