"""Train an MLP classifier on collected landmark window features."""

from __future__ import annotations

import joblib
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

import config


def load_dataset() -> tuple[np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[str] = []
    if not config.SAMPLES_DIR.exists():
        raise SystemExit(
            f"No samples found at {config.SAMPLES_DIR}. "
            "Run `python collect.py` first."
        )
    for label_dir in sorted(config.SAMPLES_DIR.iterdir()):
        if not label_dir.is_dir():
            continue
        label = label_dir.name
        files = sorted(label_dir.glob("*.npy"))
        if not files:
            print(f"Warning: no samples for '{label}'")
            continue
        for f in files:
            arr = np.load(f)
            xs.append(arr.astype(np.float32).ravel())
            ys.append(label)
    if not xs:
        raise SystemExit(
            f"No .npy samples under {config.SAMPLES_DIR}. "
            "Run `python collect.py` first."
        )
    return np.stack(xs, axis=0), np.array(ys)


def _class_counts(y: np.ndarray) -> dict[str, int]:
    labels, counts = np.unique(y, return_counts=True)
    return {str(label): int(count) for label, count in zip(labels, counts)}


def _print_counts(title: str, counts: dict[str, int]) -> None:
    print(title)
    for label, n in sorted(counts.items()):
        print(f"  {label:<12} {n}")


def balance_classes(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Oversample classes below the median count (repeat + Gaussian noise std 0.02)."""
    counts = _class_counts(y)
    _print_counts("Class counts before balancing:", counts)
    median = int(np.median(list(counts.values())))
    print(f"Median class count: {median}")

    rng = np.random.default_rng(0)
    extras_x: list[np.ndarray] = []
    extras_y: list[str] = []
    for label, n in counts.items():
        if n >= median:
            continue
        need = median - n
        idx = np.where(y == label)[0]
        picks = rng.choice(idx, size=need, replace=True)
        noise = rng.normal(0.0, 0.02, size=(need, X.shape[1])).astype(np.float32)
        extras_x.append(X[picks] + noise)
        extras_y.extend([label] * need)
        print(f"  balancing '{label}': {n} → {median} (+{need})")

    if extras_x:
        X = np.concatenate([X, *extras_x], axis=0)
        y = np.concatenate([y, np.array(extras_y)], axis=0)

    _print_counts("Class counts after balancing:", _class_counts(y))
    return X, y


def augment(X: np.ndarray, y: np.ndarray, copies: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Light Gaussian noise augmentation."""
    extras_x = []
    extras_y = []
    rng = np.random.default_rng(42)
    for _ in range(copies):
        noise = rng.normal(0.0, 0.02, size=X.shape).astype(np.float32)
        extras_x.append(X + noise)
        extras_y.append(y.copy())
    return np.concatenate([X, *extras_x], axis=0), np.concatenate([y, *extras_y], axis=0)


def main() -> None:
    X, y = load_dataset()
    print(f"Loaded {len(X)} samples across {len(set(y))} labels")
    X, y = balance_classes(X, y)
    X, y = augment(X, y, copies=2)
    print(f"After augmentation: {len(X)} samples")

    encoder = LabelEncoder()
    y_enc = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=config.MLP_HIDDEN,
                    max_iter=config.MLP_MAX_ITER,
                    random_state=42,
                    early_stopping=True,
                    n_iter_no_change=20,
                ),
            ),
        ]
    )
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    labels = list(encoder.classes_)
    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            preds,
            labels=list(range(len(labels))),
            target_names=labels,
            zero_division=0,
        )
    )

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"pipeline": clf, "label_encoder": encoder}
    joblib.dump(payload, config.CLASSIFIER_PATH)
    print(f"Saved classifier → {config.CLASSIFIER_PATH}")


if __name__ == "__main__":
    main()
