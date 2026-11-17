import argparse
import hashlib
import json
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_split(path, name):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{name} file not found: {path}")
    df = pd.read_csv(path)
    required = {"Query", "Intent"}
    if not required.issubset(df.columns):
        raise ValueError(f"{path} must contain columns {sorted(required)}")
    if df["Query"].isna().any() or df["Intent"].isna().any():
        raise ValueError(f"{path} contains missing Query/Intent")
    df["Query"] = df["Query"].astype(str).str.strip()
    df["Intent"] = df["Intent"].astype(str).str.strip()
    if (df["Query"] == "").any() or (df["Intent"] == "").any():
        raise ValueError(f"{path} contains blank Query/Intent")
    print(f"{name}: {len(df)} rows, {df['Intent'].nunique()} intents")
    return df


def cache_name(texts, split, model_name):
    joined = "\n".join(texts).encode("utf-8")
    digest = hashlib.sha256(joined).hexdigest()[:16]
    safe = model_name.split("/")[-1].replace("-", "_")
    return f"{split}_{safe}_{digest}.npy"


def embed_split(encoder, texts, split, cache_dir, batch_size, device):
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / cache_name(list(texts), split, MODEL_NAME)
    if cache_path.exists():
        print(f"Loading cached {split} embeddings: {cache_path.name}")
        return np.load(cache_path)
    print(f"Encoding {split} queries...")
    t0 = time.perf_counter()
    vectors = encoder.encode(
        list(texts), batch_size=batch_size, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True, device=device
    )
    np.save(cache_path, vectors)
    print(f"{split} embeddings: {vectors.shape}; time={time.perf_counter()-t0:.2f}s")
    return vectors


def get_metrics(y_true, y_pred, labels):
    pm, rm, fm, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    pw, rw, fw, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(pm), "macro_recall": float(rm),
        "macro_f1": float(fm), "weighted_precision": float(pw),
        "weighted_recall": float(rw), "weighted_f1": float(fw)
    }


def main():
    ap = argparse.ArgumentParser(description="Train MiniLM embedding + Logistic Regression intent classifier.")
    ap.add_argument("--train", default="data/processed/intent/train.csv")
    ap.add_argument("--validation", default="data/processed/intent/validation.csv")
    ap.add_argument("--test", default="data/processed/intent/test.csv")
    ap.add_argument("--out", default="outputs/intent_minilm")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = ap.parse_args()

    out = Path(args.out)
    cache_dir = out / "embeddings"
    out.mkdir(parents=True, exist_ok=True)

    train = load_split(args.train, "Train")
    val = load_split(args.validation, "Validation")
    test = load_split(args.test, "Test")
    train_labels = set(train["Intent"].unique())
    if set(val["Intent"].unique()) != train_labels or set(test["Intent"].unique()) != train_labels:
        raise ValueError("Intent classes differ between splits; verify taxonomy.")
    labels = sorted(train_labels)

    import torch
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
        device = args.device
    print(f"Embedding model: {MODEL_NAME}")
    print(f"Device: {device}")
    encoder = SentenceTransformer(MODEL_NAME, device=device)

    e0 = time.perf_counter()
    X_train = embed_split(encoder, train["Query"], "train", cache_dir, args.batch_size, device)
    X_val = embed_split(encoder, val["Query"], "validation", cache_dir, args.batch_size, device)
    X_test = embed_split(encoder, test["Query"], "test", cache_dir, args.batch_size, device)
    embedding_seconds = time.perf_counter() - e0

    # Labels are mapped to stable integers for classifier training.
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train["Intent"])
    y_val = label_encoder.transform(val["Intent"])
    y_test = label_encoder.transform(test["Intent"])
    numeric_labels = np.arange(len(label_encoder.classes_))

    candidates = [
        {"name": "minilm_logistic_regression_C1", "C": 1.0},
        {"name": "minilm_logistic_regression_C4", "C": 4.0},
    ]
    rows, trained = [], {}
    for candidate in candidates:
        clf = LogisticRegression(C=candidate["C"], max_iter=3000, solver="lbfgs", random_state=42)
        print(f"\nTraining {candidate['name']}...")
        t0 = time.perf_counter()
        clf.fit(X_train, y_train)
        train_seconds = time.perf_counter() - t0
        pred_val = clf.predict(X_val)
        m = get_metrics(y_val, pred_val, numeric_labels)
        # Encoder latency is measured separately from classifier latency.
        sample = list(val["Query"].iloc[:min(100, len(val))])
        encoder.encode(sample[:1], convert_to_numpy=True, normalize_embeddings=True, device=device)
        t0 = time.perf_counter()
        sample_vec = encoder.encode(sample, batch_size=args.batch_size, show_progress_bar=False,
                                    convert_to_numpy=True, normalize_embeddings=True, device=device)
        encoder_batch_seconds = time.perf_counter() - t0
        t0 = time.perf_counter()
        clf.predict(sample_vec)
        classifier_batch_seconds = time.perf_counter() - t0
        row = {
            "model": candidate["name"], "C": candidate["C"],
            "classifier_train_seconds": round(train_seconds, 4),
            "validation_accuracy": m["accuracy"], "validation_macro_precision": m["macro_precision"],
            "validation_macro_recall": m["macro_recall"], "validation_macro_f1": m["macro_f1"],
            "validation_weighted_f1": m["weighted_f1"],
            "encoder_batch_latency_ms_per_query_approx": round(encoder_batch_seconds / len(sample) * 1000, 4),
            "classifier_latency_ms_per_query_approx": round(classifier_batch_seconds / len(sample) * 1000, 4),
            "embedding_device": device
        }
        rows.append(row)
        trained[candidate["name"]] = clf
        print(json.dumps(row, indent=2))

    comparison = pd.DataFrame(rows).sort_values(
        ["validation_macro_f1", "validation_accuracy"], ascending=False
    )
    comparison.to_csv(out / "validation_model_comparison.csv", index=False)
    winner_name = str(comparison.iloc[0]["model"])
    winner = trained[winner_name]
    print(f"\nSelected by validation macro-F1: {winner_name}")

    pred_test = winner.predict(X_test)
    test_m = get_metrics(y_test, pred_test, numeric_labels)
    test_metrics = {
        "selected_model": winner_name,
        "encoder": MODEL_NAME,
        "encoder_frozen": True,
        "test_accuracy": test_m["accuracy"],
        "test_macro_precision": test_m["macro_precision"],
        "test_macro_recall": test_m["macro_recall"],
        "test_macro_f1": test_m["macro_f1"],
        "test_weighted_precision": test_m["weighted_precision"],
        "test_weighted_recall": test_m["weighted_recall"],
        "test_weighted_f1": test_m["weighted_f1"],
        "embedding_generation_seconds_all_splits_first_run": round(embedding_seconds, 3),
        "embedding_device": device
    }
    (out / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
    report = classification_report(
        y_test, pred_test, labels=numeric_labels, target_names=label_encoder.classes_,
        output_dict=True, zero_division=0
    )
    pd.DataFrame(report).T.to_csv(out / "test_classification_report.csv")
    pred_names = label_encoder.inverse_transform(pred_test)
    pd.DataFrame({
        "Query": test["Query"], "True_Intent": test["Intent"],
        "Predicted_Intent": pred_names, "Correct": test["Intent"].to_numpy() == pred_names
    }).to_csv(out / "test_predictions.csv", index=False)

    cm = confusion_matrix(y_test, pred_test, labels=numeric_labels)
    size = max(14, len(labels) * 0.34)
    fig, ax = plt.subplots(figsize=(size, size))
    ConfusionMatrixDisplay(cm, display_labels=label_encoder.classes_).plot(
        ax=ax, xticks_rotation=90, colorbar=False, values_format="d"
    )
    ax.set_title(f"Test Confusion Matrix — {winner_name}")
    fig.tight_layout()
    fig.savefig(out / "test_confusion_matrix.png", dpi=180)
    plt.close(fig)

    joblib.dump(winner, out / "intent_minilm_classifier.joblib")
    joblib.dump(label_encoder, out / "label_encoder.joblib")
    metadata = {
        "encoder": MODEL_NAME, "encoder_frozen": True, "classifier": "LogisticRegression",
        "classes": list(label_encoder.classes_), "input_feature": "Query", "target": "Intent",
        "train_rows": len(train), "validation_rows": len(val), "test_rows": len(test),
        "embedding_device": device, "embedding_generation_seconds_first_run": round(embedding_seconds, 3)
    }
    (out / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("\nFINAL HELD-OUT TEST METRICS")
    print(json.dumps(test_metrics, indent=2))
    print(f"\nArtifacts saved to: {out.resolve()}")
    print("Note: save/retain the encoder name and install sentence-transformers to reproduce inference.")

if __name__ == "__main__":
    main()
