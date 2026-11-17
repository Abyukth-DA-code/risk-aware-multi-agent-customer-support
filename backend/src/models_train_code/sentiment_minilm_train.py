"""Train frozen MiniLM sentence embeddings + Logistic Regression for sentiment v2."""
from __future__ import annotations
import argparse, hashlib, json, time
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder


def load_split(path: Path, name: str, text_col: str, target_col: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{name} file not found: {path}")
    df = pd.read_csv(path)
    missing = [c for c in (text_col, target_col) if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns {missing}; available: {list(df.columns)}")
    if df[[text_col, target_col]].isna().any().any():
        raise ValueError(f"{name} contains missing text or labels")
    df[text_col] = df[text_col].astype(str).str.strip()
    df[target_col] = df[target_col].astype(str).str.strip()
    if (df[text_col] == "").any() or (df[target_col] == "").any():
        raise ValueError(f"{name} contains empty text or labels")
    return df


def normalized(text):
    return " ".join(str(text).lower().split())


def metrics(y_true, y_pred, labels, elapsed=None):
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    wp, wr, wf, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    result = {"accuracy": float(accuracy_score(y_true, y_pred)), "macro_precision": float(p),
              "macro_recall": float(r), "macro_f1": float(f), "weighted_precision": float(wp),
              "weighted_recall": float(wr), "weighted_f1": float(wf)}
    if elapsed is not None:
        result["classifier_inference_ms_per_query"] = float(elapsed * 1000 / max(1, len(y_true)))
    for i, label in enumerate(labels):
        pp, rr, ff, ss = precision_recall_fscore_support(y_true, y_pred, labels=[i], average=None, zero_division=0)
        result[f"{label.lower()}_precision"] = float(pp[0])
        result[f"{label.lower()}_recall"] = float(rr[0])
        result[f"{label.lower()}_f1"] = float(ff[0])
        result[f"{label.lower()}_support"] = int(ss[0])
    return result


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed/sentiment/v2_refined")
    parser.add_argument("--output-dir", default="outputs/sentiment_model2/v2_minilm")
    parser.add_argument("--text-col", default="Query")
    parser.add_argument("--target-col", default="Sentiment")
    parser.add_argument("--model-name", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, help="Optional device: cpu or cuda")
    parser.add_argument("--force-reembed", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    cache_dir = out_dir / "embedding_cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = {s: data_dir / f for s, f in [("train", "train.csv"), ("validation", "validation.csv"), ("test", "test.csv")]}
    dfs = {s: load_split(p, s.title(), args.text_col, args.target_col) for s, p in paths.items()}
    for name, df in dfs.items():
        print(f"{name.title()}: {len(df)} rows, {df[args.target_col].nunique()} classes")
        print("  label distribution:", {str(k): int(v) for k, v in df[args.target_col].value_counts().items()})
    labels_train = set(dfs["train"][args.target_col].unique())
    for s in ("validation", "test"):
        if set(dfs[s][args.target_col].unique()) != labels_train:
            raise ValueError(f"Sentiment classes differ in {s} split")
    query_sets = {s: set(dfs[s][args.text_col].map(normalized)) for s in dfs}
    for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        overlap = query_sets[a] & query_sets[b]
        if overlap:
            raise ValueError(f"Found {len(overlap)} normalized query overlaps between {a} and {b}")

    encoder_labels = LabelEncoder()
    y = {s: encoder_labels.fit_transform(dfs[s][args.target_col]) if s == "train" else None for s in dfs}
    # Transform validation/test with the train-fitted label encoder.
    y["validation"] = encoder_labels.transform(dfs["validation"][args.target_col])
    y["test"] = encoder_labels.transform(dfs["test"][args.target_col])
    class_names = list(encoder_labels.classes_)
    y["train"] = encoder_labels.transform(dfs["train"][args.target_col])
    print("Classes:", class_names)

    encoder_kwargs = {"device": args.device} if args.device else {}
    print(f"\nLoading sentence encoder: {args.model_name}")
    encoder = SentenceTransformer(args.model_name, **encoder_kwargs)

    def embed(split):
        texts = dfs[split][args.text_col].astype(str).tolist()
        fingerprint = hashlib.sha256((args.model_name + "|" + split + "|" + "\n".join(texts)).encode()).hexdigest()[:16]
        cache_path = cache_dir / f"{split}_{fingerprint}.npy"
        if cache_path.exists() and not args.force_reembed:
            arr = np.load(cache_path)
            print(f"Loaded cached {split} embeddings: {arr.shape}")
            return arr, 0.0
        start = time.perf_counter()
        arr = encoder.encode(texts, batch_size=args.batch_size, show_progress_bar=True,
                             convert_to_numpy=True, normalize_embeddings=True)
        elapsed = time.perf_counter() - start
        np.save(cache_path, arr.astype(np.float32))
        print(f"Encoded {split}: {arr.shape} in {elapsed:.2f}s")
        return arr, elapsed

    x = {}
    embed_times = {}
    for split in ("train", "validation", "test"):
        x[split], embed_times[split] = embed(split)

    comparisons, models = [], {}
    for c in (1.0, 4.0):
        print(f"\nTraining MiniLM + Logistic Regression (C={c}) ...")
        clf = LogisticRegression(C=c, max_iter=3000, solver="lbfgs", random_state=args.seed)
        t0 = time.perf_counter(); clf.fit(x["train"], y["train"]); train_s = time.perf_counter() - t0
        t0 = time.perf_counter(); val_pred = clf.predict(x["validation"]); pred_s = time.perf_counter() - t0
        m = metrics(y["validation"], val_pred, class_names, pred_s)
        row = {"model": "minilm_logistic_regression", "C": c, "train_seconds": train_s,
               "validation_accuracy": m["accuracy"], "validation_macro_precision": m["macro_precision"],
               "validation_macro_recall": m["macro_recall"], "validation_macro_f1": m["macro_f1"],
               "validation_weighted_f1": m["weighted_f1"],
               "classifier_inference_ms_per_query": m["classifier_inference_ms_per_query"]}
        print(json.dumps(row, indent=2)); comparisons.append(row); models[c] = clf
    comparison = pd.DataFrame(comparisons).sort_values("validation_macro_f1", ascending=False)
    comparison.to_csv(out_dir / "validation_model_comparison.csv", index=False)
    best_c = float(comparison.iloc[0]["C"]); best_model = models[best_c]
    print(f"\nSelected C={best_c} using validation macro-F1")

    t0 = time.perf_counter(); test_pred = best_model.predict(x["test"]); pred_s = time.perf_counter() - t0
    test_metrics = metrics(y["test"], test_pred, class_names, pred_s)
    test_metrics.update({"selected_model": "minilm_logistic_regression", "selected_C": best_c,
                         "test_rows": int(len(dfs["test"])), "embedding_model": args.model_name,
                         "train_embedding_seconds_this_run": embed_times["train"],
                         "validation_embedding_seconds_this_run": embed_times["validation"],
                         "test_embedding_seconds_this_run": embed_times["test"],
                         "note": "Classifier latency excludes embedding generation; measure end-to-end latency separately."})
    (out_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
    report = classification_report(y["test"], test_pred, labels=list(range(len(class_names)),),
                                   target_names=class_names, output_dict=True, zero_division=0)
    pd.DataFrame(report).transpose().to_csv(out_dir / "test_classification_report.csv")
    cm = confusion_matrix(y["test"], test_pred, labels=list(range(len(class_names))))
    pd.DataFrame(cm, index=[f"true_{n}" for n in class_names], columns=[f"pred_{n}" for n in class_names]).to_csv(out_dir / "test_confusion_matrix.csv")
    fig, ax = plt.subplots(figsize=(7, 6)); im = ax.imshow(cm, interpolation="nearest"); fig.colorbar(im, ax=ax)
    ax.set(title="MiniLM Sentiment v2 — Test Confusion Matrix", xlabel="Predicted sentiment", ylabel="True sentiment",
           xticks=np.arange(len(class_names)), yticks=np.arange(len(class_names)), xticklabels=class_names, yticklabels=class_names)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]): ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > cm.max()/2 else "black")
    fig.tight_layout(); fig.savefig(out_dir / "test_confusion_matrix.png", dpi=160); plt.close(fig)
    pred_df = dfs["test"].copy(); pred_df["True_Sentiment"] = encoder_labels.inverse_transform(y["test"])
    pred_df["Predicted_Sentiment"] = encoder_labels.inverse_transform(test_pred)
    pred_df["Correct"] = pred_df["True_Sentiment"] == pred_df["Predicted_Sentiment"]
    pred_df.to_csv(out_dir / "test_predictions.csv", index=False)
    joblib.dump(best_model, out_dir / "sentiment_classifier.joblib")
    joblib.dump(encoder_labels, out_dir / "label_encoder.joblib")
    metadata = {"embedding_model": args.model_name, "classifier": "LogisticRegression", "selected_C": best_c,
                "classes": class_names, "seed": args.seed, "requested_device": args.device or "auto",
                "batch_size": args.batch_size, "data_dir": str(data_dir), "output_dir": str(out_dir),
                "input_sha256": {f: sha256(p) for f, p in paths.items()},
                "row_counts": {s: len(dfs[s]) for s in dfs},
                "class_distributions": {s: {str(k): int(v) for k, v in dfs[s][args.target_col].value_counts().items()} for s in dfs}}
    (out_dir / "experiment_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("\nFINAL HELD-OUT TEST METRICS\n" + json.dumps(test_metrics, indent=2))
    print(f"\nArtifacts saved to: {out_dir}")

if __name__ == "__main__":
    main()
