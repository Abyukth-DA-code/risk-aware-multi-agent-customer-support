import argparse, json, time
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)

def load_split(path, name, text_col, target_col):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{name} file not found: {path}")
    df = pd.read_csv(path)
    if text_col not in df.columns or target_col not in df.columns:
        raise ValueError(f"{path} missing '{text_col}' or '{target_col}'. Available: {list(df.columns)}")
    if df[text_col].isna().any() or df[target_col].isna().any():
        raise ValueError(f"{path} contains missing text/target values.")
    df[text_col] = df[text_col].astype(str).str.strip()
    df[target_col] = df[target_col].astype(str).str.strip()
    print(f"{name}: {len(df)} rows, {df[target_col].nunique()} sentiment classes")
    print(f"  Distribution: {df[target_col].value_counts().to_dict()}")
    return df

def make_features():
    return FeatureUnion([
        ("word", TfidfVectorizer(analyzer="word", ngram_range=(1,2), min_df=1,
            sublinear_tf=True, strip_accents="unicode", token_pattern=r"(?u)\b\w+\b", max_features=250000)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), min_df=2,
            sublinear_tf=True, max_features=250000))
    ])

def calculate_metrics(y_true, y_pred, labels, label_names):
    pm, rm, fm, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    pw, rw, fw, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    report = classification_report(y_true, y_pred, labels=labels, target_names=label_names, output_dict=True, zero_division=0)
    neg_recall = None
    for label in label_names:
        if label.strip().lower() in {"negative", "neg"}:
            neg_recall = float(report[label]["recall"])
            break
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(pm),
        "macro_recall": float(rm),
        "macro_f1": float(fm),
        "weighted_precision": float(pw),
        "weighted_recall": float(rw),
        "weighted_f1": float(fw),
        "negative_class_recall": neg_recall
    }

def main():
    ap = argparse.ArgumentParser(description="Train sentiment TF-IDF baselines.")
    ap.add_argument("--train", default="data/processed/sentiment/v2_refined/train.csv")
    ap.add_argument("--validation", default="data/processed/sentiment/v2_refined/validation.csv")
    ap.add_argument("--test", default="data/processed/sentiment/v2_refined/test.csv")
    ap.add_argument("--out", default="outputs/sentiment_model2/v2_tfidf")
    ap.add_argument("--text-col", default="Query")
    ap.add_argument("--target-col", default="Sentiment")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    tr = load_split(args.train, "Train", args.text_col, args.target_col)
    va = load_split(args.validation, "Validation", args.text_col, args.target_col)
    te = load_split(args.test, "Test", args.text_col, args.target_col)

    encoder = LabelEncoder()
    y_train = encoder.fit_transform(tr[args.target_col])
    y_val = encoder.transform(va[args.target_col])
    y_test = encoder.transform(te[args.target_col])
    labels = list(range(len(encoder.classes_)))

    models = {
        "tfidf_logistic_regression": Pipeline([
            ("features", make_features()),
            ("clf", LogisticRegression(max_iter=3000, C=4.0, solver="lbfgs", random_state=42))
        ]),
        "tfidf_linear_svm": Pipeline([
            ("features", make_features()),
            ("clf", LinearSVC(C=1.0, random_state=42))
        ])
    }

    trained = []; rows = []
    for name, model in models.items():
        print(f"\nTraining {name} ...")
        t0 = time.perf_counter()
        model.fit(tr[args.text_col], y_train)
        train_seconds = time.perf_counter() - t0

        pred = model.predict(va[args.text_col])
        m = calculate_metrics(y_val, pred, labels, list(encoder.classes_))

        sample = te[args.text_col].iloc[:min(100, len(te))]
        t0 = time.perf_counter()
        model.predict(sample)
        latency = (time.perf_counter() - t0) / len(sample) * 1000

        row = {
            "model": name, "train_seconds": round(train_seconds, 4),
            "validation_accuracy": m["accuracy"], "validation_macro_f1": m["macro_f1"],
            "validation_negative_recall": m["negative_class_recall"],
            "approx_inference_ms_per_query": round(latency, 4)
        }
        rows.append(row); trained.append((name, model))
        print(json.dumps(row, indent=2))

    comp = pd.DataFrame(rows).sort_values("validation_macro_f1", ascending=False)
    comp.to_csv(out / "validation_model_comparison.csv", index=False)
    winner_name = str(comp.iloc[0]["model"])
    winner = dict(trained)[winner_name]
    print(f"\nSelected Winner: {winner_name}")

    # Final evaluation on held-out test split
    pred = winner.predict(te[args.text_col])
    m = calculate_metrics(y_test, pred, labels, list(encoder.classes_))
    test_metrics = {"selected_model": winner_name, **m}
    (out / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")

    report = classification_report(y_test, pred, labels=labels, target_names=list(encoder.classes_), output_dict=True, zero_division=0)
    pd.DataFrame(report).T.to_csv(out / "test_classification_report.csv")

    pred_labels = encoder.inverse_transform(pred)
    pd.DataFrame({
        "Query": te[args.text_col],
        "True_Sentiment": te[args.target_col],
        "Predicted_Sentiment": pred_labels,
        "Correct": te[args.target_col].to_numpy() == pred_labels
    }).to_csv(out / "test_predictions.csv", index=False)

    cm = confusion_matrix(y_test, pred, labels=labels)
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(cm, display_labels=encoder.classes_).plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(f"Sentiment Test Confusion Matrix — {winner_name}")
    fig.tight_layout()
    fig.savefig(out / "test_confusion_matrix.png", dpi=180)
    plt.close(fig)

    joblib.dump(winner, out / "sentiment_model.joblib")
    joblib.dump(encoder, out / "label_encoder.joblib")
    print("\nFINAL HELD-OUT TEST METRICS (TF-IDF Baseline):")
    print(json.dumps(test_metrics, indent=2))

if __name__ == "__main__":
    main()