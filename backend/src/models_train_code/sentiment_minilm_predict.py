"""Predict sentiment using saved MiniLM embeddings + Logistic Regression classifier."""
import argparse
from pathlib import Path
import joblib
from sentence_transformers import SentenceTransformer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--model-dir", default="outputs/sentiment_model2/v2_minilm")
    parser.add_argument("--model-name", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    model_dir = Path(args.model_dir).resolve()
    clf = joblib.load(model_dir / "sentiment_classifier.joblib")
    le = joblib.load(model_dir / "label_encoder.joblib")
    kwargs = {"device": args.device} if args.device else {}
    encoder = SentenceTransformer(args.model_name, **kwargs)
    embedding = encoder.encode([args.query], convert_to_numpy=True, normalize_embeddings=True)
    pred = clf.predict(embedding)[0]
    print("Query:", args.query)
    print("Predicted sentiment:", le.inverse_transform([pred])[0])
    if hasattr(clf, "predict_proba"):
        for class_id, prob in zip(clf.classes_, clf.predict_proba(embedding)[0]):
            print(f"  {le.inverse_transform([class_id])[0]}: {prob:.4f}")

if __name__ == "__main__":
    main()
