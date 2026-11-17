import argparse
from pathlib import Path
import joblib
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classifier", default="outputs/intent_minilm/intent_minilm_classifier.joblib")
    ap.add_argument("--labels", default="outputs/intent_minilm/label_encoder.joblib")
    ap.add_argument("--query", required=True)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    a = ap.parse_args()
    for path in [a.classifier, a.labels]:
        if not Path(path).exists():
            raise FileNotFoundError(f"Missing artifact: {path}. Train first.")
    import torch
    device = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    clf = joblib.load(a.classifier)
    le = joblib.load(a.labels)
    encoder = SentenceTransformer(MODEL_NAME, device=device)
    vector = encoder.encode([a.query], convert_to_numpy=True, normalize_embeddings=True, device=device)
    pred = clf.predict(vector)
    print("Query:", a.query)
    print("Predicted intent:", le.inverse_transform(pred)[0])

if __name__ == "__main__":
    main()
