import argparse
from pathlib import Path
import joblib

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",default="outputs/sentiment_model2/sentiment_model.joblib")
    ap.add_argument("--labels",default="outputs/sentiment_model2/label_encoder.joblib")
    ap.add_argument("--query",required=True)
    a=ap.parse_args()
    if not Path(a.model).exists() or not Path(a.labels).exists():
        raise FileNotFoundError("Sentiment model artifacts not found. Train first or provide correct paths.")
    model=joblib.load(a.model); encoder=joblib.load(a.labels)
    pred=model.predict([a.query])
    print("Query:",a.query)
    print("Predicted sentiment:",encoder.inverse_transform(pred)[0])

if __name__=="__main__":
    main()
