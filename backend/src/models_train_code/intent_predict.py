import argparse
from pathlib import Path
import joblib

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",default="outputs/intent_model1/intent_model.joblib")
    ap.add_argument("--query",required=True)
    a=ap.parse_args()
    if not Path(a.model).exists():
        raise FileNotFoundError(f"Model not found: {a.model}. Run training first.")
    model=joblib.load(a.model)
    print("Query:",a.query)
    print("Predicted intent:",model.predict([a.query])[0])

if __name__=="__main__":
    main()
