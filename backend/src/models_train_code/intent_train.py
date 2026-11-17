import argparse, json, time
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix, ConfusionMatrixDisplay

def load(path, name):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{name} file not found: {path}")
    df = pd.read_csv(path)
    required = {"Query", "Intent"}
    if not required.issubset(df.columns):
        raise ValueError(f"{path} must contain columns: {sorted(required)}")
    if df["Query"].isna().any() or df["Intent"].isna().any():
        raise ValueError(f"{path} contains missing Query/Intent values")
    df["Query"] = df["Query"].astype(str).str.strip()
    df["Intent"] = df["Intent"].astype(str).str.strip()
    if (df["Query"] == "").any() or (df["Intent"] == "").any():
        raise ValueError(f"{path} contains blank Query/Intent values")
    print(f"{name}: {len(df)} rows, {df['Intent'].nunique()} intents")
    return df

def features():
    return FeatureUnion([
        ("word", TfidfVectorizer(analyzer="word", ngram_range=(1,2), min_df=1,
                                 sublinear_tf=True, strip_accents="unicode",
                                 token_pattern=r"(?u)\b\w+\b", max_features=250000)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), min_df=2,
                                 sublinear_tf=True, max_features=250000))
    ])

def metrics(model, X, y, labels):
    pred = model.predict(X)
    pm, rm, fm, _ = precision_recall_fscore_support(y, pred, labels=labels, average="macro", zero_division=0)
    pw, rw, fw, _ = precision_recall_fscore_support(y, pred, labels=labels, average="weighted", zero_division=0)
    return {"accuracy": float(accuracy_score(y,pred)), "macro_precision":float(pm),
            "macro_recall":float(rm), "macro_f1":float(fm),
            "weighted_precision":float(pw), "weighted_recall":float(rw),
            "weighted_f1":float(fw), "predictions":pred}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/processed/intent/train.csv")
    ap.add_argument("--validation", default="data/processed/intent/validation.csv")
    ap.add_argument("--test", default="data/processed/intent/test.csv")
    ap.add_argument("--out", default="outputs/intent_model1")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tr, va, te = load(a.train,"Train"), load(a.validation,"Validation"), load(a.test,"Test")
    label_set = set(tr.Intent.unique())
    if set(va.Intent.unique()) != label_set or set(te.Intent.unique()) != label_set:
        raise ValueError("Intent classes differ between splits; check taxonomy.")
    labels = sorted(label_set)
    (out/"dataset_summary.json").write_text(json.dumps({
        "train_rows":len(tr),"validation_rows":len(va),"test_rows":len(te),
        "num_intents":len(labels),"input_feature":"Query","target":"Intent",
        "ignored_columns":["Case_ID","Domain","Data_Origin"],
        "selection_rule":"highest validation macro-F1"
    }, indent=2), encoding="utf-8")
    models = {
        "tfidf_logistic_regression": Pipeline([("features",features()),("clf",LogisticRegression(max_iter=3000,C=4.0,solver="lbfgs",random_state=42))]),
        "tfidf_linear_svm": Pipeline([("features",features()),("clf",LinearSVC(C=1.0,random_state=42))])
    }
    trained, rows = {}, []
    for name, model in models.items():
        print(f"\nTraining {name} ...")
        t0=time.perf_counter(); model.fit(tr.Query,tr.Intent); train_sec=time.perf_counter()-t0
        m=metrics(model,va.Query,va.Intent,labels)
        sample=te.Query.iloc[:min(100,len(te))]
        model.predict(sample)
        t0=time.perf_counter(); model.predict(sample); latency=(time.perf_counter()-t0)/len(sample)*1000
        row={"model":name,"train_seconds":round(train_sec,4),
             "validation_accuracy":m["accuracy"],"validation_macro_precision":m["macro_precision"],
             "validation_macro_recall":m["macro_recall"],"validation_macro_f1":m["macro_f1"],
             "validation_weighted_f1":m["weighted_f1"],"approx_inference_ms_per_query":round(latency,4)}
        rows.append(row); trained[name]=model; print(json.dumps(row,indent=2))
    comp=pd.DataFrame(rows).sort_values(["validation_macro_f1","validation_accuracy"],ascending=False)
    comp.to_csv(out/"validation_model_comparison.csv",index=False)
    winner_name=comp.iloc[0]["model"]; winner=trained[winner_name]
    print(f"\nSelected by validation macro-F1: {winner_name}")
    result=metrics(winner,te.Query,te.Intent,labels); pred=result.pop("predictions")
    test_metrics={"selected_model":winner_name,**{f"test_{k}":v for k,v in result.items()}}
    (out/"test_metrics.json").write_text(json.dumps(test_metrics,indent=2),encoding="utf-8")
    pd.DataFrame(classification_report(te.Intent,pred,labels=labels,output_dict=True,zero_division=0)).T.to_csv(out/"test_classification_report.csv")
    pd.DataFrame({"Query":te.Query,"True_Intent":te.Intent,"Predicted_Intent":pred,
                  "Correct":te.Intent.to_numpy()==pred}).to_csv(out/"test_predictions.csv",index=False)
    cm=confusion_matrix(te.Intent,pred,labels=labels)
    size=max(14,len(labels)*0.34); fig,ax=plt.subplots(figsize=(size,size))
    ConfusionMatrixDisplay(cm,display_labels=labels).plot(ax=ax,xticks_rotation=90,colorbar=False,values_format="d")
    ax.set_title(f"Test Confusion Matrix — {winner_name}"); fig.tight_layout()
    fig.savefig(out/"test_confusion_matrix.png",dpi=180); plt.close(fig)
    joblib.dump(winner,out/"intent_model.joblib")
    (out/"intent_labels.json").write_text(json.dumps(labels,indent=2),encoding="utf-8")
    print("\nFINAL HELD-OUT TEST METRICS")
    print(json.dumps(test_metrics,indent=2))
    print(f"\nArtifacts saved to: {out.resolve()}")

if __name__ == "__main__":
    main()
