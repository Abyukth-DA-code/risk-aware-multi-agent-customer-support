"""
Unified Model Inference Service
Encapsulates all six existing scikit-learn models and encoders:
- M1: Intent Classifier (LinearSVC / 47 classes)
- M2: Sentiment Analyzer (LogisticRegression + LabelEncoder / 3 classes)
- M3: Risk Tier Classifier (Pipeline + ColumnTransformer / Low, Medium, High)
- M4: Complexity Classifier (Pipeline + ColumnTransformer / Low, Medium, High)
- M5: Domain Classifier (LinearSVC / E-commerce, Education, Small Business)
- M6: Escalation & Autonomy Engine (Pipeline + ColumnTransformer / ANSWER, ORCHESTRATE, ESCALATE)
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import warnings

import joblib
import numpy as np
import pandas as pd

# Suppress harmless sklearn version mismatch warnings when loading pre-trained pipelines
warnings.filterwarnings("ignore", category=UserWarning)


class ModelService:
    """Unified inference interface for the 6 pre-trained capstone models."""

    _instance: Optional["ModelService"] = None

    def __init__(self, models_dir: Optional[Path] = None):
        if models_dir is None:
            # Resolve relative to project root: <project_root>/src/models
            self.models_dir = Path(__file__).resolve().parents[1] / "models"
        else:
            self.models_dir = Path(models_dir)

        self._load_models()

    @classmethod
    def get_instance(cls, models_dir: Optional[Path] = None) -> "ModelService":
        """Singleton accessor to prevent reloading heavy models multiple times."""
        if cls._instance is None:
            cls._instance = cls(models_dir=models_dir)
        return cls._instance

    def _load_models(self) -> None:
        """Load all 6 model pipelines and label encoders from disk."""
        m1_path = self.models_dir / "intent_model_(M1).joblib"
        m2_path = self.models_dir / "sentiment_model_(M2).joblib"
        m2_enc_path = self.models_dir / "sentiment_encoder_(M2).joblib"
        m3_path = self.models_dir / "risk_model_(M3).joblib"
        m4_path = self.models_dir / "complexity_model_(M4).joblib"
        m5_path = self.models_dir / "domain_model_(M5).joblib"
        m6_path = self.models_dir / "Escalation_model_(M6).joblib"

        for p in [m1_path, m2_path, m2_enc_path, m3_path, m4_path, m5_path, m6_path]:
            if not p.exists():
                raise FileNotFoundError(f"Required model artifact not found: {p}")

        # M1: Intent
        self.intent_model = joblib.load(m1_path)

        # M2: Sentiment and its encoder
        self.sentiment_model = joblib.load(m2_path)
        self.sentiment_encoder = joblib.load(m2_enc_path)

        # M3: Risk (dict containing pipeline and classes)
        risk_artifact = joblib.load(m3_path)
        self.risk_pipeline = risk_artifact["pipeline"] if isinstance(risk_artifact, dict) else risk_artifact
        if hasattr(self.risk_pipeline, "named_steps") and "model" in self.risk_pipeline.named_steps:
            self.risk_pipeline.named_steps["model"].n_jobs = 1
        elif hasattr(self.risk_pipeline, "n_jobs"):
            self.risk_pipeline.n_jobs = 1

        # M4: Complexity (dict containing pipeline and classes)
        comp_artifact = joblib.load(m4_path)
        self.complexity_pipeline = comp_artifact["pipeline"] if isinstance(comp_artifact, dict) else comp_artifact

        # M5: Domain
        self.domain_model = joblib.load(m5_path)

        # M6: Escalation / Autonomy Decision
        self.escalation_model = joblib.load(m6_path)

    @staticmethod
    def _softmax(scores: np.ndarray) -> np.ndarray:
        """Compute softmax probabilities from raw decision function margins."""
        exp = np.exp(scores - np.max(scores))
        return exp / np.sum(exp)

    def predict_domain(self, query: str) -> Dict[str, Any]:
        """Predict domain for the query (M5)."""
        query_str = str(query).strip()
        pred = self.domain_model.predict([query_str])[0]
        confidence = 1.0
        if hasattr(self.domain_model, "decision_function"):
            margins = self.domain_model.decision_function([query_str])[0]
            confidence = float(np.max(self._softmax(margins)))
        return {"domain": str(pred), "confidence": round(confidence, 4)}

    def predict_intent(self, query: str) -> Dict[str, Any]:
        """Predict customer intent across 47 classes (M1)."""
        query_str = str(query).strip()
        pred = self.intent_model.predict([query_str])[0]
        confidence = 1.0
        if hasattr(self.intent_model, "decision_function"):
            margins = self.intent_model.decision_function([query_str])[0]
            confidence = float(np.max(self._softmax(margins)))
        return {"intent": str(pred), "confidence": round(confidence, 4)}

    def predict_sentiment(self, query: str) -> Dict[str, Any]:
        """Predict customer query sentiment (M2: Negative, Neutral, Positive)."""
        query_str = str(query).strip()
        probas = self.sentiment_model.predict_proba([query_str])[0]
        idx = int(np.argmax(probas))
        label = self.sentiment_encoder.inverse_transform([idx])[0]
        confidence = float(probas[idx])
        return {"sentiment": str(label), "confidence": round(confidence, 4)}

    def predict_risk(
        self,
        query: str,
        domain: str = "E-commerce",
        sensitive_action: str = "No",
        financial_impact: str = "No",
        action_type: str = "Information/Support",
    ) -> Dict[str, Any]:
        """Predict operational risk tier (M3: Low, Medium, High)."""
        query_str = str(query).strip()
        df = pd.DataFrame(
            [
                {
                    "Query": query_str,
                    "Domain": domain,
                    "Sensitive_Action": sensitive_action,
                    "Financial_Impact": financial_impact,
                    "Action_Type": action_type,
                }
            ]
        )
        probas = self.risk_pipeline.predict_proba(df)[0]
        classes = self.risk_pipeline.classes_
        idx = int(np.argmax(probas))
        return {
            "risk": str(classes[idx]),
            "confidence": round(float(probas[idx]), 4),
            "class_probabilities": {
                str(c): round(float(p), 4) for c, p in zip(classes, probas)
            },
        }

    def predict_complexity(
        self, query: str, domain: str = "E-commerce"
    ) -> Dict[str, Any]:
        """Predict task complexity tier (M4: Low, Medium, High)."""
        query_str = str(query).strip()
        df = pd.DataFrame([{"Query": query_str, "Domain": domain}])
        probas = self.complexity_pipeline.predict_proba(df)[0]
        classes = self.complexity_pipeline.classes_
        idx = int(np.argmax(probas))
        return {
            "complexity": str(classes[idx]),
            "confidence": round(float(probas[idx]), 4),
            "class_probabilities": {
                str(c): round(float(p), 4) for c, p in zip(classes, probas)
            },
        }

    def predict_escalation(
        self,
        query: str,
        domain: str = "E-commerce",
        risk: str = "Low",
        complexity: str = "Low",
        evidence_quality: str = "Strong",
        verification_required: str = "No",
        sensitive_action: str = "No",
        financial_impact: str = "No",
        sentiment: str = "Neutral",
    ) -> Dict[str, Any]:
        """Predict autonomy tier & escalation requirement (M6: ANSWER, ORCHESTRATE, ESCALATE)."""
        query_str = str(query).strip()
        df = pd.DataFrame(
            [
                {
                    "Query": query_str,
                    "Domain": domain,
                    "Risk": risk,
                    "Complexity": complexity,
                    "Evidence_Quality": evidence_quality,
                    "Verification_Required": verification_required,
                    "Sensitive_Action": sensitive_action,
                    "Financial_Impact": financial_impact,
                    "Sentiment": sentiment,
                }
            ]
        )
        pred = self.escalation_model.predict(df)[0]
        confidence = 1.0
        class_probas = {}
        if hasattr(self.escalation_model, "predict_proba"):
            probas = self.escalation_model.predict_proba(df)[0]
            classes = self.escalation_model.classes_
            idx = int(np.argmax(probas))
            confidence = float(probas[idx])
            class_probas = {
                str(c): round(float(p), 4) for c, p in zip(classes, probas)
            }

        return {
            "autonomy_tier": str(pred),
            "escalation_required": "Yes" if pred == "ESCALATE" else "No",
            "confidence": round(confidence, 4),
            "class_probabilities": class_probas,
        }

    def analyze_query(
        self,
        query: str,
        domain: Optional[str] = None,
        sensitive_action: str = "No",
        financial_impact: str = "No",
        action_type: str = "Information/Support",
        evidence_quality: str = "Strong",
        verification_required: str = "No",
    ) -> Dict[str, Any]:
        """
        Cascade all six models sequentially to perform complete query classification:
        1. Predict Domain (or use explicitly provided domain)
        2. Predict Intent (M1)
        3. Predict Sentiment (M2)
        4. Predict Risk (M3)
        5. Predict Complexity (M4)
        6. Predict Escalation / Autonomy Decision (M6)
        """
        # Step 1: Domain
        dom_res = self.predict_domain(query) if not domain else {"domain": domain, "confidence": 1.0}
        resolved_domain = dom_res["domain"]

        # Step 2 & 3: Intent & Sentiment
        intent_res = self.predict_intent(query)
        sentiment_res = self.predict_sentiment(query)

        # Step 4 & 5: Risk & Complexity
        risk_res = self.predict_risk(
            query=query,
            domain=resolved_domain,
            sensitive_action=sensitive_action,
            financial_impact=financial_impact,
            action_type=action_type,
        )
        complexity_res = self.predict_complexity(
            query=query, domain=resolved_domain
        )

        # Step 6: Escalation / Autonomy
        escalation_res = self.predict_escalation(
            query=query,
            domain=resolved_domain,
            risk=risk_res["risk"],
            complexity=complexity_res["complexity"],
            evidence_quality=evidence_quality,
            verification_required=verification_required,
            sensitive_action=sensitive_action,
            financial_impact=financial_impact,
            sentiment=sentiment_res["sentiment"],
        )

        return {
            "query": query,
            "domain": dom_res,
            "intent": intent_res,
            "sentiment": sentiment_res,
            "risk": risk_res,
            "complexity": complexity_res,
            "escalation": escalation_res,
        }


if __name__ == "__main__":
    svc = ModelService.get_instance()
    test_q = "Where is my order? It has been delayed for 3 days."
    print("Testing ModelService self-check...")
    result = svc.analyze_query(test_q)
    print("Inference successful:")
    for k, v in result.items():
        print(f"  {k}: {v}")
