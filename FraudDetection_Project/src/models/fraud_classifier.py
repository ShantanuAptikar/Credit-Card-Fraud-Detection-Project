"""
Fraud Detection ML Classifier
Credit & Debit Card Fraud Detection System
Authors: Karan Sumbe, Isha Ghokane, Shantanu Aptikar, Shreya Pawar
Guide: Prof. Aradhana Pawar & Dr. Lakshmikant Malphedwar
"""

import numpy as np
import pandas as pd
import pickle
import os
from datetime import datetime
from typing import Dict, Tuple, Optional

# sklearn imports
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, f1_score
)
from sklearn.utils import resample

# Optional: try importing imblearn for SMOTE
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False


class FraudClassifier:
    """
    ML-based transaction fraud classifier using Random Forest with
    feature engineering tailored for card transaction data.
    """

    def __init__(self, model_type: str = "random_forest", model_path: str = "models/"):
        self.model_type = model_type
        self.model_path = model_path
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.fraud_threshold = 0.65
        self.review_threshold = 0.45
        self.is_trained = False

        os.makedirs(model_path, exist_ok=True)
        self._init_model()

    def _init_model(self):
        """Initialize the ML model based on type."""
        if self.model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=500,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight="balanced",
                n_jobs=-1,
                random_state=42,
                max_features="sqrt"
            )
        elif self.model_type == "gradient_boosting":
            self.model = GradientBoostingClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=5,
                random_state=42
            )
        elif self.model_type == "logistic_regression":
            self.model = LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def extract_features(self, transaction: Dict) -> np.ndarray:
        """
        Extract and engineer features from a transaction dict.

        Args:
            transaction: Dict with keys:
                - amount (float): Transaction amount in USD
                - hour (int): Hour of day (0-23)
                - day_of_week (int): Day of week (0=Mon, 6=Sun)
                - merchant_category (str): MCC category string
                - distance_from_home (float): km from cardholder home
                - distance_from_last_tx (float): km from last transaction location
                - time_since_last_tx (float): minutes since last transaction
                - daily_tx_count (int): transactions in last 24h
                - daily_spend (float): total spend in last 24h
                - weekly_avg_spend (float): avg daily spend last 7 days
                - is_foreign (bool): transaction in foreign country
                - device_match (bool): device seen before
                - is_online (bool): CNP (card-not-present) transaction
                - velocity_flag (bool): > 3 transactions in 1 hour

        Returns:
            np.ndarray: Feature vector
        """
        # Amount features
        amount = float(transaction.get("amount", 0))
        log_amount = np.log1p(amount)
        amount_zscore = (amount - transaction.get("weekly_avg_spend", amount)) / max(
            transaction.get("weekly_avg_spend", 1) * 0.3, 1
        )

        # Time features
        hour = int(transaction.get("hour", 12))
        is_night = 1 if hour < 6 or hour > 22 else 0
        is_weekend = 1 if transaction.get("day_of_week", 0) >= 5 else 0

        # Merchant category encoding
        mcc_risk = {
            "gambling": 0.8, "crypto": 0.7, "electronics": 0.5,
            "travel": 0.4, "grocery": 0.1, "gas": 0.2,
            "restaurant": 0.1, "utility": 0.05, "other": 0.3
        }
        category = transaction.get("merchant_category", "other").lower()
        mcc_risk_score = mcc_risk.get(category, 0.3)

        # Location features
        dist_home = float(transaction.get("distance_from_home", 0))
        dist_last = float(transaction.get("distance_from_last_tx", 0))
        is_foreign = 1 if transaction.get("is_foreign", False) else 0

        # Velocity features
        daily_count = int(transaction.get("daily_tx_count", 1))
        daily_spend = float(transaction.get("daily_spend", amount))
        weekly_avg = float(transaction.get("weekly_avg_spend", daily_spend))
        spend_ratio = daily_spend / max(weekly_avg, 1)

        # Time since last tx
        time_since = float(transaction.get("time_since_last_tx", 1440))  # minutes
        rapid_tx = 1 if time_since < 5 else 0

        # Device and channel
        device_match = 1 if transaction.get("device_match", True) else 0
        is_online = 1 if transaction.get("is_online", False) else 0
        velocity_flag = 1 if transaction.get("velocity_flag", False) else 0

        features = np.array([
            log_amount,           # Log-transformed amount
            amount_zscore,        # Amount z-score vs weekly avg
            is_night,             # Night transaction flag
            is_weekend,           # Weekend flag
            mcc_risk_score,       # Merchant category risk
            np.log1p(dist_home),  # Log distance from home
            np.log1p(dist_last),  # Log distance from last tx
            is_foreign,           # Foreign country flag
            daily_count,          # Daily transaction count
            spend_ratio,          # Today's spend / weekly avg
            np.log1p(time_since), # Log time since last tx
            rapid_tx,             # Rapid succession flag
            device_match,         # Known device flag (0 = unknown = risky)
            is_online,            # Card-not-present flag
            velocity_flag,        # Velocity violation flag
        ], dtype=np.float32)

        self.feature_names = [
            "log_amount", "amount_zscore", "is_night", "is_weekend",
            "mcc_risk", "log_dist_home", "log_dist_last", "is_foreign",
            "daily_count", "spend_ratio", "log_time_since", "rapid_tx",
            "device_match", "is_online", "velocity_flag"
        ]

        return features

    def generate_synthetic_training_data(self, n_samples: int = 10000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic transaction data for demonstration/testing.
        In production, use real labeled transaction data.
        """
        np.random.seed(42)
        n_fraud = int(n_samples * 0.05)  # 5% fraud rate
        n_legit = n_samples - n_fraud

        def gen_transaction(is_fraud: bool) -> Dict:
            if is_fraud:
                return {
                    "amount": np.random.choice([
                        np.random.uniform(500, 5000),   # large amounts
                        np.random.uniform(1, 5)          # micro-transactions
                    ]),
                    "hour": np.random.choice([0, 1, 2, 3, 4, 23]),  # night
                    "day_of_week": np.random.randint(0, 7),
                    "merchant_category": np.random.choice(["gambling", "crypto", "electronics"]),
                    "distance_from_home": np.random.uniform(500, 5000),
                    "distance_from_last_tx": np.random.uniform(200, 3000),
                    "time_since_last_tx": np.random.uniform(0.5, 30),
                    "daily_tx_count": np.random.randint(5, 20),
                    "daily_spend": np.random.uniform(1000, 8000),
                    "weekly_avg_spend": np.random.uniform(50, 200),
                    "is_foreign": np.random.random() > 0.3,
                    "device_match": np.random.random() > 0.7,
                    "is_online": np.random.random() > 0.3,
                    "velocity_flag": np.random.random() > 0.5
                }
            else:
                avg_spend = np.random.uniform(50, 300)
                return {
                    "amount": np.random.uniform(5, avg_spend * 1.5),
                    "hour": np.random.randint(8, 21),
                    "day_of_week": np.random.randint(0, 7),
                    "merchant_category": np.random.choice(["grocery", "restaurant", "gas", "utility"]),
                    "distance_from_home": np.random.uniform(0, 50),
                    "distance_from_last_tx": np.random.uniform(0, 100),
                    "time_since_last_tx": np.random.uniform(60, 2880),
                    "daily_tx_count": np.random.randint(1, 5),
                    "daily_spend": np.random.uniform(10, avg_spend * 2),
                    "weekly_avg_spend": avg_spend,
                    "is_foreign": np.random.random() > 0.95,
                    "device_match": np.random.random() > 0.05,
                    "is_online": np.random.random() > 0.7,
                    "velocity_flag": np.random.random() > 0.95
                }

        X_list, y_list = [], []
        for _ in range(n_legit):
            tx = gen_transaction(False)
            X_list.append(self.extract_features(tx))
            y_list.append(0)
        for _ in range(n_fraud):
            tx = gen_transaction(True)
            X_list.append(self.extract_features(tx))
            y_list.append(1)

        X = np.array(X_list)
        y = np.array(y_list)
        # Shuffle
        idx = np.random.permutation(len(y))
        return X[idx], y[idx]

    def train(self, X: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None,
              use_synthetic: bool = False) -> Dict:
        """
        Train the fraud classifier.

        Args:
            X: Feature matrix (n_samples, n_features). If None, generates synthetic data.
            y: Labels (0=genuine, 1=fraud). If None, generates synthetic data.
            use_synthetic: Force use of synthetic data.

        Returns:
            Dict with training metrics.
        """
        if X is None or use_synthetic:
            print("Generating synthetic training data...")
            X, y = self.generate_synthetic_training_data(n_samples=20000)

        print(f"Training on {X.shape[0]} samples with {X.shape[1]} features")
        print(f"Fraud rate: {y.mean():.1%}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.15, stratify=y, random_state=42
        )

        # Apply SMOTE if available
        if SMOTE_AVAILABLE:
            print("Applying SMOTE oversampling...")
            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)
        else:
            print("SMOTE not available, using class_weight='balanced'")

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        print(f"Training {self.model_type} model...")
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True

        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        y_prob = self.model.predict_proba(X_test_scaled)[:, 1]

        metrics = {
            "accuracy": float((y_pred == y_test).mean()),
            "auc_roc": float(roc_auc_score(y_test, y_prob)),
            "f1_score": float(f1_score(y_test, y_pred)),
            "classification_report": classification_report(y_test, y_pred),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
        }

        print("\n=== Training Results ===")
        print(f"Accuracy:  {metrics['accuracy']:.3f}")
        print(f"AUC-ROC:   {metrics['auc_roc']:.3f}")
        print(f"F1 Score:  {metrics['f1_score']:.3f}")
        print("\nClassification Report:")
        print(metrics["classification_report"])

        return metrics

    def predict(self, transaction: Dict) -> Dict:
        """
        Predict fraud probability for a single transaction.

        Args:
            transaction: Transaction feature dict

        Returns:
            Dict with fraud_score, decision, features
        """
        if not self.is_trained:
            # Load model if available, otherwise use heuristic
            if not self.load_model():
                return self._heuristic_score(transaction)

        features = self.extract_features(transaction)
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        fraud_prob = float(self.model.predict_proba(features_scaled)[0, 1])

        if fraud_prob >= self.fraud_threshold:
            decision = "REJECT"
            risk_level = "HIGH"
        elif fraud_prob >= self.review_threshold:
            decision = "HOLD_FOR_REVIEW"
            risk_level = "MEDIUM"
        else:
            decision = "APPROVE"
            risk_level = "LOW"

        return {
            "fraud_score": round(fraud_prob, 4),
            "decision": decision,
            "risk_level": risk_level,
            "fraud_threshold": self.fraud_threshold,
            "model_type": self.model_type
        }

    def _heuristic_score(self, transaction: Dict) -> Dict:
        """Fallback heuristic scoring when model not trained."""
        score = 0.0
        if transaction.get("amount", 0) > 1000:
            score += 0.3
        if transaction.get("is_foreign", False):
            score += 0.2
        if not transaction.get("device_match", True):
            score += 0.3
        if transaction.get("velocity_flag", False):
            score += 0.2
        score = min(score, 0.99)
        decision = "REJECT" if score > 0.65 else ("HOLD_FOR_REVIEW" if score > 0.45 else "APPROVE")
        return {"fraud_score": round(score, 4), "decision": decision, "risk_level": "UNKNOWN", "note": "heuristic_mode"}

    def save_model(self, filename: str = "fraud_model.pkl"):
        """Save trained model and scaler to disk."""
        path = os.path.join(self.model_path, filename)
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "scaler": self.scaler, "features": self.feature_names}, f)
        print(f"Model saved to {path}")

    def load_model(self, filename: str = "fraud_model.pkl") -> bool:
        """Load model from disk. Returns True if successful."""
        path = os.path.join(self.model_path, filename)
        if not os.path.exists(path):
            return False
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.feature_names = data.get("features", [])
        self.is_trained = True
        print(f"Model loaded from {path}")
        return True


if __name__ == "__main__":
    print("=== Fraud Classifier Training Demo ===\n")
    clf = FraudClassifier(model_type="random_forest", model_path="./models/")
    metrics = clf.train(use_synthetic=True)
    clf.save_model()

    print("\n=== Prediction Examples ===")
    legit_tx = {
        "amount": 45.0, "hour": 14, "day_of_week": 2,
        "merchant_category": "grocery", "distance_from_home": 3.0,
        "distance_from_last_tx": 5.0, "time_since_last_tx": 480,
        "daily_tx_count": 2, "daily_spend": 80.0, "weekly_avg_spend": 120.0,
        "is_foreign": False, "device_match": True, "is_online": False, "velocity_flag": False
    }
    fraud_tx = {
        "amount": 2499.0, "hour": 2, "day_of_week": 6,
        "merchant_category": "gambling", "distance_from_home": 3200.0,
        "distance_from_last_tx": 1500.0, "time_since_last_tx": 2,
        "daily_tx_count": 12, "daily_spend": 4800.0, "weekly_avg_spend": 90.0,
        "is_foreign": True, "device_match": False, "is_online": True, "velocity_flag": True
    }

    print("\nLegitimate transaction:")
    result = clf.predict(legit_tx)
    for k, v in result.items():
        print(f"  {k}: {v}")

    print("\nFraudulent transaction:")
    result = clf.predict(fraud_tx)
    for k, v in result.items():
        print(f"  {k}: {v}")
