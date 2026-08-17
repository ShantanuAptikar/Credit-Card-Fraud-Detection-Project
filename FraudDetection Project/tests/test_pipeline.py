"""
Unit & Integration Tests
Credit & Debit Card Fraud Detection System
Authors: Karan Sumbe, Isha Ghokane, Shantanu Aptikar, Shreya Pawar
"""

import sys
import os
import numpy as np
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.fraud_classifier import FraudClassifier
from models.face_recognition import FaceRecognitionEngine


class TestFraudClassifier(unittest.TestCase):
    """Tests for the ML fraud classifier."""

    @classmethod
    def setUpClass(cls):
        cls.clf = FraudClassifier(model_type="random_forest", model_path="/tmp/test_models/")
        cls.clf.train(use_synthetic=True)

    def _legit_tx(self, **overrides):
        tx = {
            "amount": 60.0, "hour": 14, "day_of_week": 2,
            "merchant_category": "grocery", "distance_from_home": 3.0,
            "distance_from_last_tx": 5.0, "time_since_last_tx": 480,
            "daily_tx_count": 2, "daily_spend": 80.0, "weekly_avg_spend": 120.0,
            "is_foreign": False, "device_match": True, "is_online": False, "velocity_flag": False
        }
        tx.update(overrides)
        return tx

    def _fraud_tx(self, **overrides):
        tx = {
            "amount": 2500.0, "hour": 2, "day_of_week": 6,
            "merchant_category": "gambling", "distance_from_home": 4000.0,
            "distance_from_last_tx": 1500.0, "time_since_last_tx": 2,
            "daily_tx_count": 15, "daily_spend": 6000.0, "weekly_avg_spend": 80.0,
            "is_foreign": True, "device_match": False, "is_online": True, "velocity_flag": True
        }
        tx.update(overrides)
        return tx

    def test_classifier_trained(self):
        self.assertTrue(self.clf.is_trained)

    def test_feature_extraction(self):
        features = self.clf.extract_features(self._legit_tx())
        self.assertEqual(features.shape, (15,))
        self.assertTrue(np.all(np.isfinite(features)))

    def test_legit_transaction_approved(self):
        result = self.clf.predict(self._legit_tx())
        self.assertIn("fraud_score", result)
        self.assertIn("decision", result)
        self.assertLessEqual(result["fraud_score"], 0.65,
            f"Expected low fraud score for legit tx, got {result['fraud_score']}")

    def test_fraud_transaction_detected(self):
        result = self.clf.predict(self._fraud_tx())
        self.assertIn(result["decision"], ["REJECT", "HOLD_FOR_REVIEW"],
            f"Expected fraud detection, got {result['decision']}")

    def test_fraud_score_range(self):
        for _ in range(10):
            tx = self._legit_tx(amount=np.random.uniform(5, 500))
            result = self.clf.predict(tx)
            self.assertGreaterEqual(result["fraud_score"], 0.0)
            self.assertLessEqual(result["fraud_score"], 1.0)

    def test_model_save_load(self):
        self.clf.save_model("test_model.pkl")
        new_clf = FraudClassifier(model_path="/tmp/test_models/")
        loaded = new_clf.load_model("test_model.pkl")
        self.assertTrue(loaded)
        result = new_clf.predict(self._legit_tx())
        self.assertIn("fraud_score", result)

    def test_synthetic_data_generation(self):
        X, y = self.clf.generate_synthetic_training_data(n_samples=500)
        self.assertEqual(X.shape[0], 500)
        self.assertEqual(X.shape[1], 15)
        self.assertIn(0, y)
        self.assertIn(1, y)

    def test_bulk_prediction_speed(self):
        import time
        transactions = [self._legit_tx(amount=np.random.uniform(10, 500)) for _ in range(100)]
        start = time.time()
        for tx in transactions:
            self.clf.predict(tx)
        elapsed = time.time() - start
        avg_ms = (elapsed / 100) * 1000
        self.assertLess(avg_ms, 200, f"Avg prediction time {avg_ms:.1f}ms exceeds 200ms")
        print(f"\n  [PERF] Avg ML prediction time: {avg_ms:.2f}ms per transaction")


class TestFaceRecognition(unittest.TestCase):
    """Tests for the face recognition engine."""

    @classmethod
    def setUpClass(cls):
        cls.engine = FaceRecognitionEngine(embedding_store_path="/tmp/test_faces/")
        cls.test_face = np.random.rand(160, 160, 3).astype(np.float32)
        cls.engine.enroll_user("test_user", cls.test_face)

    def test_enrollment_success(self):
        success = self.engine.enroll_user("test_user_2",
                                           np.random.rand(160, 160, 3).astype(np.float32))
        self.assertTrue(success)

    def test_user_listed_after_enrollment(self):
        self.assertIn("test_user", self.engine.get_enrolled_users())

    def test_unknown_user_rejected(self):
        result = self.engine.verify_user("non_existent_user",
                                          np.random.rand(160, 160, 3).astype(np.float32))
        self.assertFalse(result["match"])
        self.assertIn("error", result)

    def test_embedding_dimensions(self):
        face = np.random.rand(160, 160, 3).astype(np.float32)
        embedding = self.engine.compute_embedding(face)
        self.assertEqual(embedding.shape, (self.engine.EMBEDDING_DIM,))

    def test_embedding_normalized(self):
        face = np.random.rand(160, 160, 3).astype(np.float32)
        embedding = self.engine.compute_embedding(face)
        norm = np.linalg.norm(embedding)
        self.assertAlmostEqual(norm, 1.0, places=5)

    def test_verify_returns_score(self):
        result = self.engine.verify_user("test_user", self.test_face)
        self.assertIn("similarity_score", result)
        self.assertIn("match", result)
        self.assertGreaterEqual(result["similarity_score"], 0.0)
        self.assertLessEqual(result["similarity_score"], 1.0)

    def test_liveness_check(self):
        result = self.engine.liveness_check()
        self.assertIsInstance(result, bool)


class TestIntegration(unittest.TestCase):
    """Integration tests for the full pipeline."""

    def test_end_to_end_legit(self):
        """Test that a clearly legitimate transaction is approved."""
        clf = FraudClassifier(model_path="/tmp/test_int/")
        clf.train(use_synthetic=True)

        result = clf.predict({
            "amount": 30.0, "hour": 13, "day_of_week": 2,
            "merchant_category": "grocery", "distance_from_home": 2.0,
            "distance_from_last_tx": 3.0, "time_since_last_tx": 600,
            "daily_tx_count": 1, "daily_spend": 30.0, "weekly_avg_spend": 100.0,
            "is_foreign": False, "device_match": True, "is_online": False, "velocity_flag": False
        })
        # Legitimate transaction should have low fraud score
        self.assertLess(result["fraud_score"], 0.8)

    def test_end_to_end_fraud(self):
        """Test that a clearly fraudulent transaction is flagged."""
        clf = FraudClassifier(model_path="/tmp/test_int/")
        clf.train(use_synthetic=True)

        result = clf.predict({
            "amount": 4999.0, "hour": 1, "day_of_week": 6,
            "merchant_category": "gambling", "distance_from_home": 5000.0,
            "distance_from_last_tx": 3000.0, "time_since_last_tx": 1,
            "daily_tx_count": 20, "daily_spend": 9000.0, "weekly_avg_spend": 50.0,
            "is_foreign": True, "device_match": False, "is_online": True, "velocity_flag": True
        })
        # Fraud transaction should have high fraud score
        self.assertGreater(result["fraud_score"], 0.4)
        print(f"\n  [INFO] Fraud tx score: {result['fraud_score']:.4f} → {result['decision']}")


if __name__ == "__main__":
    print("Running Credit Card Fraud Detection Test Suite\n" + "="*50)
    unittest.main(verbosity=2)
