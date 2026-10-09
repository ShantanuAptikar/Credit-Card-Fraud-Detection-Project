"""
System Configuration
Credit & Debit Card Fraud Detection System
"""

import os

# ── ML THRESHOLDS ────────────────────────────────────────────────────
FRAUD_REJECT_THRESHOLD = float(os.getenv("FRAUD_REJECT_THRESHOLD", "0.65"))
FRAUD_REVIEW_THRESHOLD = float(os.getenv("FRAUD_REVIEW_THRESHOLD", "0.45"))

# ── FACE RECOGNITION ─────────────────────────────────────────────────
FACE_MATCH_THRESHOLD = float(os.getenv("FACE_MATCH_THRESHOLD", "0.85"))
FACE_EMBEDDING_DIM = 512
CNN_INFERENCE_TIMEOUT_MS = 1000

# ── PATHS ──────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models/")
DATA_PATH = os.path.join(BASE_DIR, "data/")
FACE_EMBEDDING_PATH = os.path.join(DATA_PATH, "face_embeddings/")

# ── API ────────────────────────────────────────────────────────────
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_WORKERS = int(os.getenv("API_WORKERS", "4"))

# ── DATABASE (Production) ──────────────────────────────────────────
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "fraud_detection")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))

# ── ALERTS ────────────────────────────────────────────────────────
ALERT_EMAIL_FROM = os.getenv("ALERT_EMAIL_FROM", "alerts@frauddetect.com")
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY", "")

# ── COMPLIANCE ────────────────────────────────────────────────────
DATA_RETENTION_DAYS = 7 * 365  # 7 years per PCI-DSS
FACE_DATA_RETENTION_DAYS = 365  # 1 year per GDPR minimization
AUDIT_LOG_PATH = os.path.join(DATA_PATH, "audit_logs/")
