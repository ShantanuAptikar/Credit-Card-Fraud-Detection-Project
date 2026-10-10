# 🛡️ SecurePay: Dual-Gate Credit & Debit Card Fraud Detection System
 
**Version:** 2.0 | March 2026

---

## 📖 Overview

SecurePay is a state-of-the-art, real-time fraud detection system designed to prevent unauthorized credit and debit card usage. It implements a robust **Two-Gate Security Pipeline**:
1. **Gate 1: Biometric Authentication (CNN)**  
   * Facial recognition using InceptionResnetV1 (FaceNet) to match the person making the transaction against the registered cardholder's face embeddings.
   * Liveness detection to prevent spoofing with printed photos.
2. **Gate 2: Behavioral Analysis (ML)**  
   * A Random Forest ML classifier analyzes transaction properties (amount, location, velocity, device matching) to calculate a real-time risk score.

Project Live URL : (https://credit-card-fraud-detection-project-pyn7ylufhg97q8akkcccdx.streamlit.app/)

```text
Transaction Initiated 
       ↓
[Gate 1: Face Match?] → NO → 🚫 BLOCKED (Identity Mismatch)
       ↓ YES
[Gate 2: ML Fraud Score?] 
       ├── HIGH RISK → 🚫 BLOCKED (Fraud Detected)
       ├── MODERATE → ⚠️ HELD FOR REVIEW
       └── LOW RISK → ✅ APPROVED
```

---

## 🚀 Key Features

* **Multi-Frontend Support:** 
  * 🖥️ **Streamlit Dashboard (`ui/app.py`):** For data scientists and system admins to monitor analytics, view transaction history, and configure risk thresholds.
  * 🌐 **SecurePay Web Portal (`securepay_faceauth.html`):** A modern, consumer-facing HTML5/JS web application utilizing MediaPipe for in-browser client-side liveness detection.
* **REST API Backend:** Built with FastAPI, secured via JWT token authentication.
* **Secure Storage:** Face embeddings are encrypted before storage.

---

## 🏗️ Architecture & Project Structure

```text
FraudDetection_Project/
├── src/
│   ├── models/
│   │   ├── fraud_classifier.py    # Random Forest ML behavior classifier
│   │   └── face_recognition.py    # FaceNet CNN biometric engine
│   ├── api/
│   │   └── server.py              # FastAPI REST backend
│   ├── ui/
│   │   └── app.py                 # Streamlit admin dashboard
│   ├── database.py                # Database connection manager
│   └── pipeline.py                # Core detection orchestrator
├── data/
│   ├── face_embeddings/           # Vector store for enrolled faces
│   └── fraud_detection.db         # SQLite storage for users and transactions
├── tests/                         # Pytest unit and integration test suites
├── requirements.txt               # Python dependencies
├── securepay_faceauth.html        # Consumer-facing Web UI
└── README.md                      # Project documentation
```

---

## 🛠️ Quick Start Guide

### 1. Create and Activate Virtual Environment & Install Dependencies
Make sure you have Python 3.9+ installed. It is highly recommended to use a virtual environment.

```bash
# Create a virtual environment named .venv
python -m venv .venv

# Activate the virtual environment:
# On Windows:
.venv\Scripts\activate


# Install the required dependencies:
pip install -r requirements.txt
# Install additional security dependencies required for the FastAPI backend
pip install fastapi uvicorn python-jose passlib bcrypt python-multipart
```

### 2. Start the Backend API Server
The API must be running for both the Streamlit UI and HTML frontend to function.

```bash
cd src
python api/server.py
```
*The API will be available at `http://localhost:8000`*  
*Swagger UI (API Docs) available at `http://localhost:8000/docs`*

### 3. Launch a Frontend

**Option A: Streamlit Admin Dashboard**
```bash
cd src
streamlit run ui/app.py
```
*Access at `http://localhost:8501`*

**Option B: Consumer Web Portal**
Simply open the `securepay_faceauth.html` file in your preferred web browser (Google Chrome Recommended).

---

## 📡 API Endpoints

The pipeline exposes several REST endpoints securely:

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/login` | Access token generator |
| POST | `/enroll` | Register a new cardholder with biometrics |
| POST | `/transaction` | Process transaction through the Dual-Gate pipeline |
| POST | `/generate-otp` | Generate SMS/Email OTP (Simulation) |
| GET | `/dashboard` | Fetch admin analytics and history |

*(Refer to `/docs` on the running API server for detailed interactive payloads)*

---

## ⚙️ Configuration Tuning

Key thresholds can be adjusted dynamically via the Admin Dashboard or hardcoded in `src/config/settings.py`:

*   `FACE_MATCH_THRESHOLD` (Default `0.85`): Cosine similarity required to pass Gate 1.
*   `FRAUD_REJECT_THRESHOLD` (Default `0.65`): ML Score to auto-block in Gate 2.
*   `FRAUD_REVIEW_THRESHOLD` (Default `0.45`): ML Score to hold-for-review in Gate 2.

---

## 🧪 Testing

Run automated tests via `pytest`:
```bash
cd src
python -m pytest tests/ -v
```

---

## 📚 Academic Context

It demonstrates the real-world integration of **Computer Vision** (CNN/FaceNet) and **Statistical Machine Learning** (Random Forest) for Fintech security applications.

**Dataset Sources:** IEEE-CIS Fraud Detection (Kaggle), Synthesized Bio-Samples  
**Metrics:** Achieves >95% fraud catch rate with P95 latency < 1000ms.
