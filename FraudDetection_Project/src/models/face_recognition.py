"""
Production-Grade Face Recognition Module
Credit & Debit Card Fraud Detection System
Uses MTCNN for detection/alignment and InceptionResnetV1 for identity embeddings.
Authors: Karan Sumbe, Isha Ghokane, Shantanu Aptikar, Shreya Pawar
"""
import numpy as np
import os
import sys
import json
import logging
from typing import Optional, Tuple, Dict
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database import DatabaseManager


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("OpenCV not installed. Running in limited mode.")

try:
    import torch  
    from facenet_pytorch import MTCNN, InceptionResnetV1  
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("facenet-pytorch or torch not installed. Running in demo mode.")

class FaceRecognitionEngine:
    """
    Production-grade facial recognition using FaceNet architecture.
    
    Pipeline:
    1. MTCNN Detection: Locates face and 5 facial landmarks.
    2. Alignment: Affine transformation based on landmarks to normalize pose.
    3. InceptionResnetV1: Generates 512-dimensional identity embedding.
    4. Cosine Similarity: Robust matching against stored biometric templates.
    """
    

    EMBEDDING_DIM = 512
    # Recommended threshold for FaceNet (VGGFace2) is ~0.6-0.8. 
    # 0.70 provides a good balance between FAR and FRR.
    MATCH_THRESHOLD = 0.70 

    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        self.db_manager = db_manager if db_manager else DatabaseManager()
        self.embeddings = {}  # Cache for enrolled users
        self.detector = None
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.is_demo_mode = not (CV2_AVAILABLE and TORCH_AVAILABLE)

        self._load_embeddings()

        if not self.is_demo_mode:
            self._build_models()
        else:
            logger.info("Face Recognition running in DEMO MODE (no real CNN required)")

    def _build_models(self):
        """Initialize MTCNN and InceptionResnetV1 models."""
        try:
            # MTCNN for face detection and alignment
            self.detector = MTCNN(
                image_size=160, 
                margin=14, 
                device=self.device,
                post_process=False # Return raw pixel values [0, 255] for consistency
            )
            
            # InceptionResnetV1 pretrained on VGGFace2 for embeddings
            self.model = InceptionResnetV1(pretrained='vggface2', device=self.device).eval()
            
            logger.info(f"Biometric models initialized on {self.device}")
            logger.info("FaceNet (InceptionResnetV1) 512-dim embedding engine ready.")
        except Exception as e:
            logger.error(f"Failed to load biometric models: {e}")
            self.is_demo_mode = True

    def _load_embeddings(self):
        """Load stored face embeddings from database."""
        try:
            user_ids = self.db_manager.list_users()
            for uid in user_ids:
                user_data = self.db_manager.get_user(uid)
                if user_data and user_data['face_embedding'] is not None:
                    self.embeddings[uid] = user_data['face_embedding']
            logger.info(f"Loaded {len(self.embeddings)} face embeddings from central vault")
        except Exception as e:
            logger.error(f"Database error loading embeddings: {e}")

    def _save_embeddings(self, user_id: str, embedding: np.ndarray):
        """Persist a single user's face embedding to database."""
        try:
            self.db_manager.upsert_user(user_id, "UNKNOWN", embedding)
        except Exception as e:
            logger.error(f"Failed to persist embedding for {user_id}: {e}")

    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Normalize and convert image to torch tensor.
        Input: numpy array (160, 160, 3) in range [0, 1]
        """
        # Convert to float tensor and normalize to [-1, 1] range expected by FaceNet
        img_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
        img_tensor = (img_tensor - 0.5) / 0.5 
        return img_tensor.to(self.device)

    def extract_face(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect and align face using MTCNN.
        Returns normalized face region or None.
        """
        if self.is_demo_mode or self.detector is None:
            return image

        try:
            # MTCNN expects uint8 [0, 255]
            if image.dtype == np.float32 or image.dtype == np.float64:
                img_uint8 = (image * 255).astype(np.uint8)
            else:
                img_uint8 = image

            # Detect and get aligned crop
            # returns tensor (3, 160, 160)
            face_tensor = self.detector(img_uint8)
            
            if face_tensor is not None:
                # Convert back to numpy float32 [0, 1] for our pipeline
                face_np = face_tensor.permute(1, 2, 0).numpy()
                face_np = np.clip(face_np / 255.0, 0, 1)
                return face_np
            
            return None
        except Exception as e:
            logger.error(f"MTCNN Detection Error: {e}")
            return None

    def compute_embedding(self, face_image: np.ndarray) -> np.ndarray:
        """
        Generate a 512-dimensional embedding from a face image.
        Includes face detection and alignment.
        """
        # Step 1: Detect and Align Face
        aligned_face = self.extract_face(face_image)
        
        if aligned_face is None:
            logger.warning("Face detection failed during embedding generation.")
            return np.zeros(self.EMBEDDING_DIM, dtype=np.float32)

        # Step 2: Generate Embedding
        if self.is_demo_mode or self.model is None:
            # Fallback for demo mode: reproducible salt-based hashing of pixels
            # to simulate stable identity features
            flat = aligned_face.flatten()
            embedding = flat[::len(flat)//self.EMBEDDING_DIM][:self.EMBEDDING_DIM]
            embedding = embedding - embedding.mean()
            return embedding / (np.linalg.norm(embedding) + 1e-9)

        try:
            with torch.no_grad():
                img_tensor = self._preprocess_image(aligned_face)
                embedding = self.model(img_tensor).cpu().numpy().flatten()
                
                # Double-check normalization (FaceNet usually outputs unit vectors)
                norm = np.linalg.norm(embedding)
                return embedding / (norm + 1e-9)
        except Exception as e:
            logger.error(f"Embedding Generation Error: {e}")
            return np.zeros(self.EMBEDDING_DIM, dtype=np.float32)

    def enroll_user(self, user_id: str, face_image: Optional[np.ndarray] = None) -> bool:
        """Enroll a user by extracting and storing their biometric identity template."""
        if face_image is None:
            return False

        logger.info(f"Generating biometric template for user: {user_id}")
        embedding = self.compute_embedding(face_image)
        
        if np.all(embedding == 0):
            logger.error(f"Enrollment failed for {user_id}: No face detected.")
            return False

        self.embeddings[user_id] = embedding
        self._save_embeddings(user_id, embedding)
        logger.info(f"Biometric template for '{user_id}' securely enrolled.")
        return True

    def verify_user(self, user_id: str, face_image: Optional[np.ndarray] = None) -> Dict:
        """Verify a presented face image against an enrolled user template.
        Returns dict with keys: match(bool), distance(float), error(optional).
        """
        if user_id not in self.embeddings:
            return {"match": False, "distance": None, "error": "user_not_enrolled"}

        if face_image is None:
            return {"match": False, "distance": None, "error": "no_image"}

        embedding = self.compute_embedding(face_image)
        if np.all(embedding == 0):
            return {"match": False, "distance": None, "error": "detection_failed"}

        enrolled = self.embeddings[user_id]
        # Ensure numpy arrays
        enrolled = np.asarray(enrolled)
        emb = np.asarray(embedding)

        # Use Euclidean distance for comparison (common with FaceNet)
        dist = float(np.linalg.norm(enrolled - emb))
        match = dist < self.MATCH_THRESHOLD
        return {"match": match, "distance": dist}

    def build_gallery_from_dataset(self, dataset_path):
        # Build a gallery by computing mean embedding per user folder
        if not os.path.isdir(dataset_path):
            logger.warning(f"Dataset path not found: {dataset_path}")
            return

        for user_id in os.listdir(dataset_path):
            folder = os.path.join(dataset_path, user_id)

            if not os.path.isdir(folder):
                continue

            embeddings_list = []

            for img in os.listdir(folder):
                img_path = os.path.join(folder, img)

                if CV2_AVAILABLE:
                    image = cv2.imread(img_path)
                    if image is None:
                        continue

                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    emb = self.compute_embedding(image)

                    if not np.all(emb == 0):
                        embeddings_list.append(emb)

            if len(embeddings_list) > 0:
                mean_embedding = np.mean(embeddings_list, axis=0)
                mean_embedding = mean_embedding / (np.linalg.norm(mean_embedding) + 1e-9)

                self.embeddings[user_id] = mean_embedding
                self._save_embeddings(user_id, mean_embedding)

        logger.info(f"Gallery built with {len(self.embeddings)} users")

    def evaluate_far_frr(self, dataset_path: str):
        if not os.path.isdir(dataset_path):
            logger.warning(f"Dataset path not found: {dataset_path}")
            return {"total": 0, "matches": 0, "mismatches": 0}
    
        results = {"total": 0, "matches": 0, "mismatches": 0}
        genuine_total = 0
        genuine_accept = 0
        genuine_reject = 0

        impostor_total = 0
        impostor_accept = 0
        impostor_reject = 0

        users = os.listdir(dataset_path)
        print("Users found:", users)

        for true_user in users:
            print("Current User:", true_user)
            true_folder = os.path.join(dataset_path, true_user)

            if not os.path.isdir(true_folder):
                continue

            for img in os.listdir(true_folder):
                print("Processing:", img)
                img_path = os.path.join(true_folder, img)

                img = cv2.imread(img_path)
                if img is None:
                    continue

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # ───────── GENUINE TEST ─────────
                res = self.verify_user(true_user, img)

                genuine_total += 1
                results["total"] += 1

                if res["match"]:
                  genuine_accept += 1
                  results["matches"] += 1
                else:
                  genuine_reject += 1
                  results["mismatches"] += 1

                # ───────── IMPOSTOR TEST ─────────
                for imposter in users:
                    if imposter == true_user:
                        continue

                    imp_res = self.verify_user(imposter, img)

                    results["total"] += 1

                    impostor_total += 1

                    if imp_res["match"]:
                       impostor_accept += 1
                       results["matches"] += 1
                    else:
                      impostor_reject += 1
                      results["mismatches"] += 1
                    far = impostor_accept / impostor_total if impostor_total > 0 else 0.0
                    frr = genuine_reject / genuine_total if genuine_total > 0 else 0.0

                results["genuine_total"] = genuine_total
                results["impostor_total"] = impostor_total
                results["false_accepts"] = impostor_accept
                results["false_rejects"] = genuine_reject
                results["FAR"] = far
                results["FRR"] = frr
            print("Genuine Accept:", genuine_accept)
            print("Genuine Reject:", genuine_reject)
            print("Impostor Accept:", impostor_accept)
            print("Impostor Reject:", impostor_reject)
        return results



if __name__ == "__main__":
    engine = FaceRecognitionEngine()
engine.build_gallery_from_dataset(dataset_path)

results = engine.evaluate_far_frr(dataset_path)

print("\nFINAL OUTPUT")
print(results)
