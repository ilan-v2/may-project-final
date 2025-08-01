import cv2
import os
import numpy as np
from skimage.metrics import structural_similarity as ssim
from concurrent.futures import ThreadPoolExecutor
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from typing import List, Tuple, Dict

class ImageClassifier:
    def __init__(self, reference_images=List[str], conf_ths=50):
        self.reference_images = reference_images
        self.reference_names = [os.path.basename(img_path) for img_path in self.reference_images]
        self._load_reference_images()
        self.conf_ths = conf_ths

    def _load_reference_images(self):
        # Load reference images from the provided paths
        self.reference_images = [cv2.imread(img_path) for img_path in self.reference_images]
        if not self.reference_images:
            raise ValueError("No reference images provided or loaded.")

    def classify(self, frame, contour):
        # Implement your classification logic here
        raise NotImplementedError("This method should be implemented by subclasses")

class ClassicVisionClassifier(ImageClassifier):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize OpenCV ORB detector
        self.orb = cv2.ORB_create()
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def ORB_distance(self, current, reference):
        # convert images to grayscale
        current = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
        reference = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
        reference = cv2.resize(reference, (current.shape[1], current.shape[0]))  # Resize to match current image size
        # Compute ORB keypoints and descriptors
        kp_current, des_current = self.orb.detectAndCompute(current, None)
        kp_reference, des_reference = self.orb.detectAndCompute(reference, None)
        if des_current is None or des_reference is None:
            return self.conf_ths * 10  # Return a high score if descriptors are not found

        # Match descriptors
        matches = self.matcher.match(des_current, des_reference)
        if not matches:
            return self.conf_ths * 10

        # Count good matches (distance below a threshold)
        good_matches = [m for m in matches if m.distance < 50]
        # Similarity: higher ratio of good matches means more similar
        similarity = 1 - (len(good_matches) / max(len(matches), 1))
        # Scale similarity to a score (lower is better)
        score = similarity * 100
        return score

    def classify(self, frame, contour):
        # Improved classification logic using perspective transform
        if contour is None:
            return None
        else:
            # focus on the contour area
            x, y, w, h = cv2.boundingRect(contour)
            # Require ROI to be at least 10x10 pixels
            min_size = 10
            if w < min_size or h < min_size:
                return [self.conf_ths * 10 for _ in self.reference_images]
            roi = frame[y:y+h, x:x+w]
            if roi.size == 0 or roi.shape[0] < min_size or roi.shape[1] < min_size:
                return [self.conf_ths * 10 for _ in self.reference_images]

            # get match score per reference image
            orb_scores = []
            for ref in self.reference_images:
                orb_score = self.ORB_distance(roi, ref)
                orb_scores.append(orb_score)
            if not orb_scores:
                return None
            return orb_scores

class CNNImageClassifier(ImageClassifier):
    def __init__(
        self,
        reference_images: List[str],
        conf_ths: float = 0.2,
        kp_ths: float = 0.2,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__(reference_images=reference_images, conf_ths=conf_ths)
        self.kp_ths = kp_ths
        self.device = device

        # Load processor & model
        self.processor = AutoImageProcessor.from_pretrained(
            "ETH-CVG/lightglue_superpoint",
            use_fast=True
        )
        self.model = AutoModel.from_pretrained(
            "ETH-CVG/lightglue_superpoint"
        ).to(self.device).eval()  # eval mode for speed :contentReference[oaicite:7]{index=7}


    def _load_reference_images(self):
        self.reference_images = [
            Image.open(p).convert("RGB") for p in self.reference_images
        ]
        if not self.reference_images:
            raise ValueError("No reference images provided or loaded.")

    def _crop_roi(self, frame: cv2.Mat, contour= None) -> Image.Image:
        if contour is None:
            patch = frame.copy()
        else:
            x, y, w, h = cv2.boundingRect(contour)
            patch = frame[y : y + h, x : x + w]
        
        rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    def classify(
        self, 
        frame: cv2.Mat, 
        contour = None
    ) -> Tuple[str, Dict[str, float]]:
        
        query_pil = self._crop_roi(frame, contour)
        scores: Dict[str, float] = {}
        for name, ref_pil in zip(self.reference_names, self.reference_images):
            # Prepare pair
            imgs = [ref_pil, query_pil]
            inputs = self.processor(
                imgs, return_tensors="pt"
            ).to(self.device)

            # Forward + post-process
            with torch.no_grad():
                outputs = self.model(**inputs)
            img_sizes = [[(ref_pil.height, ref_pil.width),
                          (query_pil.height, query_pil.width)]]
            matches = self.processor.post_process_keypoint_matching(
                outputs, img_sizes, threshold=self.conf_ths
            )

            # Count matches above threshold
            ms = matches[0]["matching_scores"]
            sim_score = (ms > self.conf_ths).sum().item()
            scores[name] = sim_score

        # Return best match name and all scores
        best = max(scores, key=scores.get)
        return best, scores
    
    def classify_batch(
        self, 
        frame: cv2.Mat, 
        contour: List[cv2.Mat] = None
    ) -> List[Tuple[str, Dict[str, float]]]: 
        # 1) Crop the query ROI and prepare reference + queryIL images
        query_pil = self._crop_roi(frame, contour)
        batch_images: List[Image.Image] = []
        for ref_pil in self.reference_images:
            batch_images.append((ref_pil, query_pil))

        print(batch_images)
        # 2) Tokenize all images in one batch on target device
        inputs = self.processor(
            batch_images,
            return_tensors="pt",
            padding=True
        ).to(self.device)

        # 3) Single forward pass for all pairs
        with torch.no_grad():
            outputs = self.model(**inputs)

        # 4) Post‐process all matches at once
        img_sizes = [
            [(ref.height, ref.width), (query_pil.height, query_pil.width)]
            for _ in self.refs_pil
        ]
        matches_batch = self.processor.post_process_keypoint_matching(
            outputs,
            img_sizes,
            threshold=self.conf_ths
        )

        # 5) Compute per-pair similarity scores
        scores: Dict[str, float] = {}
        for idx, name in enumerate(self.reference_names):
            matching_scores = matches_batch[idx]["matching_scores"]
            # Example metric: count of matches > threshold
            sim_score = float((matching_scores > self.conf_ths).sum().item())
            scores[name] = sim_score

        # 6) Select best match
        best_match = max(scores, key=scores.get)
        return best_match, scores