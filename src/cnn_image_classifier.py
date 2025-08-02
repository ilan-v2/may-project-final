import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from typing import List, Tuple, Dict
from time import time
from image_classifier import ImageClassifier
import cv2

class CNNImageClassifier(ImageClassifier):
    def __init__(
        self,
        reference_images: List[str],
        conf_ths: float=50,
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
        ).to(self.device).eval() 


    def _load_reference_images(self):
        self.reference_images = [
            Image.open(p).convert('L').convert('RGB')for p in self.reference_images
        ]
        if not self.reference_images:
            raise ValueError("No reference images provided or loaded.")

    def _crop_roi(self, frame: cv2.Mat, contour= None) -> Image.Image:
        if contour is None:
            patch = frame.copy()
        else:
            x, y, w, h = cv2.boundingRect(contour)
            patch = frame[y : y + h, x : x + w]

        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        return Image.fromarray(gray).convert('RGB')

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
                imgs, return_tensors="pt",
                padding=True
            ).to(self.device)

            # Forward + post-process
            with torch.no_grad():
                outputs = self.model(**inputs)
            img_sizes = [[(ref_pil.height, ref_pil.width),
                          (query_pil.height, query_pil.width)]]
            processed = self.processor.post_process_keypoint_matching(
                outputs, img_sizes
            )

            # Compute score
            scores[name] = self._calc_score(processed[0])

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
            batch_images.append([ref_pil, query_pil])

        # 2) Tokenize all images in one batch on target device
        start = time()
        inputs = self.processor(
            batch_images,
            return_tensors="pt",
            device=self.device,
            padding=True
        ).to(self.device)
        # print(f"Tokenization took {time() - start:.2f} seconds")

        # 3) Single forward pass for all pairs
        start = time()
        with torch.no_grad():
            outputs = self.model(**inputs)
        # print(f"Model forward pass took {time() - start:.2f} seconds")

        # 4) Post‐process all matches at once
        start = time()
        img_sizes = [
            [(ref.height, ref.width), (query_pil.height, query_pil.width)]
            for ref, _ in batch_images
        ]
        processed_results_batch = self.processor.post_process_keypoint_matching(
            outputs,
            img_sizes
        )
        # print(f"Post-processing took {time() - start:.2f} seconds")

        # 5) Compute per-pair similarity scores
        start = time()
        scores: Dict[str, float] = {}
        for idx, name in enumerate(self.reference_names):
            scores[name] = self._calc_score(processed_results_batch[idx])
        # print(f"Scoring took {time() - start:.2f} seconds")

        # Return best match name and all scores
        best = max(scores, key=scores.get)
        return best, scores

    def _calc_score(self, processed_outputs):
        """
        Calculate the similarity score based on the model outputs.
        Inlier-Ratio: num matches / max(ref_keypoints, query_keypoints)
        """
        matching_scores = processed_outputs["matching_scores"]
        good_matches_index = matching_scores > self.kp_ths
        good_matches = matching_scores[good_matches_index]
        return float(good_matches.sum())
