import cv2
import os
import numpy as np


class ImageClassifier:
    def __init__(self, reference_images=list[str], conf_ths=50):
        self.reference_images = reference_images
        self.reference_names = [os.path.basename(img_path) for img_path in self.reference_images]
        self.load_reference_images()
        self.conf_ths = conf_ths

    def load_reference_images(self):
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
        # Sort matches by distance
        matches = sorted(matches, key=lambda x: x.distance)
        # Calculate similarity score
        score = sum(m.distance for m in matches) / len(matches) if matches else self.conf_ths * 10
        return score
    

    def classify(self, frame, contour):
        # Example classification logic using OpenCV
        if contour is None:
            return None
        else:
            # focus on the contour area
            x, y, w, h = cv2.boundingRect(contour)
            roi = frame[y:y+h, x:x+w]
            # get match score per reference image
            orb_scores = []
            for ref in self.reference_images:
                orb_score = self.ORB_distance(roi, ref)
                orb_scores.append(orb_score)
            if not orb_scores:
                return None
            return orb_scores
