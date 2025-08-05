from dataclasses import dataclass
import cv2
import os
import numpy as np
from typing import List, Tuple, Dict
from skimage.metrics import structural_similarity as ssim   


class ImageClassifier:
    def __init__(self, reference_images: List[str], conf_ths=50):
        self.reference_images = reference_images
        self.reference_names = [
            os.path.basename(img_path).split('.')[0]  
            for img_path in self.reference_images]
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


@dataclass
class Contours:
    contours_array: np.ndarray
    bbox: tuple # (x, y, w, h)
    bbox_area: int
    bbox_ratio: float
    
    
class MultiScoreImageClassifier(ImageClassifier):
    def __init__(
            self, 
            reference_images: List[str], 
            bbox_scale_factor: float = 2.0, 
            min_bbox_area_threshold: int = 100,
            max_bbox_area_threshold: int = 4000,
            ratio_threshold: float = 2.0,
            **kwargs
        ):
        super().__init__(reference_images=reference_images, **kwargs)
        self.bbox_scale_factor = bbox_scale_factor
        self.min_bbox_area_threshold = min_bbox_area_threshold
        self.max_bbox_area_threshold = max_bbox_area_threshold
        self.ratio_threshold = ratio_threshold
        self.reference_contours = [self._find_contours(ref)[0] for ref in self.reference_images]

    def _find_contours(self, image):
        """
        Find contours in the image and return them.
        """
        blur = cv2.GaussianBlur(image, (5, 5), 0)
        edged = cv2.Canny(blur, 50, 150)
        cnts, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return cnts
    
    def _get_bounding_box(self, contours):
        """
        Get the bounding box for the contours.
        """
        if contours is None:
            return None
        x, y, w, h = cv2.boundingRect(contours)
        # Scale the bounding box by the factor
        x = int(x - (w * (self.bbox_scale_factor - 1) / 2))
        y = int(y - (h * (self.bbox_scale_factor - 1) / 2))
        w = int(w * self.bbox_scale_factor)
        h = int(h * self.bbox_scale_factor)
        return (x, y, w, h)


    def _query_contours_process(self, query):
        query_contours = self._find_contours(query)
        if not query_contours:
            return None
        cnts_list = []
        for contours in query_contours:
            bbox = self._get_bounding_box(contours)
            if bbox is None:
                continue
            
            # check if bbox is valid - not out of bounds
            x, y, w, h = bbox
            if x < 0 or y < 0 or x + w > query.shape[1] or y + h > query.shape[0]:
                continue

            # calculate area
            area = w*h
            if area < self.min_bbox_area_threshold or area > self.max_bbox_area_threshold:
                continue

            # check aspect ratio
            bbox_ratio = (w/h if w>=h else h/w)
            if bbox_ratio > self.ratio_threshold:
                continue

            cnts_list.append(Contours(
                contours_array=contours,
                bbox=(x, y, w, h),
                bbox_area=area,
                bbox_ratio=bbox_ratio
            ))
        
        return cnts_list

    def _load_reference_images(self):
        super()._load_reference_images()
        if self.reference_images is not None:
            # convert to grayscale
            self.reference_images = [cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY) for ref in self.reference_images]
    
    def _prepare_query(self, query):
        """
        Prepare the query image for comparison.
        """
        gray = cv2.cvtColor(query, cv2.COLOR_BGR2GRAY)
        return gray

    def _color_histogram_distance(self, query, reference):
        #Warning: Not in use
        """
        Calculate the color histogram distance between two images.
        Assuming images are in grayscale.
        """
        hist_query = cv2.calcHist([query], [0], None, [256], [0, 256])
        hist_reference = cv2.calcHist([reference], [0], None, [256], [0, 256])
        cv2.normalize(hist_query, hist_query)
        cv2.normalize(hist_reference, hist_reference)
        corr =  cv2.compareHist(hist_query, hist_reference, cv2.HISTCMP_CORREL)
        return (corr + 1) /2
    
    def _simple_color_similarity(self, query, reference):
        #Warning: Not in use
        """
        calculate mean pixel value similarity between two images.
        """
        mean_query = query.mean()
        mean_reference = reference.mean()
        return 1 - (abs(mean_query - mean_reference) / 255)

    def _shape_similarity(self, query_contours, reference_contours):
        #Warning: Not in use
        """
        Calculate the shape similarity between two images using matching contours.
        """
        if query_contours is None or reference_contours is None:
            return 0.0
        # Calculate shape similarity using cv2.matchShapes
        dist = cv2.matchShapes(query_contours, reference_contours, 2, 0.0)
        sim = 1/ (1 + dist)
        return sim
    
    def _ssim_similarity(self, query, reference):
        """
        Calculate the Structural Similarity Index (SSIM) between two images.
        """
        # match the size of the images
        if query.shape != reference.shape:
            reference = cv2.resize(reference, (query.shape[1], query.shape[0]))
        score, _ = ssim(query, reference, full=True)
        return score

    def classify(self, frame):
        """
        Classify using only data inside the focus area.
        """
        query = self._prepare_query(frame)
        query_contours = self._query_contours_process(query)
        if not query_contours:
            return None

        cnt_scores = []
        for eligible_contour in query_contours:
            # extract focus area for color comparison
            x, y, w, h = eligible_contour.bbox
            query_patch = query[y:y+h, x:x+w]

            # for debugging
            # plt.imshow(query_patch, cmap='gray')
            # plt.title(f'Query Patch, area: {eligible_contour.bbox_area}, ratio: {eligible_contour.bbox_ratio:.2f}')
            # plt.show()

            # get the contours
            query_contours = eligible_contour.contours_array
            # calculate scores for each reference image
            scores = {}
            detailed_scores = {}
            for ref_name, ref_image, ref_contours in zip(
                self.reference_names, self.reference_images, self.reference_contours
                ):
                ssim_score = self._ssim_similarity(query_patch, ref_image)

                scores[ref_name] = ssim_score
            
            cnt_scores.append(scores)

        # select the score with the most confidence
        final_scores = [max(score_dict.values()) for score_dict in cnt_scores]
        best_cnt_idx = np.argmax(final_scores)
        return cnt_scores[best_cnt_idx]