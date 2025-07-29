import cv2
import numpy as np
from page_locator import RectangleLocator, FixedRectangleLocator
from yolo_locator import YoloLocator
from image_classifier import ClassicVisionClassifier


def flip_image(frame):
    # flip the image 180 degrees, because the camera is upside down
    return cv2.flip(frame, -1)


class VisionDetector:
    """
    A class for detecting and classifying images in a video stream.

    Attributes:
        locator: An instance of a Locator class to find page contours.
        classifier: An instance of a Classifier class to classify the winner.
        refresh_rate: Number of ms between scans.
        conf_frames: Number of frames to consider for a decision.
    """
    def __init__(self, locator, classifier, refresh_rate=1, conf_frames=10):
        self.locator = locator
        self.classifier = classifier
        self.refresh_rate = refresh_rate
        self.conf_frames = conf_frames

    def _find_winner(self, frame, focus_contour, with_scores=False):
        """
        Find the index of the image with the highest score.

        Args:
            frame: The current video frame.
            focus_contour: The contour of the page to focus on.
        """
        winner_index = None
        scores = self.classifier.classify(frame, focus_contour)
        if scores is not None:
            if min(scores) < self.classifier.conf_ths:
                winner_index = np.argmin(scores)
        winner = self.classifier.reference_names[winner_index] if winner_index is not None else None
        if with_scores:
            return winner, scores
        else:
            return winner

    def debug_loop(self):
        """
        Debug loop for testing the locator and classifier.
        """
        # put the main func inside the debug loop (copy the code to here)
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open video stream.")
            return
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = flip_image(frame)
            page_contour = self.locator.get_page_contour(frame)

            if page_contour is not None:
                cv2.drawContours(frame, [page_contour], -1, (0, 255, 0), 3)
                
                # give the focus frame to the classifier
                winner, scores = self._find_winner(frame, page_contour, with_scores=True)
                # show scores in blue, with more space below the winner text
                y_start = 70  # Start scores lower to add space after winner
                if scores is not None:
                    for i, score in enumerate(scores):
                        cv2.putText(frame, f"{self.classifier.reference_names[i]}: {score:.2f}",
                                    (30, y_start + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                if winner is not None:
                    cv2.putText(frame, f"Winner: {winner}", (30, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                else:
                    cv2.putText(frame, "No clear winner", (30, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)


            else:
                cv2.putText(frame, "No page found", (30, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255), 2, cv2.LINE_AA)


            cv2.imshow("Page Detection Debug", frame)

            key = cv2.waitKey(self.refresh_rate)  
            if key == 27:  
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def run(self):
        """
        Main loop for detecting and classifying images in a video stream.
        A generator outputing the the decision for winner
        """
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("Could not open video stream.")
        frame_count = 0
        winner = None
        previous_winner = None
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = flip_image(frame)
            page_contour = self.locator.get_page_contour(frame)

            if page_contour is not None:
                current_winner = self._find_winner(frame, page_contour)
                if current_winner != previous_winner:
                    frame_count = 0
                    previous_winner = current_winner
                else:
                    frame_count += 1
                
                if frame_count >= self.conf_frames:
                    winner = current_winner

            yield winner
            key = cv2.waitKey(self.refresh_rate)
            if key == 27:
                break

if __name__ == "__main__":
    default_rect = (0.1, 0.1, 0.85, 0.8)  # Example rectangle coordinates
    locator = FixedRectangleLocator(default_rect)
    # locator = RectangleLocator()  # Use rectangle locator
    ref_path = "static/chapter_ref" # run from the root of the project
    image_files = ["darkness.jpg", "discover.jpg", "enlightenment.jpg"]
    classifier = ClassicVisionClassifier(reference_images=[f"{ref_path}/{img}" for img in image_files], conf_ths=50)
    detector = VisionDetector(locator, classifier, refresh_rate=1, conf_frames=10)
    detector.debug_loop()  # Start the debug loop


