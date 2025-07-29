import cv2
import numpy as np
from page_locator import RectangleLocator, FixedRectangleLocator
from yolo_locator import YoloLocator
from image_classifier import ClassicVisionClassifier


def flip_image(frame):
    # flip the image 180 degrees, because the camera is upside down
    return cv2.flip(frame, -1)


def main(locator, classifier):
    cap = cv2.VideoCapture(0)  # Use camera index 0 (default)

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    img_names = classifier.reference_names
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = flip_image(frame)
        page_contour = locator.get_page_contour(frame)

        if page_contour is not None:
            cv2.drawContours(frame, [page_contour], -1, (0, 255, 0), 3)
            # find scores for reference images
            scores = classifier.classify(frame, page_contour)
            if scores is not None:
                for i, score in enumerate(scores):
                    cv2.putText(frame, f"Ref {img_names[i]}: {score:.2f}", (30, 60 + i * 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                # write the winner
                winner_index = np.argmin(scores)
                winner = img_names[winner_index]
                if min(scores) < classifier.conf_ths:
                    cv2.putText(frame, f"Winner: {winner}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2, cv2.LINE_AA)
                else:
                    cv2.putText(frame, "No clear winner", (30, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 0, 255), 2, cv2.LINE_AA)


        else:
            cv2.putText(frame, "No page found", (30, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2, cv2.LINE_AA)
            
        # TODO: set minimum number of frames with winner before declaring a winner
        
        cv2.imshow("Page Detection (Live)", frame)

        key = cv2.waitKey(50)  # Wait 50 ms between scans
        if key == 27:  # ESC key to exit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    default_rect = (0.1, 0.1, 0.85, 0.8)  # Example rectangle coordinates
    locator = FixedRectangleLocator(default_rect)
    # locator = RectangleLocator()  # Use rectangle locator
    ref_path = "../static/chapter_ref"
    image_files = ["darkness.jpeg", "discover.jpeg", "enlightenment.png"]
    classifier = ClassicVisionClassifier(reference_images=[f"{ref_path}/{img}" for img in image_files], conf_ths=50)
    main(locator, classifier)