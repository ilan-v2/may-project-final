import cv2
import numpy as np
from image_classifier import extract_warped_roi
from page_locator import RectangleLocator, FixedRectangleLocator

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    # Use a locator to get a contour (replace with your preferred locator)
    locator = RectangleLocator()  # Or FixedRectangleLocator((0.1, 0.1, 0.85, 0.8))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Optionally flip or preprocess frame here
        page_contour = locator.get_page_contour(frame)
        if page_contour is not None:
            warped = extract_warped_roi(frame, page_contour)
            if warped is not None:
                cv2.imshow("Aligned ROI (Warped)", warped)
            else:
                cv2.imshow("Aligned ROI (Warped)", np.zeros((400, 300, 3), dtype=np.uint8))
            # Draw contour on original frame for reference
            cv2.drawContours(frame, [page_contour], -1, (0, 255, 0), 2)
        else:
            cv2.imshow("Aligned ROI (Warped)", np.zeros((400, 300, 3), dtype=np.uint8))

        cv2.imshow("Original Frame", frame)
        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
