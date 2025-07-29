import cv2
import numpy as np

class Locator:
    def __init__(self):
        pass

    def find_page_contour(self, frame):
        raise NotImplementedError("This method should be implemented by subclasses")
    
    
    def get_page_contour(self, frame):
        contour = self.find_page_contour(frame)
        if contour is not None:
            return contour
        return None


# basic locator class for page detection, looking for rectangular contours
class RectangleLocator(Locator):
    def find_page_contour(self,frame):
        processed = frame.copy()
        # processed = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # processed = cv2.GaussianBlur(processed, (5, 5), 0)
        processed = cv2.Canny(processed, 75, 200)

        contours, _ = cv2.findContours(processed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # Find the largest 4-point contour directly
        max_contour = None
        max_area = 0
        for contour in contours:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) == 4:
                area = cv2.contourArea(approx)
                if area > max_area:
                    max_area = area
                    max_contour = approx

        if max_contour is not None:
            return max_contour

        return None
    
class FixedRectangleLocator(Locator):
    def __init__(self, rect):
        """
        rect: tuple (x, y, width, height) - coordinates of the rectangle relative to the frame.
        coordinate should be between 0 and 1, representing the percentage of the frame size.
        """
        self.rect = rect

    def find_page_contour(self, frame):
        """
        Returns a rectangle contour centered at the given location on the frame.
        """
        x, y, width, height = self.rect
        h, w = frame.shape[:2]
        x1 = int(x * w)
        y1 = int(y * h)
        x2 = int((x + width) * w)
        y2 = int((y + height) * h)
        return np.array([
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2]
        ], dtype=np.int32)