from dataclasses import dataclass
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
        
        contours, hierarchy = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Return the largest contour by area
            return max(contours, key=cv2.contourArea)
        else:
            return None

@dataclass
class RectanglePolygon:
    x: float
    y: float
    width: float
    height: float


class FixedRectangleLocator(Locator):
    def __init__(self, rect: RectanglePolygon):
        """
        rect: RectanglePolygon - coordinates of the rectangle relative to the frame.
        coordinate should be between 0 and 1, representing the percentage of the frame size.
        """
        self.rect = rect

    def find_page_contour(self, frame):
        """
        Returns a rectangle contour centered at the given location on the frame.
        """
        h, w = frame.shape[:2]
        x1 = int(self.rect.x * w)
        y1 = int(self.rect.y * h)
        x2 = int((self.rect.x + self.rect.width) * w)
        y2 = int((self.rect.y + self.rect.height) * h)
        return np.array([
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2]
        ], dtype=np.int32)
    
class CornerRectLocator(Locator):
    def __init__(
            self, 
            quartile:str, 
            width_prop:float, 
            height_prop:float,
            x_space:float=0.0,
            y_space:float=0.0
            ):
        # quartile: 0-3, representing the quarter of the frame to crop
        self.quartile = quartile # 'top-left', 'top-right', 'bottom-left', 'bottom-right'
        self.width = width_prop
        self.height = height_prop
        self.x_space = x_space  # Space between the rectangle and the frame edge in x direction
        self.y_space = y_space  # Space between the rectangle and the frame edge in y

    def find_page_contour(self, frame):
        """
        Returns a rectangle contour in the specified quartile of the frame.
        Quartiles: 'top-left', 'top-right', 'bottom-left', 'bottom-right'
        """
        h, w = frame.shape[:2]
        rect_w = int(self.width * w//2)
        rect_h = int(self.height * h//2)

        if self.quartile == 'top-left':
            x1, y1 = 0 + self.x_space* rect_w, 0 + self.y_space* rect_h
            x2, y2 = x1 + rect_w, y1 + rect_h
        elif self.quartile == 'top-right':
            x2, y1 = w - self.x_space* rect_w, 0 + self.y_space* rect_h
            x1, y2 = x2 - rect_w, y1 + rect_h
        elif self.quartile == 'bottom-left':
            x1, y2 = 0 + self.x_space* rect_w, h - self.y_space* rect_h
            x2, y1 = x1 + rect_w, y2 - rect_h
        elif self.quartile == 'bottom-right':
            x2, y2 = w - self.x_space* rect_w, h - self.y_space* rect_h
            x1, y1 = x2 - rect_w, y2 - rect_h
        else:
            raise ValueError("Quartile must be one of: 'top-left', 'top-right', 'bottom-left', 'bottom-right'")

        return np.array([
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2]
        ], dtype=np.int32)

    
#fixed Trapezoid locator, used for fixed rectangle detection
@dataclass
class TrapezoidPolygon:
    x_top_center: float   # Center x of top edge (0-1)
    y_top: float          # y of top edge (0-1)
    top_width: float      # width of top edge (0-1)
    x_bottom_center: float # Center x of bottom edge (0-1)
    y_bottom: float        # y of bottom edge (0-1)
    bottom_width: float    # width of bottom edge (0-1)

class FixedTrapezoidLocator(Locator):
    def __init__(self, trapezoid: TrapezoidPolygon):
        """
        trapezoid: TrapezoidPolygon - robust definition (see above)
        """
        self.trapezoid = trapezoid

    def find_page_contour(self, frame):
        """
        Returns a trapezoidal contour based on the robust coordinates.
        """
        h, w = frame.shape[:2]
        # Top edge
        y1 = int(self.trapezoid.y_top * h)
        cx1 = int(self.trapezoid.x_top_center * w)
        half_top = int((self.trapezoid.top_width * w) / 2)
        x1 = max(0, min(w-1, cx1 - half_top))  # top-left
        x2 = max(0, min(w-1, cx1 + half_top))  # top-right

        # Bottom edge
        y2 = int(self.trapezoid.y_bottom * h)
        cx2 = int(self.trapezoid.x_bottom_center * w)
        half_bottom = int((self.trapezoid.bottom_width * w) / 2)
        x3 = max(0, min(w-1, cx2 + half_bottom))  # bottom-right
        x4 = max(0, min(w-1, cx2 - half_bottom))  # bottom-left

        return np.array([
            [x1, y1],  # top-left
            [x2, y1],  # top-right
            [x3, y2],  # bottom-right
            [x4, y2]   # bottom-left
        ], dtype=np.int32)
