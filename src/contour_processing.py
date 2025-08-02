import cv2
import numpy as np
from typing import Optional, Tuple

def _order_points(pts: np.ndarray) -> np.ndarray:
    """
    Order four points in the order:
    top-left, top-right, bottom-right, bottom-left.
    """
    s = pts.sum(axis=1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.vstack([tl, tr, br, bl]).astype("float32")


def extract_warped_roi(
    frame: np.ndarray,
    contour_input: np.ndarray,
    output_size: Tuple[int, int] = (300, 400),
    epsilon_ratio: float = 0.02
) -> Optional[np.ndarray]:
    """
    Extract a top-down (bird’s-eye) view of the quadrilateral defined
    by `contour` in `frame`.

    Parameters
    ----------
    frame : np.ndarray
        Source image.
    contour_input : np.ndarray
        Input contour (e.g. from cv2.findContours).
    output_size : (width, height)
        Desired size of the warped ROI.
    epsilon_ratio : float
        Approximation accuracy: contour perimeter × this ratio → epsilon.

    Returns
    -------
    warped : np.ndarray or None
        The warped ROI of shape (height, width), or None if the contour
        doesn’t approximate to four points.
    """
 # — normalize input into a valid cv2 contour array —
    contour = np.asarray(contour_input, dtype=np.float32)
    # if someone passed an Nx2 array / list of (x,y) points, reshape it
    if contour.ndim == 2 and contour.shape[1] == 2:
        contour = contour.reshape(-1, 1, 2)

    # 1) perimeter & polygon approximation
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon_ratio * peri, True)
    if approx.shape[0] != 4:
        return None

    # 2) sort to TL,TR,BR,BL
    pts = approx.reshape(4, 2)
    rect = _order_points(pts)

    # 3) build destination rectangle
    w, h = output_size
    dst = np.array([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1]
    ], dtype="float32")

    # 4) warp
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(frame, M, (w, h))

    return warped