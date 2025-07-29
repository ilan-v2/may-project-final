from ultralytics import YOLO
from page_locator import Locator


class YoloLocator(Locator):
    def __init__(self, model = None):
        self.page_like_classes = ["page", "book", "document", "paper"]
        self.model = model if model is not None else YOLO("yolo11n.pt")

    def get_page_contour(self, frame):
        # inference model
        results = self.model(frame)[0]
        detections = results.boxes
        # Process results to extract page contour
        for box in results.boxes:
            x1, y1, x2, y2, conf, cls_id = box
            class_name = self.model.names[int(cls_id)]

            if class_name in self.page_like_classes:
                return (x1, y1, x2, y2)

        return None