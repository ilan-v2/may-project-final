# demo of yolo model for debgging
import cv2
from ultralytics import YOLO
import math

def main():
    models_path = "../models/"  # Path to the models directory
    model = YOLO(models_path + "yolo11n.pt") # Load the YOLOv8 model
    cap = cv2.VideoCapture(0)  # Use camera index 0 (default)

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    while True:
        ret, frame = cap.read()
        # flip frame 
        frame = cv2.flip(frame, -1)  # flip the image 180 degrees, because the camera is upside down
    
        if not ret:
            break
        results = model(frame,stream= True)  # inference on the frame
        # coordinates
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int 
                # put box in frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

                # confidence
                conf = math.ceil(box.conf[0] * 100)/100  # confidence in percentage

                # class name
                cls_num = box.cls[0]  # class index
                cls_name = model.names[int(cls_num)]  # get class name
                
                # object details 
                org = (x1, y1 - 10)  # origin for text
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                color = (0, 255, 0)
                thickness = 2
                cv2.putText(frame, f"{cls_name} {conf}", org, font, font_scale, color, thickness)
        
        cv2.imshow("YOLO Detection (Live)", frame)
        key = cv2.waitKey(50)  
        if key == 27:  # ESC key to exit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()