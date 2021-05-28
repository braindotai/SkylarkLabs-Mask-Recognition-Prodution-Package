from PIL import Image
import cv2
import mediapipe as mp
from core.detector import detect

def fancyDraw(img, bbox, label, l = 30, t = 3, rt = 1):
    x, y, w, h = bbox
    x1, y1 = x + w, y + h

    color = (120, 255, 0) if 'No' not in label else (0, 90, 255)

    overlay = img.copy()
    cv2.rectangle(overlay, bbox, color, -1) # A filled rectangle
    alpha = 0.2 # Transparency factor.
    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    cv2.rectangle(img, bbox, color, 1)
    
    cv2.line(img, (x, y), (x + l, y), color, t)
    cv2.line(img, (x, y), (x, y+l), color, t)

    cv2.line(img, (x1, y), (x1 - l, y), color, t)
    cv2.line(img, (x1, y), (x1, y+l), color, t)

    cv2.line(img, (x, y1), (x + l, y1), color, t)
    cv2.line(img, (x, y1), (x, y1 - l), color, t)

    cv2.line(img, (x1, y1), (x1 - l, y1), color, t)
    cv2.line(img, (x1, y1), (x1, y1 - l), color, t)

    cv2.putText(img, label, (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN, 1.5, color, 2)

    return img

cap = cv2.VideoCapture(0)

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection(0.74)

i = -1
while cap.isOpened():
    success, img = cap.read()
    
    if success:
        i += 1
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = faceDetection.process(imgRGB)

        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                query_image = img[bbox[1]: bbox[1] + bbox[3], bbox[0]: bbox[0] + bbox[2]]

                label = detect(query_image)
                img = fancyDraw(img, bbox, label)

        cv2.imshow("Image", img)
        cv2.waitKey(1)
    else:
        break
