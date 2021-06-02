import cv2
from core.detector import detect

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, img = cap.read()
    
    if success:
        outputs = detect(img)

        for box in outputs['has_mask']:
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 3)

        for box in outputs['has_no_mask']:
            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)
    else:
        break
