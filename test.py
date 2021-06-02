import cv2
from numpy.lib.type_check import imag
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
