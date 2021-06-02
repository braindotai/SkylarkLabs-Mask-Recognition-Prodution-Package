import cv2
import onnxruntime as rt
import numpy as np

class FaceDetection:
    def __init__(self, modelFile, faceThreshold = 0.9):
        sessOptions = rt.SessionOptions()
        sessOptions.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
        sessOptions.execution_mode = rt.ExecutionMode.ORT_PARALLEL
        self.inferenceSession = rt.InferenceSession(modelFile, sessOptions)
        self.faceThreshold = faceThreshold
        self.anchorsCache = {}

        self.featStride = [32, 16, 8]
        self.retinaAnchors = {
            32: np.array([[-248.,-248.,263.,263.],[-120.,-120.,135.,135.]], dtype=np.float32), 
            16: np.array([[-56.,-56.,71.,71.],[-24.,-24.,39.,39.]], dtype=np.float32), 
            8: np.array([[-8.,-8.,23.,23.],[0.,0.,15.,15.]], dtype=np.float32)
        }

    def detect(self, inputImage):
        resizedImage, resizedRatio = resize2TargetSize(inputImage)
        inputImageTensor = resizedImage.transpose(2,0,1)[np.newaxis]
        inferenceOutputs = self.inferenceSession.run([], {"input": inputImageTensor})
        faceOutputs = self.infer2Box(inferenceOutputs, resizedImage.shape[0:2], resizedRatio)
        return faceOutputs

    def infer2Box(self,outputs, imageSize, resizedRatio):
        faceBoxes = []
        faceScores = []
        maskScores = []

        featIdx = 0

        for currentStride in self.featStride:
            scaleFaceScores = outputs[featIdx].transpose(0,2,3,1).reshape(-1, 1)
            scaleFaceBoxes = outputs[featIdx+1]
            scaleMaskScores = outputs[featIdx+2].transpose(0,2,3,1).reshape(-1, 1)
            featIdx+=3
            
            height, width = scaleFaceBoxes.shape[2], scaleFaceBoxes.shape[3]
            if (height, width, currentStride) in self.anchorsCache:
                anchors = self.anchorsCache[(height, width, currentStride)]
            else:
                anchors = generateAnchor(height, width, currentStride, self.retinaAnchors[currentStride])
                anchors = anchors.reshape((height*width*2, 4))
                self.anchorsCache[(height, width, currentStride)] = anchors

            scaleFaceBoxes = scaleFaceBoxes.transpose(0,2,3,1).reshape(-1, 4)

            scaleFaceBoxes = bboxTransform(anchors, scaleFaceBoxes)
            scaleFaceBoxes = boundBoxes(scaleFaceBoxes, imageSize[0], imageSize[1])

            faceBoxes.append(scaleFaceBoxes)
            faceScores.append(scaleFaceScores)
            maskScores.append(scaleMaskScores)
        
        faceBoxes = np.vstack(faceBoxes)
        faceScores = np.vstack(faceScores)
        maskScores = np.vstack(maskScores)

        keepIdx = np.where(faceScores.ravel() >= self.faceThreshold)[0]
        faceBoxes = faceBoxes[keepIdx,:]
        faceScores = faceScores[keepIdx]
        maskScores = maskScores[keepIdx]

        faceBoxes /= resizedRatio

        faces = np.hstack((faceBoxes[:,0:4], faceScores)).astype(np.float32)
        keepIdx = nms(faces,0.4)
        faces = np.hstack((faces, maskScores))
        faces = faces[keepIdx, :]

        return faces

def resize2TargetSize(inputImage, maxSize=1080, targetSize=640):
    h,w,_ = inputImage.shape
    inputMaxSize = max(h,w)
    inputMinSize = min(h,w)
    resizedRatio = float(targetSize) / float(inputMinSize)
    if np.round(resizedRatio * inputMaxSize) > maxSize:
        resizedRatio = float(maxSize) / float(inputMaxSize)
    resizedImage = cv2.resize(inputImage, None, fx=resizedRatio, fy=resizedRatio, interpolation=cv2.INTER_LINEAR)
    return resizedImage, resizedRatio

def nms(dets, thresh):
    ### From Fast-RCNN implementation ###
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep

def generateAnchor(height, width, stride, baseAnchors):
    A = baseAnchors.shape[0]
    allAnchors = np.zeros((height, width, A, 4), dtype=np.float32)
    for iw in range(width):
        sw = iw * stride
        for ih in range(height):
            sh = ih * stride
            for k in range(A):
                allAnchors[ih, iw, k, 0] = baseAnchors[k, 0] + sw
                allAnchors[ih, iw, k, 1] = baseAnchors[k, 1] + sh
                allAnchors[ih, iw, k, 2] = baseAnchors[k, 2] + sw
                allAnchors[ih, iw, k, 3] = baseAnchors[k, 3] + sh
    return allAnchors

def bboxTransform(anchors, faceBoxes):
    ### From FaceBoxes Imp ###
    if anchors.shape[0] == 0:
        return np.zeros((0, faceBoxes.shape[1]))

    anchors = anchors.astype(np.float32, copy=False)

    widths = anchors[:, 2] - anchors[:, 0] + 1.0
    heights = anchors[:, 3] - anchors[:, 1] + 1.0
    ctr_x = anchors[:, 0] + 0.5 * (widths - 1.0)
    ctr_y = anchors[:, 1] + 0.5 * (heights - 1.0)

    pred_ctr_x = faceBoxes[:, 0] * widths + ctr_x
    pred_ctr_y = faceBoxes[:, 1] * heights + ctr_y
    pred_w = np.exp(faceBoxes[:, 2]) * widths
    pred_h = np.exp(faceBoxes[:, 3]) * heights

    pred_boxes = np.zeros(faceBoxes.shape)
    pred_boxes[:, 0] = pred_ctr_x - 0.5 * (pred_w - 1.0)
    pred_boxes[:, 1] = pred_ctr_y - 0.5 * (pred_h - 1.0)
    pred_boxes[:, 2] = pred_ctr_x + 0.5 * (pred_w - 1.0)
    pred_boxes[:, 3] = pred_ctr_y + 0.5 * (pred_h - 1.0)

    return pred_boxes

def boundBoxes(boxes, h,w):
    boxes[:,[0,2]] = np.clip(boxes[:,[0,2]],0,w - 1)
    boxes[:,[1,3]] = np.clip(boxes[:,[1,3]],0,h - 1)
    return boxes