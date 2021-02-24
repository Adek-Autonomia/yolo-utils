"""
Utils and handling for YOLO object predictions.
"""
import numpy as np
import cv2


IMAGE = 'path/to/img.jpg'
YOLO_CONFIG = 'cfg/yolov3_custom.cfg'
YOLO_WEIGHTS = 'backup/yolov3_custom_last.weights'


def get_net_layers(yconfig='cfg/yolov3_custom.cfg', yweights='backup/yolov3_custom_last.weights'):
    """
    Gets cv2.dnn.Net from darknet driectory
    """
    net = cv2.dnn.readNetFromDarknet(yconfig, yweights)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return net, output_layers


def get_predictions(net, img, out_layers):
    """
    Gets raw predictions from YOLO with image.
    """
    if isinstance(img, str):
        img = cv2.imread(img)
    cv2.resize(img, (416, 416))
    blob = cv2.dnn.blobFromImage(img)
    net.setInput(img)
    return net.forward(out_layers)


def get_labels(predictions, thresh):
    """
    Extracting labels from raw YOLO outputs.
    """
    class_ids = []
    confidences = []
    boxes = []
    width, height = (32, 32)     ## 416/13
    for out in predictions:      ## 3
        for detection in out:    ## 509, 2018, 8112
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > thresh:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    return confidences, class_ids, boxes
    


## ROS TESTING
img = cv2.imread(IMAGE)
net, outlayers = get_net_layers(YOLO_CONFIG, YOLO_WEIGHTS) ## Filepaths correct?
raw_preds = get_predictions(net, img, outlayers)
confs, class_ids, boxes = get_labels(raw_preds, .1)
for c, ci, b in confs, class_ids, boxes:
    print(f'Detected class {ci} at {b} with confidence {c}.')
if len(c) == 0:
    print('No objects found.')

