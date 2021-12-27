#!usr/env/bin/python3

"""
tensorrt_demos-based publisher-subscriber combo using yolov3 [416].
"""

import cv_bridge
import rospy
from sensor_msgs.msg import Image as ImgMsg

import pycuda.autoinit ## IMPORTANT - before trt_demos imports

from tensorrt_demos.yolo_with_plugins import TrtYOLO, get_input_shape
from tensorrt_demos.utils.yolo_classes import get_cls_dict
from tensorrt_demos.utils.visualization import BBoxVisualization

import cv2

MODEL = 'yolov3-416'
IMAGE = '/home/adek/dog.jpg'


class YOLOPublisher():
    """ YOLO snippet extraction and publishing to \'yolo_predictions\'. """

    def __init__(self, model:str):
        try:
            self.bridge = cv_bridge.CvBridge() ## Bridge
            self.pub = rospy.Publisher('yolo_predictions', ImgMsg, queue_size=10) ## Publisher
            self.sub = rospy.Subscriber('yolo_predictions', ImgMsg, self.callback) ## Subsciber for saving
            h, w = get_input_shape(model)
            self.net = TrtYOLO(model, (h, w)) ## Tensorrt optimized yolo   
        except Exception as e:
            raise RuntimeError('Instance initialization failed') from e


    def do_publishing(self, image):
        
        img = cv2.imread(image) ## image
        
        ## Processing
        boxes, scores, classes = self.detect(img)
        img = self.draw(img, boxes, scores, classes)

        ## Publishing
        try:
            print('Publishing...')
            imgmsg = self.bridge.cv2_to_imgmsg(img, 'bgr8')
            self.pub.publish(imgmsg)
        except Exception as e:
            raise RuntimeError('Image publishing failed') from e


    def detect(self,   img):
        try:
            print('Attempting detection...')
            return self.net.detect(img)
        except Exception as e:
            raise RuntimeError('TrtYOLO detection failed') from e


    def draw(self, img, boxes, scores, classes):
        """ TrtYolo-provided image processing """
        try:
            print('Appending detection boxes to image...')
            vis = BBoxVisualization(get_cls_dict(80))
            img = vis.draw_bboxes(img, boxes, scores, classes)
            return img
        except Exception as e:
            raise RuntimeError('Post-detection image processing failed') from e


    def callback(self, data):
        """ Subscriber callback to receive and save image """
        try:
            print('Received image.')
            image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            cv2.imwrite('YOLO_TEST.jpg', image)
            print('Saved image.')
        except Exception as e:
            raise RuntimeError('Subscriber callback failed') from e



def main():
    ypub = YOLOPublisher(MODEL)
    rospy.init_node('YOLO_PREDICTIONS')
    ypub.do_publishing(IMAGE)
    

if __name__ == '__main__':
    main()
