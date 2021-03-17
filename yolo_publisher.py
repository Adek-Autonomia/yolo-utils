#!usr/env/bin/python3

"""
tensorrt_demos-based publisher-subscriber combo using yolov3 [416].

"""
import cv_bridge
import rospy
from sensor_msgs.msg import Image as ImgMsg

from tensorrt_demos.yolo_with_plugins import TrtYOLO, get_input_shape
from tensorrt_demos.utils.yolo_classes import get_cls_dict
from tensorrt_demos.utils.visualization import BBoxVisualization

import pycuda.autoinit

import argparse
import cv2


def parse_args():
    """ Arguments for publisher/subscriber node """
    parser = argparse.ArgumentParser(description='Publisher arguments: image and model')
    parser.add_argument('-i', '--image', type=str, help='Path to image', required=True)
    parser.add_argument('-m', '--model', type=str, 
    help=('[yolov3|yolov3-tiny|yolov3-spp|yolov4|yolov4-tiny]-'
              '[{dimension}], where dimension could be a single '
              'number (e.g. 288, 416, 608) or WxH (e.g. 416x256)'),
              default='yolov3-416', required=True)
    args = parser.parse_args()
    return args






class YOLOPublisher():
    """ YOLO snippet extraction and publishing to \'yolo_predictions\'. """

    def __init__(self, model:str):
        try:
            self.bridge = cv_bridge.CvBridge() ## Bridge
            self.pub = rospy.Publisher('yolo_predictions_2', ImgMsg, queue_size=10) ## Publisher
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
    args = parse_args()
    IMAGE = args.image
    MODEL = args.model
    ypub = YOLOPublisher(MODEL)
    rospy.init_node('YOLO_PREDICTIONS')
    ypub.do_publishing(IMAGE)
    

if __name__ == '__main__':
    main()


