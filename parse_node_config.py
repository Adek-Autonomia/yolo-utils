"""
Module for parsing config file for nodes.
"""
import json

from yolo_nodes.msg import Imagelist, IDlist
from sensor_msgs.msg import Image as ImgMsg

import rospy
from cv_bridge import CvBridge
from trtyolo import TRTYOLO
from keras.models import load_model

from preprocessing import ImgPreprocessor

CFG_FILE_PATH = '/home/adek/config/config.json'

def get_cfg():
    """Get cofing info from file."""
    with open(CFG_FILE_PATH, 'r') as cfg:
        configdict = json.loads(cfg)
    return configdict


def setup_tsc(tscobject, testmode: bool):
    """
    Setup TSC node object with publisher, subscriber and other config info.
    Modifies object inplace.
    """
    cfg = get_cfg()
    tscobject.sub = rospy.Subscriber(cfg['topics']['sign_image_topic'], 
                                    Imagelist, 
                                    tscobject._callback)
    if not testmode:
        tscobject.pub = rospy.Publisher(cfg['topics']['sign_classes_topic'], 
                                        IDlist,
                                        queue_size=cfg['tsc']['publisher_queue_size'])
    
    tscobject.node_name = cfg['node_names']['tsc']
    tscobject.img_width, tscobject.imgheight = cfg['tsc']['input_shape']
    tscobject.model = load_model(cfg['tsc']['modelpath'])
    tscobject.preprocessing = ImgPreprocessor()
    tscobject.bridge = CvBridge()
    tscobject.img_width, tscobject.img_height = cfg['tsc']['input_shape']



def setup_tlc(tlcobject, testmode: bool):
    """
    Setup TLC node object with publisher, subscriber and other config info.
    Modifies object inplace.
    """
    cfg = get_cfg()
    tlcobject.sub = rospy.Subscriber(cfg['topics']['light_image_topic'], 
                                    Imagelist, 
                                    tlcobject._callback)
    if not testmode:
        tlcobject.pub = rospy.Publisher(cfg['topics']['light_classes_topic'], 
                                        IDlist, 
                                        queue_size=cfg['tlc']['publisher_queue_size'])

    tlcobject.node_name = cfg['node_names']['tlc']
    tlcobject.imshape = cfg['tlc']['input_shape']
    tlcobject.model = load_model(cfg['tlc']['modelpath'])
    tlcobject.bridge = CvBridge()
    



def setup_yolo(yoloobject, testmode: bool):
    """
    Setup YOLO node object with publisher, subscriber and other config info.
    Modifies object inplace.
    """
    cfg = get_cfg()
    yoloobject.sub = rospy.Subscriber(cfg['topics']['camera_image_topic'],
                                      ImgMsg,
                                      yoloobject._callback)
    yoloobject.tsc_indx, yoloobject.tlc_indx = cfg['yolo']['sign_detections_index'], cfg['yolo']['light_detections_index']

    yoloobject.tsc_pub = rospy.Publisher(cfg['topics']['sign_image_topic'],
                                         Imagelist,
                                         queue_size=cfg['yolo']['publisher_queue_size'])
    yoloobject.tlc_pub = rospy.Publisher(cfg['topics']['light_image_topic'],
                                         Imagelist,
                                         queue_size=cfg['yolo']['publisher_queue_size'])
    yoloobject.yolo = TRTYOLO(cfg['yolo']['modelpath'],
                              tuple(cfg['yolo']['input_shape']),
                              cfg['yolo']['num_classes'],
                              cfg['yolo']['conf_thresh'],
                              cfg['yolo']['nms_thresh'])
    yoloobject.bridge = CvBridge()
    yoloobject.nodename = cfg['node_names']['yolo']
    if testmode:
        yoloobject.tsc_sub = rospy.Subscriber(cfg['topics']['sign_image_topic'],
                                              Imagelist,
                                              yoloobject._testing_callback)
        yoloobject.tlc_sub = rospy.Subscriber(cfg['topics']['light_image_topic'],
                                              Imagelist,
                                              yoloobject._testing_callback)