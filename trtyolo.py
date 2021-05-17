"""
Module for using YOLOv3(v4) with TensorRT & CUDA. Contains runtime classes and functions, 
assuming YOLO is already in the .trt engine format.
Compatible with TensorRT 7+.
# ALWAYS import pycuda.autoinit before importing this module.
To change look of bounding boxes drawn on the image, do: \\
`trtyolo.TEXT_SCALE = 1.5`
"""
import numpy as np
import cv2
import pycuda.driver as cuda
import tensorrt as trt

import ctypes

from random import seed, shuffle
from colorsys import hsv_to_rgb

from typing import List, Dict, Tuple

from dataclasses import dataclass

try:
    ctypes.cdll.LoadLibrary('/home/adek/yoloutils/libyolo_layer.so')
except OSError as e:
    raise SystemExit('Shared object library (for YOLO layer in trt.PluginField) could not be found. Check ~/yoloutils') from e



# Visualization constants
ALPHA = 0.5
FONT = cv2.FONT_HERSHEY_PLAIN
TEXT_SCALE = 1.0
TEXT_THICKNESS = 1
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)






class YOLO_Visualization():
    """
    Visualization class for drawing detection bounding boxes over the image. 
    """
    def __init__(self, num_categories: int):
        """
        Args: \\
            num_categories - number of YOLO categories
        """
        self.num_categories = num_categories
        self.colors = self._generate_colors()
        self.cls_dict = self._get_cls_dict(num_categories)


    def _generate_colors(self) -> List[Tuple[int, int, int]]:
        """
        Generate list of colors corresponding to different classes. \\
        Returns: \\
            bgrs - list of tuples of B,G,R color values
        """
        ## Different hues
        hsvs = [[float(x) / self.num_categories, 1., 0.7] for x in range(self.num_categories)]

        seed(420)
        shuffle(hsvs)

        ## Turn every HSV 3-tuple into an RGB 3-tuple
        rgbs = list(map(lambda x: list(hsv_to_rgb(*x)), hsvs))

        ## Convert to BGR - used in opencv & cv_bridge
        bgrs = [(int(rgb[2] * 255), int(rgb[1] * 255),  int(rgb[0] * 255))
                for rgb in rgbs]
        return bgrs


    def _draw_boxed_text(self, 
                        img: np.array, 
                        text: str, 
                        textloc: Tuple[int, int], 
                        color: Tuple[int, int, int])-> np.array:
        """
        Draws a box with text over the image. \\
        Text style, font, ... can be set as constants in this module. \\
        This function modifies the image inplace! \\
        Args: \\
            img - uint8 np array corresponding to the image \\
            text - text to overlay (string) \\
            textloc - 2-tuple corresponding to top left corner of the textbox \\
            color - 3-tuple (or list) with BGR values of the box color \\
        Returns: \\
            np.array corresponding to image with text and textbox overlayed onto it
        """
        h, w, _ = img.shape

        ## Out of bounds box - exit function
        if textloc[0] >= w or textloc[1] >= h:
            return img

        
        margin = 3
        size = cv2.getTextSize(text, FONT, TEXT_SCALE, TEXT_THICKNESS)
        w = size[0][0] + margin * 2
        h = size[0][1] + margin * 2

        # The patch is used to draw boxed text
        patch = np.zeros((h, w, 3), dtype=np.uint8)

        ## Set every element (numpy elipsis constant) to the box color
        patch[...] = color

        cv2.putText(patch, text, (margin+1, h-margin-2), FONT, TEXT_SCALE,
                    WHITE, thickness=TEXT_THICKNESS, lineType=cv2.LINE_8)
        cv2.rectangle(patch, (0, 0), (w-1, h-1), BLACK, thickness=1)

        w = min(w, w - textloc[0])  # clip overlay at image boundary
        h = min(h, h - textloc[1])

        # Overlay the boxed text onto region of interest in img
        region = img[textloc[1] : textloc[1]+h, textloc[0] : textloc[0]+w, :]

        ## Overlay box & text partly transparently
        cv2.addWeighted(patch[0:h, 0:w, :], ALPHA, region, 1 - ALPHA, 0, region)

        return img


    def draw(self, 
            img: np.array, 
            boxes: List[List[int]], 
            confidences: List[float], 
            class_ids: List[int]) -> np.array:
        """
        Draws bounding boxes and adds detection info over the image. \\
        Args: \\
            img - uint8 np.array corresponding to image \\
            boxes - list of bounding boxes alerady clipped to image bounds \\
            confidences - list of confidences corresponding to each bounding box \\
            class_ids - list of class numbers corresponding to each bounding box \\
            Returns: \\
            image with detections drawn onto it
        """

        ## Draw all boxes onto image before returning it.
        for bb, cf, cl in zip(boxes, confidences, class_ids):
            assert img.dtype == np.uint8

            cl = int(cl)
            x_min, y_min, x_max, y_max = bb[0], bb[1], bb[2], bb[3]

            ## Get color for this class.
            color = self.colors[cl]

            ## Draw 2-pixel thick box.
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)

            txt_loc = (max(x_min+2, 0), max(y_min+2, 0))
            cls_name = self.cls_dict[cl]

            ## Format label text, e.g. 'Dog 0.87'.
            txt = '{} {:.2f}'.format(cls_name, cf)

            ## Add detection info.
            img = self._draw_boxed_text(img, txt, txt_loc, color)
        return img


    @staticmethod
    def _get_cls_dict(self, num_categories: int) -> Dict:
        """
        Gets dict class ids and COCO class names. Static methos. \\
        Args: \\
            num_categories - number of categories model was trained on
        Returns: \\
            -dict of class id (int) keys with class name (str) values
        """
        COCO_CLASSES_LIST = [
            'person',
            'bicycle',
            'car',
            'motorbike',
            'aeroplane',
            'bus',
            'train',
            'truck',
            'boat',
            'traffic light',
            'fire hydrant',
            'stop sign',
            'parking meter',
            'bench',
            'bird',
            'cat',
            'dog',
            'horse',
            'sheep',
            'cow',
            'elephant',
            'bear',
            'zebra',
            'giraffe',
            'backpack',
            'umbrella',
            'handbag',
            'tie',
            'suitcase',
            'frisbee',
            'skis',
            'snowboard',
            'sports ball',
            'kite',
            'baseball bat',
            'baseball glove',
            'skateboard',
            'surfboard',
            'tennis racket',
            'bottle',
            'wine glass',
            'cup',
            'fork',
            'knife',
            'spoon',
            'bowl',
            'banana',
            'apple',
            'sandwich',
            'orange',
            'broccoli',
            'carrot',
            'hot dog',
            'pizza',
            'donut',
            'cake',
            'chair',
            'sofa',
            'pottedplant',
            'bed',
            'diningtable',
            'toilet',
            'tvmonitor',
            'laptop',
            'mouse',
            'remote',
            'keyboard',
            'cell phone',
            'microwave',
            'oven',
            'toaster',
            'sink',
            'refrigerator',
            'book',
            'clock',
            'vase',
            'scissors',
            'teddy bear',
            'hair drier',
            'toothbrush',
        ]
    
        if num_categories <= len(COCO_CLASSES_LIST):
            return {i : name for i, name in enumerate(COCO_CLASSES_LIST) if i < num_categories}
        else:
            return {i : 'Class{}'.format(i) for i in range(num_categories)}



class HostDeviceMemory():
    """
    Helper class for host & device memory pairs. More readable than just a tuple. \\
    host - memory buffer on host (RAM) \\
    device - memory on device (GPU)
    """
    def __init__(self, host, device):
        self.host = host
        self.device = device


    def __repr__(self):
        return 'Host: ' + str(self.host) + '\nDevice: ' + str(self.device)




class TRTYOLO():
    """
    TensorRT-based YOLO class for very fast inference on GPU.
    """
    def __init__(self, 
                enginepath: str,
                input_shape: Tuple[int, int] = (416, 416), 
                num_categories :int = 80, 
                confidence_threshold: float = .5, 
                nms_threshold: float = .7):
        """
        Args: \\
            enignepath - path to .trt engine file \\
            input_shape - tuple with image height and width (in that order). 416x416 by default \\
            num_categories - number of categories to recognize. 80 by default \\
            confidence_threshold - minimum confidence to recognize detection. 0.5 by default \\
            nms_threshold - when to perform NMS. 0.7 by default \\
        """
        self.logger = trt.Logger(trt.Logger.INFO)
        self.engine = self._load_engine(enginepath)

        self.input_shape = input_shape
        self.num_categories = num_categories
        self.conf_thresh = confidence_threshold
        self.nms_thresh = nms_threshold

        try:
            self.context = self.engine.create_execution_context()
            self.inputs, self.outputs, self.bindings, self.stream = self._allocate_net_bindings()
        except Exception as e:
            raise RuntimeError('Failed to allocate CUDA resources') from e


    def __del__(self):
        """
        Free all the memory previously allocated for CUDA stream, engine ins & outs.
        """
        del self.inputs
        del self.outputs
        del self.stream

    def __exit__(self):
        """Alias for __del__()."""
        self.__del__()


    def _load_engine(self, enginepath: str) -> trt.ICudaEngine:
        """
        Load TensorRT engine from a .trt file. \\
        Args: \\
            enginepath - path to engine (.trt) file \\
        Returns: \\
            engine - trt.ICudaEngine instance read in from file
        """
        with open(enginepath, 'rb') as f, trt.Runtime(self.logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def _allocate_net_bindings(self):
        """
        On __init__, allocate memory for engine inputs, outputs & bindings as well as CUDA stream. \\
        If this raises ValueErrors, support TRT 6 & lower should be added. \\
        Returns: \\
            inputs - host/device memory pairs corresponding to input buffer in RAM & input buffer on GPU \\
            outputs - ---||--- output buffers ---||--- \\
            bindings -  array of linear indices into PyCUDA.driver 's memory corresponding to net bindings (input/outpu buffer addresses) \\
            stream - PyCUDA.Stream instance for sending data between RAM and GPU
        """
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()

        ## 1 input + 2 outputs OR 1 input + 3 outputs
        ## Number of outputs (2/3) is the number of 'yolo' layers in net.
        assert 3 <= len(self.engine) <= 4 

        ## Preallocate memory on host & GPU for every binding, so that inputs & outputs 
        ## can be quickly accessed.
        ## Binding type (in/out) is stored in the engine.
        for binding in self.engine:

            ## Tensor shape of input or output: tuple
            binding_dims = self.engine.get_binding_shape(binding)

            ## This assumes len(binding_dims) == 4, meaning
            ## batch case is given explicitly, which only works in TensorRT 7+
            if len(binding_dims) == 4:
                binding_size = trt.volume(binding_dims)
            elif len(binding_dims) == 3:
                raise ValueError('''TensorRT version is 6 or lower or has implicit batch case, 
                                which is not supported. Add implicit batch support or 
                                check TensorRT version.''')
            else:
                raise ValueError('Shape of binding {} is wrong: {}'.format(binding, binding_dims))

            ## Cast dtype to numpy dtype. Pagelocked memory allocated by driver (line below)
            ## uses numpy pagelocked internally, and therefore it needs to have a numpy dtype.
            binding_dtype = trt.nptype(self.engine.get_binding_dtype(binding))

            ## Allocate host & GPU memory buffers.
            ## Using pagelocked memory prevents the driver from copying the memory every time it's
            ## sent to GPU, resulting in a 2x speed increase. This is a linear piece of memory 
            ## because binding_size is the product of a shape tuple and therefore an int.
            ## cuda.mem_alloc(), correspondingly, returns a linear piece of GPU memory
            ## with the same size.
            host_memory = cuda.pagelocked_empty(binding_size, binding_dtype)
            device_memory = cuda.mem_alloc(host_memory.nbytes)

            ## Casting a DeviceAllocation to int returns a linear index into the context memory.
            bindings.append(int(device_memory))

            if self.engine.binding_is_input(binding):
                inputs.append(HostDeviceMemory(host_memory, device_memory))

            else:
                ## Each grid has 3 anchors, each anchor generates a detection
                ## with 7 float32 values: [x, y, w, h, box_conf, class_id, class_probability]
                assert binding_size % 7 == 0
                outputs.append(HostDeviceMemory(host_memory, device_memory))


        return inputs, outputs, bindings, stream


    def _preprocess(self, img: np.array) -> np.array:
        """
        Resize image before sending it into YOLO net and convert it to BRG (?!) colorspace. \\
        Args: \\
            img - image to run detection on as an int8 numpy array \\
        Returns: \\
            float32 numpy array correspinding to resized image
        """
        img = cv2.resize(img, (self.input_shape[1], self.input_shape[0]))

        ## First convert to RGB, then to BRG. //<-why!?
        ## Is original YOLO trained on BRG colorspace ?????
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose((2, 0, 1)).astype(np.float32)

        ## Rescale pixel values to (0, 1) range.
        img /= 255.

        return img


    ## NOTE JIT compile this? (or maybe just the inside of the `while`)
    ## This is all numpy arrays, so seems like it should compile.
    ## Does numba wirk with numpy #functions#? (like np.maximum)
    def _nms_boxes(self, detections: np.array) -> np.array:
        """
        Apply non-max suppression on a subset of YOLO detections. \\
        Args: \\
            detections - Nx7 numpy array with all detections of one class \\
        Returns: \\
            indices of the boxes to keep, referring to items in detections 
        """
        x_coords = detections[:, 0]
        y_coords = detections[:, 1]
        widths = detections[:, 2]
        heights = detections[:, 3]
        box_confidences = detections[:, 4] * detections[:, 6]

        box_areas = widths * heights

        ## Argsort returns indices in ascending order, so we do ::-1 to get max indices first.
        ## This is an array of indices, not values!
        boxes_ordered = box_confidences.argsort()[::-1] 

        indices_to_keep = []

        while boxes_ordered.size > 0:
            ## Index of current item (argsort returns indices).
            i = boxes_ordered[0]
            indices_to_keep.append(i)

            ## For every remaining item in coordinates get more central value (coordinates of 
            ## intersection corners).
            ## xx1, yy1 are intersection's topleft corner x, y coordinates.
            ## xx2, yy2 are intersection's bottom-right corner coordinates.
            xx1 = np.maximum(x_coords[i], x_coords[boxes_ordered[1:]])
            yy1 = np.maximum(y_coords[i], y_coords[boxes_ordered[1:]])
            xx2 = np.minimum(x_coords[i] + widths[i], x_coords[boxes_ordered[1:]] + widths[boxes_ordered[1:]])
            yy2 = np.minimum(y_coords[i] + heights[i], y_coords[boxes_ordered[1:]] + heights[boxes_ordered[1:]])

            ## Calculate intersection for all other boxes area to later calculate IoUs.
            inters_widths = np.maximum(0.0, xx2 - xx1 + 1)
            inters_heights = np.maximum(0.0, yy2 - yy1 + 1)
            intersections = inters_widths * inters_heights

            ## Calclulate union area values, for all boxes with current box, and calculate IoUs. 
            unions = (box_areas[i] + box_areas[boxes_ordered[1:]] - intersections)
            ious = intersections / unions

            ## Find which indices to still consider as separate boxes.
            ## np.where returns a 1-tuple, so to get array [0] is needed.
            indexes = np.where(ious <= self.nms_thresh)[0]

            ## Shift indices to keep by 1, to synchronize with the fact that
            ## indexes does not have current item.
            ## Will not go out of bounds because indexes are all remaining items,
            ## but boxes_ordered also considers current element.
            boxes_ordered = boxes_ordered[indexes + 1]

        keep = np.array(indices_to_keep)
        return indices_to_keep


    ## NOTE JIT compile this?
    ## See TRTYOLO._nms_boxes() note.
    def _postprocess(self, net_outputs: List[np.array], img_shape: Tuple[int, int, int]) -> Tuple[np.array, np.array, np.array]:
        """
        Postprocess yolo output image. Reshape and rescale, apply NMS. \\
        Args:\\
            net_outputs - 2- or 3-long list of Nx7 numpy arrays corresponding to anchor boxes \\
            img_shape - shape of the original image\\
        Returns:\\
            boxes - numpy array of detection anchor boxes\\
            scores - numpy array of confidence scores for each detection \\
            classes - numpy array of class ids for each detection
        """
        all_detections = []

        for output in net_outputs:

            ## Ensure outputs have final dimension 7
            detections = output.reshape((-1, 7))

            #print(detections.shape)
            ## Filter only detections over threshold (box_conf*class_prob => thresh)
            detections = detections[detections[:, 4] * detections[:, 6] >= self.conf_thresh]
            #print(detections.shape)
            all_detections.append(detections)

        all_detections = np.concatenate(all_detections, axis=0)

        ## Ensure correct return type even if nothing has actually been detected
        if len(all_detections) == 0:
            print(all_detections)
            print('Using 0len arrays')
            boxes = np.zeros((0, 4), dtype=np.int)
            scores = np.zeros((0,), dtype=np.float32)
            classes = np.zeros((0,), dtype=np.float32)


        ## Rescale box sizes to original image
        img_h, img_w = img_shape[0], img_shape[1]
        all_detections[:, 0:4] *= np.array([img_w, img_h, img_w, img_h], dtype=np.float32)
    
        #return detections

        ## NMS

        ## Create an array before to make sure something is returned even if 
        ## NMS returns nothing
        nms_detections = np.zeros((0, 7), dtype=all_detections.dtype)

        ## Only consider each detected class once -> use a set
        for class_id in set(all_detections[:, 5]):

            ## Do NMS for this class where it's been detected
            idxs = np.where(all_detections[:, 5] == class_id)
            this_cls_detections = all_detections[idxs]

            boxes_to_keep = self._nms_boxes(this_cls_detections)

            ## This concatenate step works with the earlier created nms_detections array
            ## To make sure this variable is a np.array and not None
            nms_detections = np.concatenate(
                [nms_detections, this_cls_detections[boxes_to_keep]], axis=0)

        ## Reshape coordinates to (N,1)
        xx = nms_detections[:, 0].reshape(-1, 1)
        yy = nms_detections[:, 1].reshape(-1, 1)
        ww = nms_detections[:, 2].reshape(-1, 1)
        hh = nms_detections[:, 3].reshape(-1, 1)

        ## Create boxes using coordinates
        ## Adding 0.5 to boxes ensures correct mathematical rounding when casting to int (below)
        boxes = np.concatenate([xx, yy, xx+ww, yy+hh], axis=1) + 0.5
        boxes = boxes.astype(np.int)

        ## Calculate box confidence scores as box_conf*class_probability
        scores = nms_detections[:, 4] * nms_detections[:, 6]
        classes = nms_detections[:, 5]

        return boxes, scores, classes




    def detect(self, img: np.array) -> Tuple[np.array, np.array, np.array]:
        """
        Perform a full detection on the input image, complete with pre- and postprocessing as well as nms. \\
        Args:\\
            img - raw image to perform detection on.\\
        Returns:\\
            boxes - numpy array of detection anchor boxes\\
            scores - numpy array of confidence scores for each detection \\
            classes - numpy array of class ids for each detection
        """

        ## Resize image to fit net.
        image_resized = self._preprocess(img)

        ## ================ INFERENCE =============================

        ## Send image to network input.
        self.inputs[0].host = np.ascontiguousarray(image_resized)

        ## Transfer input data to the GPU.
        ## htod = host to device.
        [cuda.memcpy_htod_async(inp.device, inp.host, self.stream) for inp in self.inputs]

        ## Run inference on GPU.
        ## Can do asynchronously (faster) because stream will be synchronized later 
        ## with pycuda.
        ## Using execute_async_v2 because TensorRT version is 7+.
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)

        ## Transfer predictions back from the GPU to RAM.
        ## dtoh = device to host.
        [cuda.memcpy_dtoh_async(out.host, out.device, self.stream) for out in self.outputs]

        # Synchronize the stream to make sure all calculations are done.
        self.stream.synchronize()

        ## =================== END OF INFERENCE =====================

        # Return only the host outputs.
        net_outputs = [out.host for out in self.outputs]

        #return net_outputs

        return self._postprocess(net_outputs, img.shape)

        ## Apply NMS, resize to original size.
        #boxes, scores, classes = self._postprocess(net_outputs, img.shape)

        ## Clip boxes so they don't go out of the image.
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, img.shape[1]-1)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, img.shape[0]-1)

        return boxes, scores, classes











