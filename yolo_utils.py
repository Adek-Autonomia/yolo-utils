"""
Utils and handling for YOLO object predictions.
"""
import numpy as np



def get_labels(predictions, imgshape=(416, 416, 3), with_classes=True, format_corners=False):
    """
    Label extraction from YOLO model predictions. Transforms a grid of YOLO labels into array of
    coordinates (and classes). \\
    Params: \\
    -predictions    -> raw output from YOLO detector \\
    -imgshape       -> shape of images fed into YOLO \\
    -with_classes   -> whether to include object class indices in output \\
    -fromat_corners -> whether to return labels as top-left and bottom-right corners instead of
    middle, width, height \\
    Returns: \\
    If not format_corners: \\
    -Array of shape (num_objects, 4) where 2 is (x_middle, y_middle, h, w) if not with_classes, otherwise
    (num_objects, 5) where 3 is (x_middle, y_middle, h, w, class_index)
    Otherwise: \\
    -Same as above, except first 4 variables are (x_top, y_left, x_bottom, y_right)
    """
    predictions = np.squeeze(predictions) ## redundant dimension reduction
    ## Y is the vertical dimension of the image, X is the horizontal
    gridtile_y = imgshape[0] // predictions.shape[0] ## Single grid tile size
    gridtile_x = imgshape[1] // predictions.shape[1]

    labels = [] ## Final outputs
    classes = []
    for x in range(len(predictions)):
        for y in range(len(predictions[x])):
            hasobject, (ym, xm, h, w) = predictions[x, y, :5]
            if hasobject:                    
                y_middle = int(y*gridtile_y + ym*gridtile_y)  ## Grid tile local to
                x_middle = int(x*gridtile_x + xm*gridtile_x)  ## image global coords
                height = int(gridtile_y*w)
                width = int(gridtile_x*h)
                if not format_corners:
                    labels.append([y_middle, x_middle, height, width])
                else:
                    y_top = y_middle - (height//2) ## Top right corner
                    x_left = x_middle - (width//2)
                    y_bottom = y_middle + (height//2) ## Bottom left corner
                    x_right = y_middle + (width//2)
                    labels.append([y_top, x_left, y_bottom, x_right])
                if with_classes:
                    classes.append(np.argmax(labels[x, y, 5:]))

    if with_classes:
        return np.concatenate((labels, classes), axis=-1)
    return np.array(labels)
    





