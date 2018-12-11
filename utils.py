import cv2
import numpy as np
import tensorflow as tf
from scipy.ndimage.filters import gaussian_filter
from keras import backend as K


def fit_image(image, min_size):
    im_shape = np.shape(image)
    min_side = np.min(im_shape[:2])
    #min_side = im_shape[0]
    scale = np.divide((min_size - min_side), min_side)
    new_width = np.multiply((1.0 + scale), im_shape[1]).astype(int)
    new_height = np.multiply((1.0 + scale), im_shape[0]).astype(int)

    return cv2.resize(image, dsize=(new_width, new_height), interpolation=cv2.INTER_NEAREST)


def select_target(frame):
    while True:
        cv2.imshow("video", frame)

        key = cv2.waitKey(0) & 0xFF
        if key == 27:
            cv2.destroyAllWindows()
            break
        else:
            if 'selected_points' not in globals() or len(selected_points) < 4:
                print("Please select a target")
                continue

            return selected_points[-4:]


def on_mouse(event, x, y, _flags, _params):
    global selected_points
    if event == cv2.EVENT_LBUTTONDBLCLK:
        if 'selected_points' not in globals():
            selected_points = np.array([y, x])
            print(selected_points)
        else:
            selected_points = np.append(selected_points, [y, x])
            print(selected_points)


def to_yxhw(quad):
    w = quad[3] - quad[1]
    h = quad[2] - quad[0]
    x = quad[1] + 0.5 * w
    y = quad[0] + 0.5 * h
    return np.array([y, x, h, w])


def to_y1x1y2x2(quad):
    y1 = quad[0] - quad[2] * 0.5
    x1 = quad[1] - quad[3] * 0.5
    y2 = quad[0] + quad[2] * 0.5
    x2 = quad[1] + quad[3] * 0.5
    return np.array([y1, x1, y2, x2])


def fit_to_frame(frame_shape, template):
    template[0] = np.maximum(0, template[0])
    template[1] = np.maximum(0, template[1])
    template[2] = np.minimum(frame_shape[0], template[2])
    template[3] = np.minimum(frame_shape[1], template[3])

    return template


def generate_anchors(target, ratios=np.array([0.5, 1.0, 2.0])):
    _target = to_yxhw(target)
    target_scale = _target[2] * _target[3]
    scales = [target_scale/2.0, target_scale, target_scale*2.0]
    anchors = []

    for scale in scales:
        size_ratios = np.multiply(scale, ratios)

        a_width = np.sqrt(size_ratios)
        a_height = a_width[::-1]
        for width, height in zip(a_width, a_height):
            anchors.append([height, width])

    return np.array(anchors)


def generate_search_window(target, template_scale):
    _target = to_yxhw(target)
    _target[2:4] = np.tile(_target[2:4].max(), 2)
    _target[2:4] = np.multiply(_target[2:4], template_scale)
    _target = to_y1x1y2x2(_target)

    return np.array(_target)


def intersection_over_union(box, query_box):
    box_area = (query_box[2] - query_box[0]) * (query_box[3] - query_box[1])

    i_width = min(box[3], query_box[3]) - max(box[1], query_box[1])

    if i_width > 0:
        i_height = min(box[2], query_box[2]) - max(box[0], query_box[0])

        if i_height > 0:
            outer_union = (box[2] - box[0]) * (box[3] - box[1]) + box_area - i_width * i_height
            return i_width * i_height / outer_union

    return 0


def gt_list_to_bbox(gt_list):
    x_coords = gt_list[::2]
    y_coords = gt_list[1::2]

    return np.array([y_coords.min(), x_coords.min(), y_coords.max(), x_coords.max()])


def gauss_kernel(shape, sigma=1.0):
    k = np.zeros(shape)

    if shape[0] % 2 > 0:
        x = shape[0] // 2
    else:
        x = [shape[0] // 2 - 1, shape[0] // 2]

    if shape[1] % 2 > 0:
        y = shape[1] // 2
    else:
        y = [shape[1] // 2 - 1, shape[1] // 2]

    k[x, y] = 1
    return gaussian_filter(k, sigma)


def get_n_max(ft_map, n):
    ft_shape = np.shape(ft_map)

    max_inds = ft_map.flatten().argsort()[-n:][::-1]
    return np.array([divmod(a, ft_shape[1]) for a in max_inds])


def entropy_loss(template):
    def loss(y_true, y_pred):
        batch_t_slice = K.expand_dims(K.flatten(template), axis=0)
        batch_t_slice = K.cast(batch_t_slice, dtype=tf.float32)
        y_pred = K.batch_flatten(y_pred)
        y_true = K.flatten(y_true)
        y_false = tf.multiply(tf.subtract(y_true, 1), -1)
        labels = tf.stack([y_true, y_false], axis=1)
        v = tf.multiply(K.tile(K.sum(tf.multiply(y_pred, batch_t_slice), axis=1, keepdims=True), [1, 2]), labels)
        v = tf.divide(v, K.max(v))
        return tf.nn.sparse_softmax_cross_entropy_with_logits(labels=K.cast(y_true, dtype=tf.int32), logits=v)

    return loss
