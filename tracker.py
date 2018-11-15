from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras import optimizers
from keras import backend as K
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import shift

import cv2
import numpy as np
import tensorflow as tf

# Architecture
tf.flags.DEFINE_integer("output_layer_block",    4,      "The block output we want to use")
tf.flags.DEFINE_integer("template_size",         10,     "Size of template")
tf.flags.DEFINE_integer("strength_queue_length", 100,    "Number of maximums used in failure detection")
tf.flags.DEFINE_float("padding_divider",         5.0,    "Is the divider used in computing padding amount (w+h)/divider")

# Similarity map computation
tf.flags.DEFINE_integer("gauss_sigma",           8,      "Standard deviation of gauss filter")
tf.flags.DEFINE_integer("map_stride",            1,      "Stride when computing similarity map")
tf.flags.DEFINE_float("search_window_scale",     3.0,    "Scale of search window relative to target size")

# Model tuning
tf.flags.DEFINE_float("positive_threshold",      0.7,    "IOU for positives")
tf.flags.DEFINE_float("negative_threshold",      0.3,    "IOU for negatives")
tf.flags.DEFINE_integer("tuning_batch_size",     32,     "Batch size when fine-tuning the model")
tf.flags.DEFINE_integer("tuning_epochs",         4,      "Number of epochs when fine-tuning the model")
tf.flags.DEFINE_float("tuning_learning_rate",    1e-4,   "Learning rate when fine-tuning the model")

# Target update \ Long-term detection
tf.flags.DEFINE_integer("anchor_score_mode",     0,      "Mode: 0 - cross-correlation, 1 - euclidean")
tf.flags.DEFINE_float("bad_detection_thresh",    3.1,    "When detection factor is considered bad")
tf.flags.DEFINE_float("good_detection_thresh",   0.9,    "When detection factor is considered good")
tf.flags.DEFINE_float("anchor_std_multiplier",   2.1,    "Anchor mean score threshold")
tf.flags.DEFINE_float("update_alpha",            0.02,   "Factor for template updating speed")

FLAGS = tf.flags.FLAGS


def track_object(video, model, gt_file=None, save_out=None):
    font = cv2.FONT_HERSHEY_PLAIN
    cv2.namedWindow("video")
    cv2.setMouseCallback("video", on_mouse)
    stride = np.power(2, FLAGS.output_layer_block - 1)

    frame_num = 0
    frame = video[frame_num]

    if gt_file is not None:
        frame_target = gt_list_to_bbox(np.round(list(map(float, gt_file.readline().split(","))))).astype(int)
    else:
        frame_target = select_target(frame)

    pad = (frame_target[2] - frame_target[0] + frame_target[3] - frame_target[1]) / FLAGS.padding_divider
    pad_frame_target = to_yxhw(frame_target)
    pad_frame_target[2:] += pad
    pad_frame_target = fit_to_frame(np.shape(frame), to_y1x1y2x2(pad_frame_target)).astype(int)
    pad_target = pad_frame_target / stride

    scale_factor = np.array([FLAGS.template_size/(pad_target[2] - pad_target[0]), FLAGS.template_size/(pad_target[3] - pad_target[1])])

    frame_in = frame[pad_frame_target[0]:pad_frame_target[2], pad_frame_target[1]:pad_frame_target[3], :]
    frame_in = cv2.resize(frame_in, dsize=(FLAGS.template_size*stride, FLAGS.template_size*stride), interpolation=cv2.INTER_CUBIC)
    frame_in = np.expand_dims(frame_in, axis=0)
    template = np.squeeze(model.predict(frame_in))

    iou_list = []
    template_scale = FLAGS.search_window_scale

    anchors = generate_anchors(pad_target)

    # cv2.rectangle(frame, tuple(frame_target[1::-1]), tuple(frame_target[3:1:-1]), (255, 255, 255), 2)
    # cv2.imshow('video', frame)
    # cv2.waitKey(0)

    if save_out is not None:
        out = cv2.VideoWriter(save_out.split("/")[-1] + '-' + str(FLAGS.output_layer_block) + '.avi', -1, 24.0, np.shape(frame)[1::-1])
    else:
        out = None

    strength_scores = list()
    use_gauss = False

    for i in range(len(video) - 1):
        frame_num += 1
        frame = video[frame_num]
        search_window = generate_search_window(np.shape(frame), pad_frame_target, template_scale).astype(int)
        # frame_t_coor = (search_window * stride).astype(int)
        frame_in = frame[search_window[0]:search_window[2], search_window[1]:search_window[3], :]
        shape_in = np.round(np.shape(frame_in)[:2] * scale_factor).astype(int)
        frame_in = cv2.resize(frame_in, dsize=(shape_in[1], shape_in[0]), interpolation=cv2.INTER_CUBIC)
        frame_in = np.expand_dims(frame_in, axis=0)
        features = np.squeeze(model.predict(frame_in), axis=0)

        ft_shape = np.shape(features)
        pad = FLAGS.template_size // 2

        pad_shape = np.array([ft_shape[0]+FLAGS.template_size, ft_shape[1]+FLAGS.template_size, ft_shape[2]])
        ft_pad = np.zeros(pad_shape)
        ft_pad[pad:ft_shape[0] + pad, pad:ft_shape[1] + pad, :] = features
        features = ft_pad

        sim_map = cc_distance_map(features, template, normalize=False)
        # sim_map = K.conv2d(features, template, (1, 1), "same")

        sim_map_mean = sim_map.mean()
        #cv2.imshow('video', fit_image(sim_map / sim_map.max(), 480))
        #cv2.waitKey(0)

        if use_gauss:
            gauss_filter = gauss_kernel(np.shape(sim_map), FLAGS.gauss_sigma)
            d = np.round((to_yxhw(pad_target)[:2] - to_yxhw(search_window//8)[:2]) * scale_factor).astype(int)
            gauss_filter = shift(gauss_filter, d, cval=0)
            sim_map = np.multiply(gauss_filter, sim_map)

        new_target, max_score, max_slice = compute_target(anchors, template, sim_map, features, pad_target, scale_factor)

        target_col = (0, 0, 255)

        strength = max_score**2 / sim_map_mean
        strength_scores.insert(0, strength)
        if len(strength_scores) > FLAGS.strength_queue_length:
            strength_scores.pop()

        lt_detection = np.mean(strength_scores) / strength

        if lt_detection > FLAGS.bad_detection_thresh:
            t_target = to_yxhw(pad_target)
            t_target[:2] = to_yxhw(new_target)[:2]
            pad_target = to_y1x1y2x2(t_target)
            pad_target += np.tile(search_window[:2]//8, 2)
            template_scale = FLAGS.search_window_scale * 2

            use_gauss = False
        else:
            pad_target = new_target
            pad_target += np.tile(search_window[:2]//8, 2)
            template_scale = FLAGS.search_window_scale

            use_gauss = True

        pad_target_h = pad_target[2] - pad_target[0]
        pad_target_w = pad_target[3] - pad_target[1]
        scale_factor = np.array([FLAGS.template_size / pad_target_h, FLAGS.template_size / pad_target_w])

        if lt_detection < FLAGS.good_detection_thresh and np.shape(max_slice)[:2] == (FLAGS.template_size, FLAGS.template_size):
            # print(np.shape(max_slice)[:2] == (FLAGS.template_size, FLAGS.template_size))
            template = template * (1 - FLAGS.update_alpha) + max_slice * FLAGS.update_alpha
            target_col = (0, 255, 255)

            #if lt_detection < FLAGS.good_detection_thresh / 2:
            #     train(model, frame, template, pad_target, scale_factor, sim_map, search_window, stride=stride, epochs=FLAGS.tuning_epochs, learning_rate=FLAGS.tuning_learning_rate)

        #print("------------------------------------------")
        #print("LTDetection strength: " + str(lt_detection))

        anchors = generate_anchors(pad_target)

        #target_h = target[2] - target[0]
        #target_w = target[3] - target[1]
        #pad_target_scale = (target_h * target_w) / (pad_target_h * pad_target_w)

        pad_frame_target = np.multiply(pad_target, stride).astype(int)

        pad_frame_target_w = pad_frame_target[3] - pad_frame_target[1]
        pad_frame_target_h = pad_frame_target[2] - pad_frame_target[0]
        pad_amount = (pad_frame_target_w + pad_frame_target_h) / (FLAGS.padding_divider + 2)
        # target_pad = (pad_frame_target_w + pad_frame_target_h) * pad_target_scale // 2
        frame_target = to_yxhw(pad_frame_target)
        frame_target[2] -= pad_amount
        frame_target[3] -= pad_amount
        frame_target = to_y1x1y2x2(frame_target).astype(int)

        #t_slice_show = np.sum(template, axis=2)
        #t_slice_show = process_map(np.multiply(np.divide(t_slice_show, t_slice_show.max()), 255), img_shape)

        #ft_map_coor = np.array([search_window[0], search_window[1], search_window[2] - (target[2] - target[0] - 1), search_window[3] - (target[3] - target[1] - 1)])


        #frame_t_coor = np.multiply(search_window, stride)
        #ft_map = process_map(np.multiply(ft_map, 255), np.shape(frame[frame_t_coor[0]:frame_t_coor[2], frame_t_coor[1]:frame_t_coor[3]]))
        #frame[frame_t_coor[0]:frame_t_coor[2], frame_t_coor[1]:frame_t_coor[3]] = frame[
        #                                                                          frame_t_coor[0]:frame_t_coor[2],
        #                                                                          frame_t_coor[1]:frame_t_coor[
        #                                                                              3]] * 0.2 + ft_map * 0.8

        #for anchor in anchors:
        #    a = np.multiply(anchor, stride).astype(int)
        #    cv2.rectangle(frame, tuple(a[1::-1]), tuple(a[3:1:-1]), (0, 255, 0), 1)

        cv2.rectangle(frame, tuple(frame_target[1::-1]), tuple(frame_target[3:1:-1]), target_col, 2)
        cv2.rectangle(frame, tuple(pad_frame_target[1::-1]), tuple(pad_frame_target[3:1:-1]), (255, 255, 255), 2)
        # cv2.rectangle(frame, tuple(search_window[1::-1]), tuple(search_window[3:1:-1]), (255, 0, 0), 2)
        cv2.putText(frame, str(np.round(lt_detection, decimals=2)), (frame_target[1], frame_target[0] - 2), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

        if gt_file is not None:
            gt_box = gt_list_to_bbox(np.round(list(map(float, gt_file.readline().split(","))))).astype(int)
            cv2.rectangle(frame, tuple(gt_box[1::-1]), tuple(gt_box[3:1:-1]), (255, 255, 0), 2)
            iou_list.append(intersection_over_union(gt_box, frame_target))

        if save_out is not None:
            out.write(frame)

        cv2.imshow('video', frame)
        cv2.waitKey(1)

    cv2.destroyAllWindows()
    if gt_file is not None:
        gt_file.close()

    if save_out is not None:
        out.release()

    return iou_list


def cc_distance_map(features, ft_slice, normalize=False):
    slice_shape = np.shape(ft_slice)
    gauss = gauss_kernel(slice_shape[:2], 2)
    gauss += (1 - gauss) * 0.01
    t_filter = np.transpose(np.tile(gauss, [slice_shape[2], 1, 1]), [2, 1, 0])
    t_slice = np.multiply(ft_slice, t_filter)

    # show_slice = np.sum(t_filter, axis=2)
    # cv2.imshow('video', fit_image(show_slice / show_slice.max(), 480))
    # cv2.waitKey(0)
    # t_slice = ft_slice

    slice_shape = np.shape(t_slice)
    slice_w = slice_shape[1]
    slice_h = slice_shape[0]

    ft_shape = np.shape(features)

    stride = FLAGS.map_stride
    steps_y = np.arange(0, ft_shape[0] - slice_h + 1, stride)
    steps_x = np.arange(0, ft_shape[1] - slice_w + 1, stride)

    sim_map = np.zeros(shape=[len(steps_y), len(steps_x)])
    for y1 in steps_y:
        y2 = y1 + slice_h
        for x1 in steps_x:
            x2 = x1 + slice_w

            feature_slice = features[y1:y2, x1:x2, :]

            if normalize:
                n = FLAGS.template_size * FLAGS.template_size * slice_shape[2]
                stds = np.std(feature_slice) * np.std(t_slice)
                feature_slice = (feature_slice - np.mean(feature_slice)) / (stds * n)
                template = (t_slice - np.mean(t_slice)) / (stds * n)
            else:
                template = t_slice

            # dist = np.sum(np.multiply(template, feature_slice))
            dist = np.tensordot(template, feature_slice, axes=((0, 1, 2), (0, 1, 2)))

            sim_map[y1//stride][x1//stride] = dist

    return sim_map


def compute_target(anchors, template, sim_map, features, prev_target, scale_factor):
    max_coor = np.where(sim_map == sim_map.max())
    max_coor = np.array([max_coor[0][0], max_coor[1][0]])
    max_coor *= FLAGS.map_stride

    a_score = []
    top_slice = None
    top_score = None
    anchor_mask = np.ones(len(anchors), dtype=bool)
    for ind, anchor in enumerate(anchors):
        anchor *= scale_factor
        pad = FLAGS.template_size // 2
        anchor_box = to_y1x1y2x2([max_coor[0] + pad, max_coor[1] + pad, anchor[0], anchor[1]])
        anchor_box = np.around(fit_to_frame(np.shape(features), anchor_box)).astype(int)

        if anchor[0] > 0 and anchor[1] > 0:
            anchor_slice = features[anchor_box[0]:anchor_box[2], anchor_box[1]:anchor_box[3], :]
            anchor_slice = cv2.resize(anchor_slice, dsize=(FLAGS.template_size, FLAGS.template_size), interpolation=cv2.INTER_CUBIC)

            sim = 0
            if FLAGS.anchor_score_mode == 0:
                sim = np.tensordot(template, anchor_slice, axes=((0, 1, 2), (0, 1, 2)))
            elif FLAGS.anchor_score_mode == 1:
                sim = np.sqrt(np.sum((anchor_slice - template) ** 2))

            a_score.append(sim)

            if top_score is None or (FLAGS.anchor_score_mode == 0 and top_score < sim) or (FLAGS.anchor_score_mode == 1 and top_score > sim):
                top_score = sim
                top_slice = anchor_slice
        else:
            anchor_mask[ind] = False

    anchors = anchors[anchor_mask]
    a_score = np.array(a_score)
    if top_score is not None and FLAGS.anchor_score_mode == 0 and a_score.mean() + a_score.std() * FLAGS.anchor_std_multiplier < top_score:
        top_anchor = anchors[a_score.argmax()]
        target = np.array([max_coor[0], max_coor[1], top_anchor[0], top_anchor[1]])
        target /= np.tile(scale_factor, 2)
        target = to_y1x1y2x2(target)
    elif top_score is not None and FLAGS.anchor_score_mode == 1 and a_score.mean() - a_score.std() * FLAGS.anchor_std_multiplier > top_score:
        top_anchor = anchors[a_score.argmin()]
        target = np.array([max_coor[0], max_coor[1], top_anchor[0], top_anchor[1]])
        target /= np.tile(scale_factor, 2)
        target = to_y1x1y2x2(target)

        top_score = np.tensordot(template, top_slice, axes=((0, 1, 2), (0, 1, 2)))
    else:
        target = to_yxhw(prev_target)
        target[:2] = max_coor / scale_factor
        target = to_y1x1y2x2(target)

        t_slice = [max_coor[0], max_coor[1], max_coor[0] + FLAGS.template_size, max_coor[1] + FLAGS.template_size]
        top_slice = features[t_slice[0]:t_slice[2], t_slice[1]:t_slice[3], :]

        top_score = np.tensordot(template, top_slice, axes=((0, 1, 2), (0, 1, 2)))

    return target, top_score, top_slice


def train(model, frame, template, target, scale_factor, ft_map, search_window, stride=16, max_batch_size=256, in_batch_size=128, epochs=3, learning_rate=1e-5):
    max_points = get_n_max(ft_map, max_batch_size)
    max_points = np.divide(max_points, scale_factor)
    max_points += search_window[:2]
    max_points *= stride

    c_target = to_yxhw(target) * stride
    samples = np.array([to_y1x1y2x2([x, y, c_target[2], c_target[3]]) for x, y in max_points])

    pos_samples = []
    neg_samples = []
    pos_iou = []
    neg_iou = []

    frame_target = to_y1x1y2x2(c_target)
    for sample in samples:
        iou = intersection_over_union(sample, frame_target)

        int_sample = sample.astype(int)
        frame_sample = frame[int_sample[0]:int_sample[2], int_sample[1]:int_sample[3], :]

        if np.shape(frame_sample)[0] > stride and np.shape(frame_sample)[1] > stride:
            frame_sample = cv2.resize(frame_sample, dsize=(FLAGS.template_size*stride, FLAGS.template_size*stride), interpolation=cv2.INTER_CUBIC)
            if iou > FLAGS.positive_threshold:
                pos_samples.append(frame_sample)
                pos_samples.append(np.fliplr(frame_sample))
                pos_samples.append(np.flipud(frame_sample))
                pos_iou.append(iou)
                pos_iou.append(iou)
                pos_iou.append(iou)
            elif iou < FLAGS.negative_threshold:
                neg_samples.append(frame_sample)
                neg_iou.append(iou)

    pos_samples = pos_samples[:in_batch_size//3]
    neg_samples = neg_samples[:in_batch_size - len(pos_samples)]

    samples = np.array(pos_samples + neg_samples)

    labels = np.zeros(len(samples))
    #labels[:len(pos_samples), 0] = 1
    labels[:len(pos_samples)] = 1  #np.array(pos_iou)[:len(pos_samples)]
    #labels[len(pos_samples):, 1] = 1
    labels[len(pos_samples):] = 0  #(np.array(neg_iou)[:len(neg_samples)] - 1)

    loss = softmax_loss(template)
    #optimizer = optimizers.Adam(lr=learning_rate)
    optimizer = optimizers.SGD(lr=learning_rate, momentum=0.9, decay=5e-4)

    model.compile(optimizer=optimizer, loss=loss)

    for epoch in range(epochs):
        perm = np.random.permutation(len(samples))
        samples = samples[perm]
        labels = labels[perm]
        losses = []

        print("=========== " + str(epoch+1) + ". EPOCH ===========")
        for batch in range(np.ceil(len(samples)/FLAGS.tuning_batch_size).astype(int)):
            od = batch*FLAGS.tuning_batch_size
            do = od + FLAGS.tuning_batch_size
            x = samples[od:do]
            y = labels[od:do]

            loss = model.train_on_batch(x, y)
            losses.append(loss)
            print(str(batch+1) + ". Batch loss: " + str(loss))
            #model.fit(x=np.array(samples), y=np.array(labels), epochs=epochs, batch_size=FLAGS.tuning_batch_size, shuffle=True)

        print("Mean loss: " + str(np.mean(losses)))
    #print("Pos length:" + str(len(pos_samples)))
    #print("Neg length:" + str(len(neg_samples)))

    return


def siam_loss(template):
    #gauss = gauss_kernel(np.shape(template)[:2], 3)
    #t_slice = np.multiply(template, gauss[..., None])

    def loss(y_true, y_pred):
        batch_t_slice = K.repeat_elements(K.expand_dims(K.flatten(template), axis=0), FLAGS.tuning_batch_size, axis=0)
        batch_t_slice = K.cast(batch_t_slice, dtype=tf.float32)
        y_pred = K.batch_flatten(y_pred)
        y_true = K.flatten(y_true)
        v = K.sum(tf.multiply(y_pred, batch_t_slice), axis=1, keepdims=True)
        v = tf.divide(v, K.max(v))
        return K.log(tf.add(1.0, K.exp(tf.negative(tf.multiply(y_true, v)))))

    return loss


def softmax_loss(template):
    slice_shape = np.shape(template)
    gauss = gauss_kernel(slice_shape[:2], 2) + 0.01
    t_filter = np.transpose(np.tile(gauss, [slice_shape[2], 1, 1]), [1, 2, 0])
    t_slice = np.multiply(template, t_filter)

    def loss(y_true, y_pred):
        batch_t_slice = K.expand_dims(K.flatten(t_slice), axis=0)
        batch_t_slice = K.cast(batch_t_slice, dtype=tf.float32)
        y_pred = K.batch_flatten(y_pred)
        y_true = K.flatten(y_true)
        y_false = tf.multiply(tf.subtract(y_true, 1), -1)
        labels = tf.stack([y_true, y_false], axis=1)
        v = tf.multiply(K.tile(K.sum(tf.multiply(y_pred, batch_t_slice), axis=1, keepdims=True), [1, 2]), labels)
        v = tf.divide(v, K.max(v))
        return tf.nn.sparse_softmax_cross_entropy_with_logits(labels=K.cast(y_true, dtype=tf.int32), logits=v)

    return loss


def fit_image(image, min_size):
    im_shape = np.shape(image)
    min_side = np.min(im_shape[:2])
    #min_side = im_shape[0]
    scale = np.divide((min_size - min_side), min_side)
    new_width = np.multiply((1.0 + scale), im_shape[1]).astype(int)
    new_height = np.multiply((1.0 + scale), im_shape[0]).astype(int)

    return cv2.resize(image, dsize=(new_width, new_height), interpolation=cv2.INTER_CUBIC)


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


def generate_search_window(ft_shape, target, template_scale):
    _target = to_yxhw(target)
    _target[2:4] = np.tile(_target[2:4].max(), 2)
    _target[2:4] = np.multiply(_target[2:4], template_scale)
    _target = fit_to_frame(ft_shape, to_y1x1y2x2(_target))

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


if __name__ == "__main__":
    tf.app.run()