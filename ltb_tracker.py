from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import oxuva
import os
import time

from keras import optimizers
from keras.applications.vgg16 import VGG16
from keras import models
from scipy.ndimage.interpolation import shift
from utils import *

import tensorflow as tf

# Architecture
tf.flags.DEFINE_integer("output_layer_block",    4,      "The block output we want to use")
tf.flags.DEFINE_integer("template_size",         8,     "Size of template")
tf.flags.DEFINE_integer("strength_queue_length", 100,    "Number of maximums used in failure detection")
tf.flags.DEFINE_float("padding_divider",         4.5,    "Is the divider used in computing padding amount (w+h)/divider")

# Similarity map computation
tf.flags.DEFINE_integer("gauss_sigma",           8,      "Standard deviation of gauss filter")
tf.flags.DEFINE_integer("map_stride",            1,      "Stride when computing similarity map")
tf.flags.DEFINE_float("search_window_scale",     2.5,    "Scale of search window relative to target size")

# Model tuning
tf.flags.DEFINE_float("positive_threshold",      0.7,    "IOU for positives")
tf.flags.DEFINE_float("negative_threshold",      0.3,    "IOU for negatives")
tf.flags.DEFINE_integer("tuning_batch_size",     32,     "Batch size when fine-tuning the model")
tf.flags.DEFINE_integer("tuning_epochs",         4,      "Number of epochs when fine-tuning the model")
tf.flags.DEFINE_float("tuning_learning_rate",    1e-4,   "Learning rate when fine-tuning the model")

# Target update \ Long-term detection
tf.flags.DEFINE_float("bad_detection_thresh",    2.4,    "When detection factor is considered bad")
tf.flags.DEFINE_float("good_detection_thresh",   0.7,    "When detection factor is considered good")
tf.flags.DEFINE_float("anchor_std_multiplier",   2.0,    "Anchor mean score threshold")
tf.flags.DEFINE_float("update_alpha",            0.00,   "Factor for template updating speed")

tf.flags.DEFINE_string("data_dir",              'D://tracker-testing//long-term-tracking-benchmark//dataset',       "Dataset dir")
tf.flags.DEFINE_string("predictions_dir",       'D://tracker-testing//long-term-tracking-results//predictions',     "Prediction dir")
tf.flags.DEFINE_string("data",                  'dev',                                                             "data")
tf.flags.DEFINE_boolean("verbose",              True,                                                               "v")
tf.flags.DEFINE_string("tracker",               'siamrp_lt',                                                        "tracker")

FLAGS = tf.flags.FLAGS


def main():
    tracker_id = 'siamrp_lt'
    tracker_preds_dir = os.path.join(FLAGS.predictions_dir, FLAGS.data, tracker_id)
    if not os.path.exists(tracker_preds_dir):
        os.makedirs(tracker_preds_dir, 0o0755)

    tasks_file = os.path.join(FLAGS.data_dir, 'tasks', FLAGS.data + '.csv')
    with open(tasks_file, 'r') as fp:
        tasks = oxuva.load_dataset_tasks_csv(fp)
    # tracks_file = os.path.join(args.data_dir, 'annotations', args.data + '.csv')
    # with open(tracks_file, 'r') as fp:
    #     tracks = oxuva.load_annotations_csv(fp)
    # tasks = {key: oxuva.make_task_from_track(track) for key, track in tracks.items()}

    imfile = lambda vid, t: os.path.join(
        FLAGS.data_dir, 'images', FLAGS.data, vid, '{:06d}.jpeg'.format(t))

    for key, task in tasks.items():
        vid, obj = key
        if FLAGS.verbose:
            print(vid, obj)
        preds_file = os.path.join(tracker_preds_dir, '{}_{}.csv'.format(vid, obj))
        if os.path.exists(preds_file):
            continue

        im = cv2.imread(imfile(vid, task.init_time), cv2.IMREAD_COLOR)
        im_shape = np.shape(im)

        print(task.init_rect)
        print(im_shape)

        region = {'x': int(task.init_rect['xmin']*im_shape[1]), 'y': int(task.init_rect['ymin']*im_shape[0]),
                  'width': int((task.init_rect['xmax'] - task.init_rect['xmin'])*im_shape[1]),
                  'height': int((task.init_rect['ymax'] - task.init_rect['ymin'])*im_shape[0])}

        tracker = SiamRP_LT(image=im, region=region)
        preds = oxuva.SparseTimeSeries()
        start = time.time()
        for t in range(task.init_time + 1, task.last_time + 1):
            im = cv2.imread(imfile(vid, t), cv2.IMREAD_COLOR)
            preds[t] = tracker.track(im)
        dur_sec = time.time() - start
        if FLAGS.verbose:
            fps = (task.last_time - task.init_time + 1) / dur_sec
            print('fps {:.3g}'.format(fps))

        tmp_preds_file = os.path.join(tracker_preds_dir, '{}_{}.csv.tmp'.format(vid, obj))
        with open(tmp_preds_file, 'w') as fp:
            oxuva.dump_predictions_csv(vid, obj, preds, fp)
        os.rename(tmp_preds_file, preds_file)


class SiamRP_LT(object):
    def __init__(self, image, region):
        self.stride = np.power(2, FLAGS.output_layer_block - 1)
        self.template_size = FLAGS.template_size
        self.padding_divider = FLAGS.padding_divider
        self.strength_queue_length = FLAGS.strength_queue_length
        self.use_gauss = False
        self.scores = list()
        self.model = init_cnn()

        pad = int((region['width'] + region['height']) / self.padding_divider)
        region['width'] += pad
        region['height'] += pad
        region['x'] -= pad//2
        region['y'] -= pad//2

        t_region = fit_to_frame(np.shape(image), np.array([region['y'], region['x'], region['y'] + region['height'], region['x'] + region['width']]))
        im_input = image[t_region[0]:t_region[2], t_region[1]:t_region[3], :]
        im_input = cv2.resize(im_input, dsize=(self.template_size * self.stride, self.template_size * self.stride), interpolation=cv2.INTER_CUBIC)
        im_input = np.expand_dims(im_input, axis=0)
        self.template = np.squeeze(self.model.predict(im_input))

        self.window_scale = FLAGS.search_window_scale
        self.target = np.array([region['y'], region['x'], region['y'] + region['height'], region['x'] + region['width']])

    def track(self, image):
        pad = int(np.sum(to_yxhw(self.target)[2:]) / self.padding_divider)
        pad_target = to_yxhw(self.target)
        pad_target[2:] += pad
        pad_target = to_y1x1y2x2(pad_target)
        ft_target = pad_target / self.stride
        anchors = generate_anchors(ft_target)

        scale_factor = np.array((self.template_size / (ft_target[2] - ft_target[0]), self.template_size / (ft_target[3] - ft_target[1])))

        search_window = generate_search_window(pad_target, self.window_scale).astype(int)
        sw_h = search_window[2] - search_window[0]
        sw_w = search_window[3] - search_window[1]
        im_input = np.zeros([sw_h, sw_w, 3])
        im_shape = np.shape(image)

        sw_region = {'ymin': max(0, -search_window[0]),
                     'ymax': min(sw_h, sw_h + im_shape[0] - search_window[2]),
                     'xmin': max(0, -search_window[1]),
                     'xmax': min(sw_w, sw_w + im_shape[1] - search_window[3])}
        im_region = {'ymin': max(0, search_window[0]),
                     'ymax': min(im_shape[0], search_window[2]),
                     'xmin': max(0, search_window[1]),
                     'xmax': min(im_shape[1], search_window[3])}

        im_input[sw_region['ymin']:sw_region['ymax'], sw_region['xmin']:sw_region['xmax'], :] = image[im_region['ymin']:im_region['ymax'], im_region['xmin']:im_region['xmax'], :].copy()
        shape_in = np.round(np.array([sw_h, sw_w]) * scale_factor).astype(int)
        im_input = cv2.resize(im_input, dsize=(shape_in[1], shape_in[0]), interpolation=cv2.INTER_CUBIC)
        im_input = np.expand_dims(im_input, axis=0)
        features = np.squeeze(self.model.predict(im_input), axis=0)

        ft_shape = np.shape(features)
        pad = FLAGS.template_size // 2

        pad_shape = np.array([ft_shape[0] + self.template_size, ft_shape[1] + self.template_size, ft_shape[2]])
        ft_pad = np.zeros(pad_shape)
        ft_pad[pad:ft_shape[0] + pad, pad:ft_shape[1] + pad, :] = features
        features = ft_pad

        sim_map = self.compute_distance_map(features)
        orig_sim_map = sim_map.copy()

        if self.use_gauss:
            gauss_filter = gauss_kernel(np.shape(sim_map), FLAGS.gauss_sigma)
            d = np.round((to_yxhw(ft_target)[:2] - to_yxhw(search_window // self.stride)[:2]) * scale_factor).astype(int)
            gauss_filter = shift(gauss_filter, d, cval=0)
            sim_map = np.multiply(gauss_filter, sim_map)

        # cv2.imshow('video', fit_image(sim_map / sim_map.max(), 480))
        # cv2.waitKey(0)
        new_target, max_score, max_slice = self.compute_target(anchors, sim_map, features, ft_target, scale_factor)

        peak = np.array([to_yxhw(new_target)[:2], to_yxhw(ft_target)[2:]]).flat
        peak *= np.tile(scale_factor, 2)
        peak = to_y1x1y2x2(peak)
        sl_region = to_yxhw(generate_search_window(peak, FLAGS.search_window_scale))
        sl_region = fit_to_frame(np.shape(orig_sim_map), to_y1x1y2x2(sl_region)).astype(int)
        side_lobe = orig_sim_map[sl_region[0]:sl_region[2], sl_region[1]:sl_region[3]]

        strength = max_score**2 / side_lobe.mean()
        confidence = np.mean(self.scores) / strength

        self.scores.insert(0, strength)
        if len(self.scores) > self.strength_queue_length:
            self.scores.pop()

        if confidence > FLAGS.bad_detection_thresh:
            ft_target = to_yxhw(ft_target)
            ft_target[:2] = to_yxhw(new_target)[:2]
            ft_target = to_y1x1y2x2(ft_target)
            ft_target += np.tile(search_window[:2] / self.stride, 2)

            self.window_scale = FLAGS.search_window_scale * 2
            self.use_gauss = False
            present = False
            self.scores.pop()
        else:
            ft_target = new_target
            ft_target += np.tile(search_window[:2] / self.stride, 2)
            self.window_scale = FLAGS.search_window_scale

            self.use_gauss = True
            present = True

        if confidence < FLAGS.good_detection_thresh:
            self.template = self.template * (1 - FLAGS.update_alpha) + max_slice * FLAGS.update_alpha

            # if confidence < FLAGS.good_detection_thresh / 2:
            #     train(frame, ft_target, orig_sim_map, search_window, epochs=FLAGS.tuning_epochs, learning_rate=FLAGS.tuning_learning_rate)

        new_target = np.multiply(ft_target, self.stride)

        target_w = new_target[3] - new_target[1]
        target_h = new_target[2] - new_target[0]
        pad_amount = (target_w + target_h) / (self.padding_divider + 2)

        new_target = to_yxhw(new_target)
        new_target[2] -= pad_amount
        new_target[3] -= pad_amount
        min_x = 0 - (new_target[3]+1)//2
        max_x = im_shape[1] + (new_target[3]+1)//2
        min_y = 0 - (new_target[2]+1)//2
        max_y = im_shape[0] + (new_target[2] + 1) // 2
        new_target[0] = min(max_y, max(min_y, new_target[0]))
        new_target[1] = min(max_x, max(min_x, new_target[1]))
        new_target[2] = max(2, new_target[2])
        new_target[3] = max(2, new_target[3])
        self.target = to_y1x1y2x2(new_target)

        output_target = {"xmin": self.target[1]/im_shape[1], "ymin": self.target[0]/im_shape[0],
                         "xmax": self.target[3]/im_shape[1], "ymax": self.target[2]/im_shape[0]}

        if present:
            cv2.rectangle(image, tuple(self.target[1::-1].astype(int)), tuple(self.target[3:1:-1].astype(int)), (0,0,255) , 2)

        cv2.rectangle(image, tuple(search_window[1::-1].astype(int)), tuple(search_window[3:1:-1].astype(int)), (255,0,0) , 2)
        cv2.putText(image, str(np.round(confidence, decimals=2)), (int(self.target[1]), int(self.target[0]) - 2), cv2.FONT_HERSHEY_PLAIN, 1,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('image', image)
        cv2.waitKey(1)

        return oxuva.make_prediction(present, score=1/confidence, **output_target)

    def compute_distance_map(self, features, normalize=False):
        slice_shape = np.shape(self.template)
        gauss = gauss_kernel(slice_shape[:2], 2)
        gauss += (1 - gauss) * 1e-2
        t_filter = np.transpose(np.tile(gauss, [slice_shape[2], 1, 1]), [2, 1, 0])
        t_slice = np.multiply(self.template, t_filter)

        #show_slice = np.sum(t_filter, axis=2)
        #cv2.imshow('video', fit_image(show_slice / show_slice.max(), 480))
        #cv2.waitKey(0)
        #t_slice = self.template

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

                sim_map[y1 // stride][x1 // stride] = dist

        return sim_map

    def compute_target(self, anchors, sim_map, features, ft_target, scale_factor):
        max_coor = np.where(sim_map == sim_map.max())
        max_coor = np.array([max_coor[0][0], max_coor[1][0]])
        max_coor *= FLAGS.map_stride

        a_score = []
        top_score = None
        top_slice = None
        anchor_mask = np.ones(len(anchors), dtype=bool)
        for ind, anchor in enumerate(anchors):
            anchor *= scale_factor
            pad = self.template_size // 2
            anchor_box = to_y1x1y2x2([max_coor[0] + pad, max_coor[1] + pad, anchor[0], anchor[1]])
            anchor_box = np.around(fit_to_frame(np.shape(features), anchor_box)).astype(int)

            if anchor[0] > 0 and anchor[1] > 0:
                anchor_slice = features[anchor_box[0]:anchor_box[2], anchor_box[1]:anchor_box[3], :]
                anchor_slice = cv2.resize(anchor_slice, dsize=(self.template_size, self.template_size),
                                          interpolation=cv2.INTER_CUBIC)

                sim = np.tensordot(self.template, anchor_slice, axes=((0, 1, 2), (0, 1, 2)))

                # a_score.append(sim)
                a_score.append(sim * ((ind//3)/10 + 1))

                if top_score is None or top_score < sim:
                    top_score = sim
                    top_slice = anchor_slice
            else:
                anchor_mask[ind] = False

        anchors = anchors[anchor_mask]
        a_score = np.array(a_score)
        bias_score = top_score * ((a_score.argmax()//3)/10 + 1)
        if top_score is not None and a_score.mean() + a_score.std() * FLAGS.anchor_std_multiplier < bias_score:
            top_anchor = anchors[a_score.argmax()]
            target = np.array([max_coor[0], max_coor[1], top_anchor[0], top_anchor[1]])
            target /= np.tile(scale_factor, 2)
            target = to_y1x1y2x2(target)
        else:
            target = to_yxhw(ft_target)
            target[:2] = max_coor / scale_factor
            target = to_y1x1y2x2(target)

            t_slice = [max_coor[0], max_coor[1], max_coor[0] + self.template_size, max_coor[1] + self.template_size]
            top_slice = features[t_slice[0]:t_slice[2], t_slice[1]:t_slice[3], :]

            top_score = np.tensordot(self.template, top_slice, axes=((0, 1, 2), (0, 1, 2)))

        return target, top_score, top_slice

    def train(self, frame, target, sim_map, search_window, scale_factor, max_batch_size=256, in_batch_size=128, epochs=3, learning_rate=1e-5):
        max_points = get_n_max(sim_map, max_batch_size)
        max_points = np.divide(max_points, scale_factor)
        max_points += search_window[:2]
        max_points *= self.stride

        c_target = to_yxhw(target) * self.stride
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

            if np.shape(frame_sample)[0] > self.stride and np.shape(frame_sample)[1] > self.stride:
                frame_sample = cv2.resize(frame_sample,
                                          dsize=(FLAGS.template_size * self.stride, FLAGS.template_size * self.stride),
                                          interpolation=cv2.INTER_CUBIC)
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

        pos_samples = pos_samples[:in_batch_size // 3]
        neg_samples = neg_samples[:in_batch_size - len(pos_samples)]

        samples = np.array(pos_samples + neg_samples)

        labels = np.zeros(len(samples))
        # labels[:len(pos_samples), 0] = 1
        labels[:len(pos_samples)] = 1  # np.array(pos_iou)[:len(pos_samples)]
        # labels[len(pos_samples):, 1] = 1
        labels[len(pos_samples):] = 0  # (np.array(neg_iou)[:len(neg_samples)] - 1)

        loss = entropy_loss(self.template)
        # optimizer = optimizers.Adam(lr=learning_rate)
        optimizer = optimizers.SGD(lr=learning_rate, momentum=0.9, decay=5e-4)

        self.model.compile(optimizer=optimizer, loss=loss)

        for epoch in range(epochs):
            perm = np.random.permutation(len(samples))
            samples = samples[perm]
            labels = labels[perm]
            losses = []

            print("=========== " + str(epoch + 1) + ". EPOCH ===========")
            for batch in range(np.ceil(len(samples) / FLAGS.tuning_batch_size).astype(int)):
                od = batch * FLAGS.tuning_batch_size
                do = od + FLAGS.tuning_batch_size
                x = samples[od:do]
                y = labels[od:do]

                loss = self.model.train_on_batch(x, y)
                losses.append(loss)
                print(str(batch + 1) + ". Batch loss: " + str(loss))
                # model.fit(x=np.array(samples), y=np.array(labels), epochs=epochs, batch_size=FLAGS.tuning_batch_size, shuffle=True)

            print("Mean loss: " + str(np.mean(losses)))
        # print("Pos length:" + str(len(pos_samples)))
        # print("Neg length:" + str(len(neg_samples)))

        return


def init_cnn():
    base_model = VGG16(weights='imagenet', include_top=False)

    for layer in base_model.layers[:11]:
        layer.trainable = False

    vgg_model = models.Model(inputs=base_model.input,
                             outputs=base_model.get_layer('block' + str(FLAGS.output_layer_block) + '_conv3').output)
    drop_layers = []
    model = models.Sequential()

    for i, layer in enumerate(vgg_model.layers):
        if i not in drop_layers:
            model.add(layer)

    # model.summary()

    return model


if __name__ == '__main__':
    main()
