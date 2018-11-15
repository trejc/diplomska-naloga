from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.applications.vgg16 import VGG16
from keras import models
from tracker import track_object
from keras import backend as K

import cv2
import glob
import os
import re
import datetime
import numpy as np
import tensorflow as tf

tf.flags.DEFINE_float("robustness_thresh", 0.5, "IoU over gt box that is considered bad detection")

FLAGS = tf.flags.FLAGS


def main(_):
    vot_directory = os.fsencode("videos/vot2017/")
    test_path = r'testing\test_' + str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.mkdir(test_path)

    FLAGS.append_flags_into_file(test_path + r'\testflags.txt')

    for i, video_dir in enumerate(os.listdir(vot_directory)):
    # for i, video_dir in enumerate([os.fsencode(b'bolt1')]):
        cfg = K.tf.ConfigProto()
        cfg.gpu_options.allow_growth = True
        K.set_session(K.tf.Session(config=cfg))
        model = init_cnn()

        video_path = os.fsdecode(vot_directory + video_dir)
        vid = [cv2.imread(file) for file in glob.glob(video_path + "/*.jpg")]
        gt_file = open(video_path + "/groundtruth.txt", "r")

        try:
            iou_list = track_object(vid, model, gt_file, save_out=None)
            print(str(i + 1) + "/" + str(len(os.listdir(vot_directory))) + ": Tracking completed (" + str(
                os.fsdecode(video_dir)) + ")")
        except Exception as e:
            iou_list = e
            print(str(i + 1) + "/" + str(len(os.listdir(vot_directory))) + ": Tracking completed with error (" + str(
                os.fsdecode(video_dir)) + ")")

        K.clear_session()
        del model

        with open(test_path + r'\iou_lists.txt', mode='a') as file:
            line = os.fsdecode(video_dir) + ": " + str(iou_list) + "\n"
            file.write(line)

    analysis(test_path)


def analysis(directory):
    mAp_list = []
    rob_list = []
    mAp_file = open(directory + r'\mAp.txt', 'a')
    rob_file = open(directory + r'\robustness.txt', 'a')

    with open(directory + r'\iou_lists.txt', 'r') as readFile:
        for line in readFile:
            video_name = line.split(":")[0]
            list_string = re.search("\[.*]", line)

            if list_string is not None:
                iou_string = list_string.group(0)
                iou_list = np.array(iou_string[1:-1].split(","))[10:].astype(float)

                mAp = iou_list.mean()
                robustness = len(iou_list[iou_list > FLAGS.robustness_thresh]) / len(iou_list)

                mAp_list.append(mAp)
                rob_list.append(robustness)

                mAp_file.write(video_name + ": " + str(mAp) + "\n")
                rob_file.write(video_name + ": " + str(robustness) + "\n")
            else:
                mAp_file.write(video_name + ": err\n")
                rob_file.write(video_name + ": err\n")

        mAp_file.write("Total mean mAp: " + str(np.mean(mAp_list)))
        rob_file.write("Total mean robustness: " + str(np.mean(rob_list)))

    mAp_file.close()
    rob_file.close()


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


if __name__ == "__main__":
    tf.app.run()
