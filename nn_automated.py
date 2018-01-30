# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Simple image classification with Inception.

Run image classification with Inception trained on ImageNet 2012 Challenge data
set.

This program creates a graph from a saved GraphDef protocol buffer,
and runs inference on an input JPEG image. It outputs human readable
strings of the top 5 predictions along with their probabilities.

Change the --image_file argument to any jpg image to compute a
classification of that image.

Please see the tutorial and website for a detailed description of how
to use this script to perform image recognition.

https://tensorflow.org/tutorials/image_recognition/
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import re
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf
import math
import operator
from skimage.segmentation import quickshift
import csv
import datetime

FLAGS = None

# pylint: disable=line-too-long
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
# pylint: enable=line-too-long

class NodeLookup(object):
    """Converts integer node ID's to human readable labels."""

    def __init__(self,
                 label_lookup_path=None,
                 uid_lookup_path=None):
        if not label_lookup_path:
            label_lookup_path = os.path.join(
                FLAGS.model_dir, 'imagenet_2012_challenge_label_map_proto.pbtxt')
        if not uid_lookup_path:
            uid_lookup_path = os.path.join(
                FLAGS.model_dir, 'imagenet_synset_to_human_label_map.txt')
        self.node_lookup = self.load(label_lookup_path, uid_lookup_path)

    def load(self, label_lookup_path, uid_lookup_path):
        """Loads a human readable English name for each softmax node.

    Args:
      label_lookup_path: string UID to integer node ID.
      uid_lookup_path: string UID to human-readable string.

    Returns:
      dict from integer node ID to human-readable string.
    """
        if not tf.gfile.Exists(uid_lookup_path):
            tf.logging.fatal('File does not exist %s', uid_lookup_path)
        if not tf.gfile.Exists(label_lookup_path):
            tf.logging.fatal('File does not exist %s', label_lookup_path)

        # Loads mapping from string UID to human-readable string
        proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path).readlines()
        uid_to_human = {}
        p = re.compile(r'[n\d]*[ \S,]*')
        for line in proto_as_ascii_lines:
            parsed_items = p.findall(line)
            uid = parsed_items[0]
            human_string = parsed_items[2]
            uid_to_human[uid] = human_string

        # Loads mapping from string UID to integer node ID.
        node_id_to_uid = {}
        proto_as_ascii = tf.gfile.GFile(label_lookup_path).readlines()
        for line in proto_as_ascii:
            if line.startswith('  target_class:'):
                target_class = int(line.split(': ')[1])
            if line.startswith('  target_class_string:'):
                target_class_string = line.split(': ')[1]
                node_id_to_uid[target_class] = target_class_string[1:-2]

        # Loads the final mapping of integer node ID to human-readable string
        node_id_to_name = {}
        for key, val in node_id_to_uid.items():
            if val not in uid_to_human:
                tf.logging.fatal('Failed to locate: %s', val)
            name = uid_to_human[val]
            node_id_to_name[key] = name

        return node_id_to_name

    def id_to_string(self, node_id):
        if node_id not in self.node_lookup:
            return ''
        return self.node_lookup[node_id]


def find_segment_pixels(image_value, qs_kernel_size=4):
    segments = quickshift(image_value, kernel_size=qs_kernel_size, max_dist=200, ratio=0.2)
    segment_pixels = {}
    for i in range(len(segments)):
        for j in range(len(segments[i])):
            segment = segments[i][j]
            if segment not in segment_pixels:
                segment_pixels[segment] = []
            segment_pixels[segment].append((i, j))
    return segment_pixels


class ChangeFinder(object):

    def __init__(self, image_values):
        pixels = image_values.shape[0] * image_values.shape[1]
        self.limited_pixels_cnt = math.floor(pixels / 1000)
        self.segment_pixels = find_segment_pixels(image_values)

    def find_values_to_modify_no_constraints(self, gradients_values):
        values_to_modify = []
        for i in range(len(gradients_values)):
            values_to_modify.append([])
            for j in range(len(gradients_values[i])):
                values_to_modify[i].append([])
                for k in range(len(gradients_values[i][j])):
                    values_to_modify[i][j].append(1 * np.sign(gradients_values[i][j][k]))

        return np.array(values_to_modify)

    def find_values_to_modify_limited_pixels(self, gradients_values):
        values_to_modify = []

        gradient_per_pixel = {}
        for i in range(len(gradients_values)):
            for j in range(len(gradients_values[i])):
                agg_gradient = 0
                for k in range(len(gradients_values[i][j])):
                    agg_gradient += math.pow(gradients_values[i][j][k], 2)
                gradient_per_pixel[(i, j)] = agg_gradient

        sorted_gradient_per_pixel = sorted(gradient_per_pixel.items(), key=operator.itemgetter(1))
        sorted_gradient_per_pixel.reverse()
        top_pixels = list(map(lambda x: x[0], sorted_gradient_per_pixel[:self.limited_pixels_cnt]))
        for i in range(len(gradients_values)):
            values_to_modify.append([])
            for j in range(len(gradients_values[i])):
                values_to_modify[i].append([])
                if (i, j) in top_pixels:
                    for k in range(len(gradients_values[i][j])):
                        values_to_modify[i][j].append(1 * np.sign(gradients_values[i][j][k]))
                else:
                    for k in range(len(gradients_values[i][j])):
                        values_to_modify[i][j].append(0)

        return np.array(values_to_modify)

    def find_values_to_modify_limited_superpixels(self, gradients_values):
        values_to_modify = []

        gradient_per_pixel = {}
        for i in range(len(gradients_values)):
            for j in range(len(gradients_values[i])):
                agg_gradient = 0
                for k in range(len(gradients_values[i][j])):
                    agg_gradient += math.pow(gradients_values[i][j][k], 2)
                gradient_per_pixel[(i, j)] = agg_gradient

        gradient_per_segment = {}
        for segment in self.segment_pixels:
            s = 0
            for pixel in self.segment_pixels[segment]:
                s += gradient_per_pixel[pixel]
            s /= len(self.segment_pixels[segment])
            gradient_per_segment[segment] = s

        sorted_gradient_per_segment = sorted(gradient_per_segment.items(), key=operator.itemgetter(1))
        sorted_gradient_per_segment.reverse()
        top_segments = list(map(lambda x: x[0], sorted_gradient_per_segment[:2]))
        top_pixels = set()
        for segment in top_segments:
            for pixel in self.segment_pixels[segment]:
                top_pixels.add(pixel)

        for i in range(len(gradients_values)):
            values_to_modify.append([])
            for j in range(len(gradients_values[i])):
                values_to_modify[i].append([])
                if (i, j) in top_pixels:
                    for k in range(len(gradients_values[i][j])):
                        values_to_modify[i][j].append(1 * np.sign(gradients_values[i][j][k]))
                else:
                    for k in range(len(gradients_values[i][j])):
                        values_to_modify[i][j].append(0)

        return np.array(values_to_modify)


def create_graph():
    """Creates a graph from saved GraphDef file and returns a saver."""
    # Creates graph from saved graph_def.pb.
    with tf.gfile.FastGFile(os.path.join(
            FLAGS.model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


def predict(sess, softmax_tensor, image_data):
    if len(image_data.shape) == 3:
        return predict_single(sess, softmax_tensor, image_data)
    else:
        return [predict_single(sess, softmax_tensor, temp_image_data) for temp_image_data in image_data]


def predict_single(sess, softmax_tensor, image_data):
    return np.squeeze(sess.run(softmax_tensor, {sess.graph.get_operations()[1].name + ':0': image_data}))


def create_predict_func(sess, softmax_tensor):
    return lambda image_data: predict(sess, softmax_tensor, image_data)


def plotable_image(image):
    modified_image = []
    for i in range(len(image)):
        modified_image.append([])
        for j in range(len(image[i])):
            modified_image[i].append([])
            for k in range(len(image[i][j])):
                value = image[i][j][k]
                if value < 0:
                    value = 0
                if value > 255:
                    value = 255
                modified_image[i][j].append(round(value))
    return np.array(modified_image).astype(np.uint8)


def find_changes_sign(sess, image, tag, gradients, softmax_tensor, iterations, find_values_to_modify_func, wanted_confidence, sign):
    images = []
    stats = []

    predictions = sess.run(softmax_tensor, {sess.graph.get_operations()[1].name + ':0': image})
    predictions = np.squeeze(predictions)
    prev_score = predictions[tag]
    orig_score = predictions[tag]

    total_time = 0
    pixels = image.shape[0] * image.shape[1]

    for i in range(iterations):
        if i == 0:
            curr_image = image
        else:
            curr_image = images[i - 1]

        a = datetime.datetime.now()
        gradients_values = sess.run(gradients, {sess.graph.get_operations()[1].name + ':0': curr_image})
        images.append(curr_image + sign * find_values_to_modify_func(gradients_values[0]))
        b = datetime.datetime.now()
        iter_time = (b - a).total_seconds()
        total_time += iter_time

        predictions = sess.run(softmax_tensor, {sess.graph.get_operations()[1].name + ':0': curr_image})
        predictions = np.squeeze(predictions)

        diff_image = images[len(images) - 1] - image

        score = predictions[tag]
        non_zero_pixels = count_non_zero_pixels(diff_image)
        stats.append({
                        'pixels': pixels,
                        'iter': i,
                        'score': score,
                        'prev_score_diff': score - prev_score,
                        'prev_score_diff_pct': (score - prev_score) / prev_score,
                        'orig_score_diff': score - orig_score,
                        'orig_score_diff_pct': (score - orig_score) / orig_score,
                        'L2': calc_l2(diff_image),
                        'changed_pixels': non_zero_pixels,
                        'changed_pixels_pct': non_zero_pixels / pixels,
                        'iter_time': iter_time,
                        'total_time': total_time
                    })
        prev_score = score

        # Creates node ID --> English string lookup.
        node_lookup = NodeLookup()

        top_k = predictions.argsort()[-FLAGS.num_top_predictions:][::-1]
        print(i)
        print("=======")
        for node_id in top_k:
            human_string = node_lookup.id_to_string(node_id)
            score = predictions[node_id]
            print('%s - %s (score = %.5f)' % (str(node_id), human_string, score))
        print()

        if i > 0 and sign * predictions[tag] < sign * wanted_confidence:
            break

    return stats


def count_non_zero_pixels(image):
    cnt = 0
    for i in range(len(image)):
        for j in range(len(image[i])):
            if image[i][j][0] != 0 or image[i][j][1] != 0 or image[i][j][2] != 0:
                cnt += 1
    return cnt


def calc_l2(change_img):
    s = 0
    for i in range(len(change_img)):
        for j in range(len(change_img[i])):
            s += change_img[i][j][0] * change_img[i][j][0]
            s += change_img[i][j][1] * change_img[i][j][1]
            s += change_img[i][j][2] * change_img[i][j][2]
    return round(math.sqrt(s), 4)


def write_stats_to_file(file_name, stats):
    field_names = list(stats[0].keys())
    with open(file_name, 'w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=field_names, delimiter=',', lineterminator='\n')
        writer.writeheader()
        for stat in stats:
            writer.writerow(stat)


def run_algo(sess, image, tag, gradients, softmax_tensor, max_iterations, find_values_to_modify_func, max_confidence):
    print('Positive')
    pos_stats = find_changes_sign(sess, image, tag, gradients, softmax_tensor, max_iterations, find_values_to_modify_func, max_confidence, -1)
    #
    # print('Negative')
    # neg_stats = find_changes_sign(sess, image, tag, gradients, softmax_tensor, max_iterations, find_values_to_modify_func, min_confidence, 1)
    #
    # return pos_stats, neg_stats
    if pos_stats[-1]['score'] >= max_confidence:
        return pos_stats
    else:
        return None


def run_inference_on_image(sess, image, image_index):
    """Runs inference on an image.

  Args:
    image: Image file name.

  Returns:
    Nothing
  """
    if not tf.gfile.Exists(image):
        tf.logging.fatal('File does not exist %s', image)
    image_data = tf.gfile.FastGFile(image, 'rb').read()

    # Some useful tensors:
    # 'softmax:0': A tensor containing the normalized prediction across
    #   1000 labels.
    # 'pool_3:0': A tensor containing the next-to-last layer containing 2048
    #   float description of the image.
    # 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG
    #   encoding of the image.
    # Runs the softmax tensor by feeding the image_data as input to the graph.
    softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
    modified_input_tensor = sess.graph.get_tensor_by_name(sess.graph.get_operations()[1].name + ':0')
    grad_tensor = sess.graph.get_tensor_by_name(sess.graph.get_operations()[2].name + ':0')
    image_values = sess.run(modified_input_tensor, {'DecodeJpeg/contents:0': image_data})

    predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
    predictions = np.squeeze(predictions)

    top_k = predictions.argsort()[-FLAGS.num_top_predictions:][::-1]

    max_iterations = 20
    change_finder = ChangeFinder(image_values)

    output_path = r'C:/Workspace/tensorflow_models/models/data/runtime_3/flowers'
    for tag in top_k[:3]:
        loss = tf.losses.softmax_cross_entropy(tf.one_hot([tag], 1008), softmax_tensor)
        gradients = tf.gradients(loss, grad_tensor)

        confidence = predictions[tag]
        if 0.07 <= confidence <= 0.5:
            super_prev_succeed = True
            for max_confidence in np.arange(0.55, 0.975, 0.025):
                if super_prev_succeed:
                    super_pixel_pos_stats = run_algo(sess, image_values, tag, gradients, softmax_tensor, max_iterations, change_finder.find_values_to_modify_limited_superpixels, max_confidence)
                    if super_pixel_pos_stats is not None:
                        write_stats_to_file(os.path.join(output_path, '_'.join([str(image_index), str(tag), 'super_pixel', str(max_confidence).replace('.', '_'), 'pos.csv'])), super_pixel_pos_stats)
                    else:
                        super_prev_succeed = False



def maybe_download_and_extract():
    """Download and extract model tar file."""
    dest_directory = FLAGS.model_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (
                filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def main(_):
    images_dir = r'C:/Workspace/tensorflow_models/models/data/flowers'
    image_index = 100
    image_index_done = 99

    # Creates graph from saved GraphDef.
    create_graph()

    with tf.Session() as sess:
        for image in os.listdir(images_dir):
            if image.endswith(".jpg"):
                if image_index > image_index_done:
                    image_path = os.path.join(images_dir, image)
                    run_inference_on_image(sess, image_path, image_index)
                image_index += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # classify_image_graph_def.pb:
    #   Binary representation of the GraphDef protocol buffer.
    # imagenet_synset_to_human_label_map.txt:
    #   Map from synset ID to a human readable string.
    # imagenet_2012_challenge_label_map_proto.pbtxt:
    #   Text representation of a protocol buffer mapping a label to synset ID.
    parser.add_argument(
        '--model_dir',
        type=str,
        default='/tmp/imagenet',
        help="""\
      Path to classify_image_graph_def.pb,
      imagenet_synset_to_human_label_map.txt, and
      imagenet_2012_challenge_label_map_proto.pbtxt.\
      """
    )
    parser.add_argument(
        '--image_file',
        type=str,
        default='',
        help='Absolute path to image file.'
    )
    parser.add_argument(
        '--num_top_predictions',
        type=int,
        default=5,
        help='Display this many predictions.'
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
