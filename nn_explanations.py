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
import matplotlib.pyplot as plt
import math
import operator
from lime.lime_image import ImageExplanation
from skimage.segmentation import mark_boundaries
from skimage.segmentation import quickshift

try:
    import lime
except:
    sys.path.append(os.path.join('..', '..')) # add the current directory
    import lime
from lime import lime_image

FLAGS = None

# pylint: disable=line-too-long
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
# pylint: enable=line-too-long


class ExplainGenerator(object):

    def __init__(self, image, hide_color=None, qs_kernel_size=4):
        from skimage.segmentation import quickshift
        segments = quickshift(image, kernel_size=qs_kernel_size, max_dist=200, ratio=0.2)

        self.image = image
        self.segments = segments

        segment_size = {}
        for i in range(len(segments)):
            for j in range(len(segments[i])):
                segment = segments[i][j]
                if segment not in segment_size:
                    segment_size[segment] = 0
                segment_size[segment] += 1
        self.segment_size = segment_size

    def explain_instance(self, label, positive_pixels, negative_pixels=None):

        if not negative_pixels:
            negative_pixels = []
        ret_exp = ImageExplanation(self.image, self.segments)

        segment_pixels = {}
        for pixel in positive_pixels:
            segment = self.segments[pixel[0], pixel[1]]
            if segment not in segment_pixels:
                segment_pixels[segment] = (0, 0)
            pos, neg = segment_pixels[segment]
            segment_pixels[segment] = (pos + 1, neg)

        if negative_pixels is not None:
            for pixel in negative_pixels:
                segment = self.segments[pixel[0], pixel[1]]
                if segment not in segment_pixels:
                    segment_pixels[segment] = (0, 0)
                pos, neg = segment_pixels[segment]
                segment_pixels[segment] = (pos, neg + 1)

        exp = []

        if negative_pixels is None:
            for segment in segment_pixels:
                segment_size = self.segment_size[segment]
                pos, neg = segment_pixels[segment]
                pos_proportion = float(pos) / segment_size
                exp.append((segment, pos_proportion))
        else:
            for segment in segment_pixels:
                segment_size = self.segment_size[segment]
                pos, neg = segment_pixels[segment]
                total_proportion = float(pos + neg) / segment_size

                pos_dist = (float(pos) / len(positive_pixels)) if len(positive_pixels) != 0 else 0
                neg_dist = (float(neg) / len(negative_pixels)) if len(negative_pixels) != 0 else 0
                is_pos = 1 if pos_dist >= neg_dist else -1

                exp.append((segment, total_proportion * is_pos))

        exp = sorted(exp, key=lambda x: np.abs(x[1]), reverse=True)
        ret_exp.local_exp[label] = exp

        return ret_exp


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
    scores = []
    for i in range(iterations):
        if i == 0:
            curr_image = image
        else:
            curr_image = images[i - 1]

        gradients_values = sess.run(gradients, {sess.graph.get_operations()[1].name + ':0': curr_image})
        images.append(curr_image + sign * find_values_to_modify_func(gradients_values[0]))

        predictions = sess.run(softmax_tensor, {sess.graph.get_operations()[1].name + ':0': curr_image})
        predictions = np.squeeze(predictions)

        score = predictions[tag]
        scores.append(score)

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

    return images[len(images) - 1], len(images)


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


def mark_explanation(temp, mask, alpha=0.3):
    with_boundaries = mark_boundaries(temp, mask, outline_color=(1, 1, 1), mode='thick')
    in_boundaries = []
    for i in range(len(mask)):
        in_boundaries.append([])
        for j in range(len(mask[i])):
            pixel = [1.0, 1.0, 1.0]
            if mask[i][j] == 1:
                pixel[0] = with_boundaries[i][j][0]
                pixel[1] = with_boundaries[i][j][1]
                pixel[2] = with_boundaries[i][j][2]
            in_boundaries[i].append(pixel)
    in_boundaries = np.array(in_boundaries)

    return alpha * with_boundaries + (1 - alpha) * in_boundaries



def run_algo(sess, image, tag, explainer, gradients, softmax_tensor, max_iterations, find_values_to_modify_func, max_confidence, min_confidence, node_lookup, output_dir):
    pixels_cnt = image.shape[0] * image.shape[1]

    orig_predictions = sess.run(softmax_tensor, {sess.graph.get_operations()[1].name + ':0': image})
    orig_predictions = np.squeeze(orig_predictions)

    print('Positive')
    pos_image, pos_iterations = find_changes_sign(sess, image, tag, gradients, softmax_tensor, max_iterations, find_values_to_modify_func, max_confidence, -1)
    pos_image = plotable_image(pos_image)
    pos_predictions = sess.run(softmax_tensor, {sess.graph.get_operations()[1].name + ':0': pos_image})
    pos_predictions = np.squeeze(pos_predictions)
    pos_change_img = image - pos_image
    pos_change_pixel_cnt = count_non_zero_pixels(pos_change_img)

    print('Negative')
    neg_image, neg_iterations = find_changes_sign(sess, image, tag, gradients, softmax_tensor, max_iterations, find_values_to_modify_func, min_confidence, 1)
    neg_image = plotable_image(neg_image)
    neg_predictions = sess.run(softmax_tensor, {sess.graph.get_operations()[1].name + ':0': neg_image})
    neg_predictions = np.squeeze(neg_predictions)
    neg_change_img = image - neg_image
    neg_change_pixel_cnt = count_non_zero_pixels(neg_change_img)

    pos_pixels = []
    neg_pixels = []
    for i in range(len(pos_change_img)):
        for j in range(len(pos_change_img[i])):
            if pos_change_img[i][j][0] != 0 or pos_change_img[i][j][1] != 0 or pos_change_img[i][j][2] != 0:
                pos_pixels.append((i, j))
            if neg_change_img[i][j][0] != 0 or neg_change_img[i][j][1] != 0 or neg_change_img[i][j][2] != 0:
                neg_pixels.append((i, j))

    pos_explanation = explainer.explain_instance(tag, pos_pixels)
    pos_temp, pos_mask = pos_explanation.get_image_and_mask(tag, positive_only=True, num_features=10, hide_rest=False)
    neg_explanation = explainer.explain_instance(tag, neg_pixels)
    neg_temp, neg_mask = neg_explanation.get_image_and_mask(tag, positive_only=True, num_features=10, hide_rest=False)

    all_pixels = []
    all_pixels.extend(pos_pixels)
    all_pixels.extend(neg_pixels)
    all_explanation = explainer.explain_instance(tag, all_pixels)
    all_temp, all_mask = all_explanation.get_image_and_mask(tag, positive_only=True, num_features=10, hide_rest=False)

    plt.imshow(image)
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'original_image.jpg'), bbox_inches='tight', pad_inches=-0.1)
    plt.clf()

    plt.imshow(mark_explanation(all_temp, all_mask))
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'positive_and_negative_explanation.jpg'), bbox_inches="tight", pad_inches=-0.1)
    plt.clf()

    plt.imshow(pos_image)
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'positive_image.jpg'), bbox_inches="tight", pad_inches=-0.1)
    plt.clf()

    plt.imshow(pos_change_img)
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'positive_changes.jpg'), bbox_inches="tight", pad_inches=-0.1)
    plt.clf()

    plt.imshow(mark_explanation(pos_temp, pos_mask))
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'positive_explanation.jpg'), bbox_inches="tight", pad_inches=-0.1)
    plt.clf()

    plt.imshow(neg_image)
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'negative_image.jpg'), bbox_inches="tight", pad_inches=-0.1)
    plt.clf()

    plt.imshow(neg_change_img)
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'negative_changes.jpg'), bbox_inches="tight", pad_inches=-0.1)
    plt.clf()

    plt.imshow(mark_explanation(neg_temp, neg_mask))
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'negative_explanation.jpg'), bbox_inches="tight", pad_inches=-0.1)
    plt.clf()

    tag_string = node_lookup.id_to_string(tag)

    with open(os.path.join(output_dir, 'info.txt'), mode='w') as f:
        f.write('Original:\n')
        f.write('\tTag: (' + str(tag) + ') - ' + tag_string + '\n')
        f.write('\tConfidence: ' + str(round(100 * orig_predictions[tag], 2)) + '%' + '\n\n')
        f.write('Positive:\n')
        f.write('\tConfidence: ' + str(round(100 * pos_predictions[tag], 2)) + '%' + '\n')
        f.write('\tIterations: ' + str(pos_iterations) + '\n')
        f.write('\tPixels changed: ' + str(pos_change_pixel_cnt) + '\n')
        f.write('\tPixels changed %: ' + str(round(100 * pos_change_pixel_cnt / pixels_cnt, 2)) + '%\n')
        f.write('\tL2-Norm: ' + str(calc_l2(pos_change_img)) + '\n\n')
        f.write('Negative:\n')
        f.write('\tConfidence: ' + str(round(100 * neg_predictions[tag], 2)) + '%' + '\n')
        f.write('\tIterations: ' + str(neg_iterations) + '\n')
        f.write('\tPixels changed: ' + str(neg_change_pixel_cnt) + '\n')
        f.write('\tPixels changed %: ' + str(round(100 * neg_change_pixel_cnt / pixels_cnt, 2)) + '%\n')
        f.write('\tL2-Norm: ' + str(calc_l2(neg_change_img)))


def run_inference_on_image(sess, image, output):
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

    node_lookup = NodeLookup()

    max_iterations = 30
    change_finder = ChangeFinder(image_values)

    explainer = ExplainGenerator(image_values)

    lime_explainer = lime_image.LimeImageExplainer()

    for tag in top_k[:1]:
        loss = tf.losses.softmax_cross_entropy(tf.one_hot([tag], 1008), softmax_tensor)
        gradients = tf.gradients(loss, grad_tensor)

        tag_string = node_lookup.id_to_string(tag).replace(' ', '_')

        confidence = predictions[tag]
        max_confidence = confidence + 5 * (1 - confidence) / 6
        min_confidence = confidence - (5 * confidence / 6)

        output_dir_no_constraints = os.path.join(output, tag_string + '_no_constraints')
        os.makedirs(output_dir_no_constraints)
        run_algo(sess, image_values, tag, explainer, gradients, softmax_tensor, max_iterations, change_finder.find_values_to_modify_no_constraints, max_confidence, min_confidence, node_lookup, output_dir_no_constraints)

        output_dir_limited_pixels = os.path.join(output, tag_string + '_limited_pixels')
        os.makedirs(output_dir_limited_pixels)
        run_algo(sess, image_values, tag, explainer, gradients, softmax_tensor, max_iterations, change_finder.find_values_to_modify_limited_pixels, max_confidence, min_confidence, node_lookup, output_dir_limited_pixels)

        output_dir_limited_superpixels = os.path.join(output, tag_string + '_limited_superpixels')
        os.makedirs(output_dir_limited_superpixels)
        run_algo(sess, image_values, tag, explainer, gradients, softmax_tensor, max_iterations, change_finder.find_values_to_modify_limited_superpixels, max_confidence, min_confidence, node_lookup, output_dir_limited_superpixels)

        output_dir_lime = os.path.join(output, tag_string + '_lime')
        os.makedirs(output_dir_lime)
        predict_func = create_predict_func(sess, softmax_tensor)
        lime_explanation = lime_explainer.explain_instance(image_values, predict_func, top_labels=5, hide_color=0, num_samples=1000)
        lime_temp, lime_mask = lime_explanation.get_image_and_mask(tag, positive_only=True, num_features=10, hide_rest=False)
        plt.imshow(mark_explanation(lime_temp, lime_mask))
        plt.axis('off')
        plt.savefig(os.path.join(output_dir_lime, 'positive_explanation.jpg'), bbox_inches="tight", pad_inches=-0.1)
        plt.clf()


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
    images_dir = r'C:/Workspace/tensorflow_models/models/data/all_input'
    output_dir = r'C:/Workspace/tensorflow_models/models/data/pics_4_paper'
    image_file_names = ['baby_lion.jpg',
                        'bee_orange_flower.jpg',
                        'bird_fly.jpg',
                        'bird_on_spiky_tree.jpg',
                        'sea_lion_in_eat_fish.jpg',
                        'sea_lions_in_beach.jpg',
                        'tiger_with_baby.jpg',
                        'two_sea_lions_in_beach.jpg',
                        'racing_car_parks.jpg',
                        'monkey_on_car.jpg',
                        'monkey_with_sea.jpg',
                        'monkey_climb_on_tree.jpg',
                        'fish_with_green_background.jpg',
                        'dog_and_boat.jpg',
                        'bullet_train_station.jpg',
                        'brown_bear_in_gress.jpg',
                        'brown_bear_in_field.jpg',
                        'bobcat_on_tree.jpg',
                        'monkey.jpg',
                        'meerkat.jpg',
                        'crab_in_rocks.jpg'
                        ]

    # Creates graph from saved GraphDef.
    create_graph()

    with tf.Session() as sess:
        for image in os.listdir(images_dir):
            if image in image_file_names:
                image_path = os.path.join(images_dir, image)
                image_output_dir = os.path.join(output_dir, image.split('.')[0])
                if not os.path.exists(image_output_dir):
                    os.makedirs(image_output_dir)
                    run_inference_on_image(sess, image_path, image_output_dir)


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
