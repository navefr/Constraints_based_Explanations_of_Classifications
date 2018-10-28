import tensorflow as tf
import argparse
import os
from six.moves import urllib
import sys
import tarfile
import numpy as np
import re
import matplotlib.pyplot as plt
from skimage.segmentation import quickshift
from skimage.segmentation import mark_boundaries
from lime.lime_image import ImageExplanation
import math
import operator

def maybe_download_and_extract(FLAGS, DATA_URL):
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
    
def create_graph(FLAGS):
    """Creates a graph from saved GraphDef file and returns a saver."""
    # Creates graph from saved graph_def.pb.
    with tf.gfile.FastGFile(os.path.join(
            FLAGS.model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')
        

class NodeLookup(object):
    """Converts integer node ID's to human readable labels."""

    def __init__(self,
                 FLAGS,
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
        
        
class Model:
    def __init__(self):
        FLAGS = None
        DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
        
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
        
        maybe_download_and_extract(FLAGS, DATA_URL)
        create_graph(FLAGS)
        self.node_lookup = NodeLookup(FLAGS)
        
        self.sess = tf.Session()
        self.modified_input_tensor = self.sess.graph.get_tensor_by_name('Cast:0')
        self.grad_tensor = self.modified_input_tensor
        self.softmax_tensor = self.sess.graph.get_tensor_by_name('softmax:0')
        
    def __del__(self):
        self.sess.close()
        
    def decode_image(self, image_file):
        image_data = tf.gfile.FastGFile(image_file, 'rb').read()
        return self.sess.run(self.modified_input_tensor, {'DecodeJpeg/contents:0': image_data})
    
    def predict_file(self, image_file):
        image_data = tf.gfile.FastGFile(image_file, 'rb').read()
        predictions = self.sess.run(self.softmax_tensor, {'DecodeJpeg/contents:0': image_data})
        return np.squeeze(predictions)
    
    def predict(self, vector):
        predictions = self.sess.run(self.softmax_tensor, {'Cast:0': vector})
        return np.squeeze(predictions)
        
    def get_perturbation(self, image, tag, is_positive=True):
        loss = tf.losses.softmax_cross_entropy(tf.one_hot([tag], 1008), self.softmax_tensor)
        gradients = tf.gradients(loss, self.grad_tensor)

        gradients_values = self.sess.run(gradients, {'Cast:0': image})[0]
        if is_positive:
            return -1 * gradients_values
        else:
            return gradients_values
        
class DisplayHelper:
    
    def __init__(self, model, image_file):
        self.model = model
        self.image_file = image_file
        self.original_image = self.model.decode_image(self.image_file)
        
        self.segments = quickshift(self.original_image / 256, kernel_size=4, max_dist=200, ratio=0.2)
        self.segment_size = {}
        for i in range(len(self.segments)):
            for j in range(len(self.segments[i])):
                segment = self.segments[i][j]
                if segment not in self.segment_size:
                    self.segment_size[segment] = 0
                self.segment_size[segment] += 1
        
    def plot_image(self, image=None):
        if image is None:
            image = self.original_image
            
        predictions = self.model.predict(image)
        predicted_label = predictions.argsort()[-1]
        score = predictions[predicted_label]
        
        image = np.clip(image, 0, 256)
        image /= 256
        
        plt.imshow(image)
        plt.axis('off')
        
        plt.title(self.model.node_lookup.id_to_string(predicted_label) + ', %s' % (str(round(100 * score, 2))) + '%')

    def plot_diff(self, image):
        diff = image - self.original_image
        diff = np.clip(diff, 0, 256)
        
        plt.imshow(diff)
        plt.axis('off')
        plt.title('Changes')
        
    def mark_explanation(self, temp, mask, alpha=0.3):
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
        
    def plot_explaination_highlight(self, image, image_neg=None):
        ret_exp = ImageExplanation(self.original_image / 256, self.segments)

        segment_pixels = {}
        positive_pixels = set(zip(*np.where(self.original_image != image)[:2]))
        for pixel in positive_pixels:
            segment = self.segments[pixel[0], pixel[1]]
            if segment not in segment_pixels:
                segment_pixels[segment] = (0, 0)
            pos, neg = segment_pixels[segment]
            segment_pixels[segment] = (pos + 1, neg)

        negative_pixels = None
        if image_neg is not None:
            negative_pixels = set(zip(*np.where(self.original_image != image_neg)[:2]))
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
        ret_exp.local_exp['label'] = exp

        temp, mask = ret_exp.get_image_and_mask('label', positive_only=True, num_features=10, hide_rest=False)
        
        plt.imshow(self.mark_explanation(temp, mask))
        plt.axis('off')
        plt.title('Explanation')
        
def find_segment_pixels(image, qs_kernel_size=4):
    segments = quickshift(image, kernel_size=qs_kernel_size, max_dist=200, ratio=0.2)
    segment_pixels = {}
    for i in range(len(segments)):
        for j in range(len(segments[i])):
            segment = segments[i][j]
            if segment not in segment_pixels:
                segment_pixels[segment] = []
            segment_pixels[segment].append((i, j))
    return segment_pixels


class Projector:
    def __init__(self, image):
        self.pixels = image.shape[0] * image.shape[1]
        self.segment_pixels = find_segment_pixels(image / 256)

    def no_constraints(self, perturbation):
        return np.sign(perturbation)

    def limited_pixels(self, perturbation, percentage):
        limited_pixels_cnt = int(self.pixels * percentage)
        pixels_to_modify = []

        perturb_per_pixel = {}
        for i in range(len(perturbation)):
            for j in range(len(perturbation[i])):
                agg_pixel_perturbation = 0
                for k in range(len(perturbation[i][j])):
                    agg_pixel_perturbation += math.pow(perturbation[i][j][k], 2)
                perturb_per_pixel[(i, j)] = agg_pixel_perturbation

        sorted_perturb_per_pixel = sorted(perturb_per_pixel.items(), key=operator.itemgetter(1))
        sorted_perturb_per_pixel.reverse()
        top_pixels = list(map(lambda x: x[0], sorted_perturb_per_pixel[:limited_pixels_cnt]))
        for i in range(len(perturbation)):
            pixels_to_modify.append([])
            for j in range(len(perturbation[i])):
                pixels_to_modify[i].append([])
                if (i, j) in top_pixels:
                    for k in range(len(perturbation[i][j])):
                        pixels_to_modify[i][j].append(1 * np.sign(perturbation[i][j][k]))
                else:
                    for k in range(len(perturbation[i][j])):
                        pixels_to_modify[i][j].append(0)

        return np.array(pixels_to_modify)

    def limited_superpixels(self, perturbation, superpixels_cnt):
        pixels_to_modify = []

        perturb_per_pixel = {}
        for i in range(len(perturbation)):
            for j in range(len(perturbation[i])):
                agg_pixel_perturbation = 0
                for k in range(len(perturbation[i][j])):
                    agg_pixel_perturbation += math.pow(perturbation[i][j][k], 2)
                perturb_per_pixel[(i, j)] = agg_pixel_perturbation

        perturb_per_segment = {}
        for segment in self.segment_pixels:
            s = 0
            for pixel in self.segment_pixels[segment]:
                s += perturb_per_pixel[pixel]
            s /= len(self.segment_pixels[segment])
            perturb_per_segment[segment] = s

        sorted_perturb_per_segment = sorted(perturb_per_segment.items(), key=operator.itemgetter(1))
        sorted_perturb_per_segment.reverse()
        top_segments = list(map(lambda x: x[0], sorted_perturb_per_segment[:superpixels_cnt]))
        top_pixels = set()
        for segment in top_segments:
            for pixel in self.segment_pixels[segment]:
                top_pixels.add(pixel)

        for i in range(len(perturbation)):
            pixels_to_modify.append([])
            for j in range(len(perturbation[i])):
                pixels_to_modify[i].append([])
                if (i, j) in top_pixels:
                    for k in range(len(perturbation[i][j])):
                        pixels_to_modify[i][j].append(1 * np.sign(perturbation[i][j][k]))
                else:
                    for k in range(len(perturbation[i][j])):
                        pixels_to_modify[i][j].append(0)

        return np.array(pixels_to_modify)