# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Convert raw PASCAL dataset to TFRecord for object_detection.

Example usage:
    python object_detection/dataset_tools/create_pascal_tf_record.py \
        --data_dir=/home/user/VOCdevkit \
        --output_path=/home/user/pascal.record
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
import logging
import os
import random
import contextlib2

from lxml import etree
import PIL.Image

import tensorflow._api.v2.compat.v1 as tf
from src.models.research.object_detection.utils import dataset_util
from src.models.research.object_detection.utils import label_map_util
from src.models.research.object_detection.dataset_tools import tf_record_creation_util

flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to raw PASCAL VOC dataset.')
flags.DEFINE_string('annotations_dir', 'Annotations',
                    '(Relative) path to annotations directory.')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('label_map_path', 'data/pascal_label_map.pbtxt',
                    'Path to label map proto')
flags.DEFINE_boolean('ignore_difficult_instances', False, 'Whether to ignore '
                                                          'difficult instances')
flags.DEFINE_string('num_shards', '10', 'Number of shards to create.')
flags.DEFINE_string('train_eval_split', '0.8', 'Percentage of dataset used in training')
FLAGS = flags.FLAGS


def dict_to_tf_example(data,
                       folder,
                       dataset_directory,
                       label_map_dict,
                       ignore_difficult_instances=False,
                       image_subdirectory='JPEGImages'):
    """Convert XML derived dict to tf.Example proto.

  Notice that this function normalizes the bounding box coordinates provided
  by the raw data.

  Args:
    data: dict holding PASCAL XML fields for a single image (obtained by
      running dataset_util.recursive_parse_xml_to_dict)
    folder: directory containing the PASCAL subdataset
    dataset_directory: Path to root directory holding PASCAL dataset
    label_map_dict: A map from string label names to integers ids.
    ignore_difficult_instances: Whether to skip difficult instances in the
      dataset  (default: False).
    image_subdirectory: String specifying subdirectory within the
      PASCAL dataset directory holding the actual image data.

  Returns:
    example: The converted tf.Example.

  Raises:
    ValueError: if the image pointed to by data['filename'] is not a valid JPEG
  """
    img_path = os.path.join(folder, image_subdirectory, data['filename'])
    full_path = os.path.join(dataset_directory, img_path)
    with tf.gfile.GFile(full_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    if image.format != 'JPEG':
        # image.save(full_path, 'jpeg')
        raise ValueError('Image format not JPEG')

    width = int(data['size']['width'])
    height = int(data['size']['height'])

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    if 'object' in data:
        for obj in data['object']:
            xmin.append(float(obj['bndbox']['xmin']) / width)
            ymin.append(float(obj['bndbox']['ymin']) / height)
            xmax.append(float(obj['bndbox']['xmax']) / width)
            ymax.append(float(obj['bndbox']['ymax']) / height)
            classes_text.append(obj['name'].encode('utf8'))
            classes.append(label_map_dict[obj['name']])

    example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(data['filename'].encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(data['filename'].encode('utf8')),
        # 'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return example


def split_dataset(examples, train_ratio=0.8):
    random.shuffle(examples)
    split_index = int(len(examples) * train_ratio)

    train_examples = examples[:split_index]
    validation_examples = examples[split_index:]

    return train_examples, validation_examples


def main(_):
    # Insert directory of a dataset
    datasets = FLAGS.data_dir.split(" ")
    label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)

    train_ratio = float(FLAGS.train_eval_split)
    num_shards_stack = [int(int(FLAGS.num_shards) * train_ratio), int(int(FLAGS.num_shards) * (1 - train_ratio))]
    output_namestack = ['pascal_train.record', 'pascal_eval.record']

    for dataset_path in datasets:
        logging.info('Reading from PASCAL %s dataset.', dataset_path)
        examples_path = os.path.join(dataset_path, 'ImageSets', 'Main', 'default.txt')
        annotations_dir = os.path.join(dataset_path, FLAGS.annotations_dir)
        examples_list = dataset_util.read_examples_list(examples_path)
        (train_examples, validation_examples) = split_dataset(examples=examples_list, train_ratio=train_ratio)
        examples_stack = [train_examples, validation_examples]

        for i, examples in enumerate(examples_stack):
            with contextlib2.ExitStack() as tf_record_close_stack:
                output_filebase = os.path.join(FLAGS.output_path, output_namestack[i])
                writer = tf.python_io.TFRecordWriter(output_filebase)
                output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(
                    tf_record_close_stack, output_filebase, num_shards_stack[i])
                for index, example in enumerate(examples):
                    if index % 100 == 0:
                        logging.info('On image %d of %d', index, len(examples))
                    path = os.path.join(annotations_dir, example + '.xml')
                    with tf.gfile.GFile(path, 'r') as fid:
                        xml_str = fid.read()
                    xml = etree.fromstring(xml_str)
                    data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']
                    tf_example = dict_to_tf_example(data, dataset_path, FLAGS.data_dir, label_map_dict,
                                                    FLAGS.ignore_difficult_instances)
                    output_shard_index = index % num_shards_stack[i]
                    output_tfrecords[output_shard_index].write(tf_example.SerializeToString())
    writer.close()


if __name__ == '__main__':
    tf.app.run()
