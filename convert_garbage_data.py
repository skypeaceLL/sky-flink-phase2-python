
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import tensorflow as tf

from datasets import dataset_utils

FLAGS = tf.app.flags.FLAGS

# The number of images in the validation set.
_NUM_VALIDATION = FLAGS.num_validation

# Seed for repeatability.
#_RANDOM_SEED = 0

# The number of shards per dataset split.
_NUM_SHARDS = 5


class ImageReader(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.compat.v1.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

    self._decode_png_data = tf.compat.v1.placeholder(dtype=tf.string)
    self._decode_png = tf.image.decode_png(self._decode_png_data, channels=3)


  def read_jpeg_dims(self, sess, image_data):
    image = self.decode_jpeg(sess, image_data)
    return image.shape[0], image.shape[1]

  def decode_jpeg(self, sess, image_data):
    image = sess.run(self._decode_jpeg,
                     feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image

  def read_png_dims(self, sess, image_data):
    image = self.decode_png(sess, image_data)
    return image.shape[0], image.shape[1]

  def decode_png(self, sess, image_data):
    image = sess.run(self._decode_png,
                     feed_dict={self._decode_png_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image

def _get_filenames_and_classes(images_dir):
  #Returns a list of filenames and inferred class names.
  images_root = images_dir
  directories = []
  class_names = []
  for filename in os.listdir(images_root):
    path = os.path.join(images_root, filename)
    if os.path.isdir(path):
      directories.append(path)
      class_names.append(filename)

  photo_filenames = []

  total = 0
  for directory in directories:
    i = 0
    for filename in os.listdir(directory):
      path = os.path.join(directory, filename)
      if(path.endswith("jpeg")|path.endswith("jpg")|path.endswith("png")):
        photo_filenames.append(path)
      else:
        continue
      i = i + 1
      total = total + 1
      #if(i>=65):
      #  break
    #print(directory[directory.rindex("/")+1:], i)

  print("total: %d, %d" % (total, _NUM_SHARDS))

  return photo_filenames, sorted(class_names)


def _get_dataset_filename(dataset_dir, split_name, shard_id):
  output_filename = 'garbage_%s_%05d-of-%05d.tfrecord' % (
      split_name, shard_id, _NUM_SHARDS)
  return os.path.join(dataset_dir, output_filename)


def _convert_dataset(split_name, filenames, class_names_to_ids, dataset_dir):
  """Converts the given filenames to a TFRecord dataset.

  Args:
    split_name: The name of the dataset, either 'train' or 'validation'.
    filenames: A list of absolute paths to png or jpg images.
    class_names_to_ids: A dictionary from class names (strings) to ids
      (integers).
    dataset_dir: The directory where the converted datasets are stored.
  """
  assert split_name in ['train', 'validation']

  num_per_shard = int(math.ceil(len(filenames) / float(_NUM_SHARDS)))

  with tf.Graph().as_default():
    image_reader = ImageReader()

    with tf.compat.v1.Session('') as sess:

      for shard_id in range(_NUM_SHARDS):
        output_filename = _get_dataset_filename(
            dataset_dir, split_name, shard_id)

        with tf.io.TFRecordWriter(output_filename) as tfrecord_writer:
          start_ndx = shard_id * num_per_shard
          end_ndx = min((shard_id+1) * num_per_shard, len(filenames))
          print("start_ndx: %d"%start_ndx)
          print("end_ndx: %d"%end_ndx)
          for i in range(start_ndx, end_ndx):
            #sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
            #    i+1, len(filenames), shard_id))
            #sys.stdout.flush()

            if i % 100 == 0:
              print("Convert dataset %d, shard %d" % (i, shard_id))

            # Read the filename:
            image_data = tf.io.gfile.GFile(filenames[i], 'rb').read()

            if filenames[i].endswith(".jpg") | filenames[i].endswith(".jpeg"):
              height, width = image_reader.read_jpeg_dims(sess, image_data)
            elif filenames[i].endswith(".png"):
              height, width = image_reader.read_png_dims(sess, image_data)

            class_name = os.path.basename(os.path.dirname(filenames[i]))
            class_id = class_names_to_ids[class_name]

            example = dataset_utils.image_to_tfexample(
                image_data, b'jpg', height, width, class_id)
            tfrecord_writer.write(example.SerializeToString())

  sys.stdout.write('\n')
  sys.stdout.flush()

def _dataset_exists(dataset_dir):
  for split_name in ['train', 'validation']:
    for shard_id in range(_NUM_SHARDS):
      output_filename = _get_dataset_filename(
          dataset_dir, split_name, shard_id)
      if not tf.gfile.Exists(output_filename):
        return False
  return True


def run(images_dir, dataset_dir, inference_dir, model_random_seed):
  """
  Args:
    images_dir: The images directory where the jpeg is read.
    dataset_dir: The dataset directory where the dataset is stored.
  """
  if (dataset_dir.endswith("data")) != True:
    print("Wrong dataset_dir name! dataset_dir must end with data.")
    exit(1)

  if _dataset_exists(dataset_dir):
    print('Dataset files already exist. Delete them firstly.')
    tf.gfile.DeleteRecursively(dataset_dir)
    #return

  if not tf.gfile.Exists(dataset_dir):
    tf.gfile.MakeDirs(dataset_dir)

  photo_filenames, class_names = _get_filenames_and_classes(images_dir)

  class_names_to_ids = dict(zip(class_names, range(len(class_names))))
  ii = 0
  for fname in photo_filenames :
    if(fname.endswith("png")):
      ii = ii + 1
  print("Found png files: %d" % ii)

  #for class_name in class_names:
  #  print(class_name)

  num_classes = len(class_names_to_ids)

  # Divide into train and test:
  random.seed(model_random_seed)
  random.shuffle(photo_filenames)
  training_filenames = photo_filenames[_NUM_VALIDATION:]
  validation_filenames = photo_filenames[:_NUM_VALIDATION]

  # First, convert the training and validation sets.
  _convert_dataset('train', training_filenames, class_names_to_ids,
                   dataset_dir)
  _convert_dataset('validation', validation_filenames, class_names_to_ids,
                   dataset_dir)

  # Finally, write the labels file:
  labels_to_class_names = dict(zip(range(len(class_names)), class_names))
  dataset_utils.write_label_file(labels_to_class_names, dataset_dir)
  dataset_utils.write_label_file(labels_to_class_names, inference_dir)

  print('\nFinished converting the garbage dataset!')
  return num_classes
