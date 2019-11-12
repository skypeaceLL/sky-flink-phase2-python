from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings
warnings.filterwarnings(action="ignore")

import tensorflow as tf
import os
from time import time
import sys
reload(sys)
sys.setdefaultencoding('utf8')

_script_root_path = os.path.dirname(os.path.abspath(__file__))
_dataset_dir = os.path.join(_script_root_path, "data")
_train_dir = os.path.join(_script_root_path, "train_dir")
_image_data_dir = os.environ["IMAGE_TRAIN_INPUT_PATH"]
_inference_dir = os.environ["MODEL_INFERENCE_PATH"]
tf.app.flags.DEFINE_string('script_root_dir', _script_root_path, 'Directory where checkpoints and event logs are written to.')
tf.app.flags.DEFINE_string('image_data_dir', _image_data_dir, 'Directory where image files are stored.')
tf.app.flags.DEFINE_string('train_dir', _train_dir, 'Directory where checkpoints and event logs are written to.')
tf.app.flags.DEFINE_string('dataset_dir', _dataset_dir, 'The directory where the dataset files are stored.')
tf.app.flags.DEFINE_string('inference_dir', _inference_dir, 'The directory where the model files are stored.')
tf.app.flags.DEFINE_integer('num_train', 5389, '_NUM_TRAIN')
tf.app.flags.DEFINE_integer('num_validation', 600, '_NUM_VALIDATION')

import convert_garbage_data
import model1
import model2
import model3

def main(_):
    FLAGS = tf.app.flags.FLAGS
    print("script root path: %s" % _script_root_path)
    print('IMAGE_TRAIN_INPUT_PATH: %s' % FLAGS.image_data_dir)
    print('MODEL_INFERENCE_PATH: %s' % FLAGS.inference_dir)
    print("dataset_dir: %s " % FLAGS.dataset_dir)
    print("train_dir: %s" % FLAGS.train_dir)
    print("num_train: %d" % FLAGS.num_train)
    print("num_validation: %d" % FLAGS.num_validation)

    print("Begin convert data ...")
    t1 = time()
    num_classes = convert_garbage_data.run(FLAGS.image_data_dir, FLAGS.dataset_dir, FLAGS.inference_dir, 0)
    print("num_classes: %s" % num_classes)
    t2 = time()
    print("End convert data %d s" % (t2-t1))

    print("Begin  model1.execute() ...")
    model1.execute()

    print("Begin  model2.execute() ...")
    #num_classes = convert_garbage_data.run(FLAGS.image_data_dir, FLAGS.dataset_dir, FLAGS.inference_dir, 1)
    model2.execute()

    print("Begin  model3.execute() ...")
    #num_classes = convert_garbage_data.run(FLAGS.image_data_dir, FLAGS.dataset_dir, FLAGS.inference_dir, 2)
    model3.execute()

if __name__ == '__main__':
  tf.compat.v1.app.run()
