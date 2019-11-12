from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os

def init_params():

    FLAGS = tf.app.flags.FLAGS
    _checkpoint_path = os.path.join(FLAGS.script_root_dir, "pre_train", "inception_v4.ckpt")

    #####################
    # Basic Flags #
    #####################
    FLAGS.checkpoint_path = _checkpoint_path
    FLAGS.max_number_of_steps = 3400 # 4000: 25 epochs
    FLAGS.save_interval_secs = 85  #68
    FLAGS.max_to_keep = 10
    FLAGS.train_image_size = 299
    FLAGS.model_name = "inception_v4"
    FLAGS.preprocessing_name = None
    FLAGS.checkpoint_exclude_scopes = 'InceptionV4/Logits,InceptionV4/AuxLogits'
    FLAGS.trainable_scopes = 'InceptionV4/Logits,InceptionV4/AuxLogits'
    FLAGS.optimizer = "rmsprop"
    FLAGS.opt_epsilon = 1.0
    FLAGS.learning_rate = 0.01
    FLAGS.end_learning_rate = 0.00001
    FLAGS.learning_rate_decay_type = "fixed"
    FLAGS.learning_rate_decay_factor = 0.94
    FLAGS.num_steps_per_decay = 336.8
