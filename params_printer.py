from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

def output():
    FLAGS = tf.app.flags.FLAGS
    print("model_name: %s" % FLAGS.model_name)
    print("checkpoint_path: %s" % FLAGS.checkpoint_path)
    print("==================================================")
    print("max_number_of_steps: %d" % FLAGS.max_number_of_steps)
    print("save_interval_secs: %d" % FLAGS.save_interval_secs)
    print("max_to_keep: %d" % FLAGS.max_to_keep)
    print("batch_size: %d" % FLAGS.batch_size)
    print("optimizer: %s" % FLAGS.optimizer)
    print("weight_decay: %f" % FLAGS.weight_decay)
    print("opt_epsilon: %.8f" % FLAGS.opt_epsilon)
    print("learning_rate: %f" % FLAGS.learning_rate)
    print("end_learning_rate: %f" % FLAGS.end_learning_rate)
    print("learning_rate_decay_type: %s" % FLAGS.learning_rate_decay_type)
    print("learning_rate_decay_factor: %f" % FLAGS.learning_rate_decay_factor)
    print("num_steps_per_decay: %f" % FLAGS.num_steps_per_decay)

