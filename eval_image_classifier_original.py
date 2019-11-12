from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os

from time import time
from datetime import datetime
from nets import nets_factory
from datasets import dataset_factory
from preprocessing import preprocessing_factory
from tensorflow.contrib import slim
import math
import sys

FLAGS = tf.app.flags.FLAGS

batch_size = 100
max_num_batches = FLAGS.num_validation // batch_size
dataset_split_name = "train"
master = ""
quantize = False

def print_train_acc():
  if not FLAGS.dataset_dir:
    raise ValueError('You must supply the dataset directory with --dataset_dir')

  print("Begin evaluate train_accuracy %s" % format(datetime.now().isoformat()))
  t1 = time()

  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
  with tf.Graph().as_default():
    tf_global_step = slim.get_or_create_global_step()

    ######################
    # Select the dataset #
    ######################
    dataset = dataset_factory.get_dataset(
        FLAGS.dataset_name, dataset_split_name, FLAGS.dataset_dir)

    ####################
    # Select the model #
    ####################
    network_fn = nets_factory.get_network_fn(
        FLAGS.model_name,
        num_classes=(dataset.num_classes - FLAGS.labels_offset),
        is_training=False)

    ##############################################################
    # Create a dataset provider that loads data from the dataset #
    ##############################################################
    provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        shuffle=False,
        common_queue_capacity=2 * batch_size,
        common_queue_min=batch_size)
    [image, label] = provider.get(['image', 'label'])
    label -= FLAGS.labels_offset

    #####################################
    # Select the preprocessing function #
    #####################################
    preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
        preprocessing_name,
        is_training=False)

    eval_image_size = FLAGS.train_image_size

    image = image_preprocessing_fn(image, eval_image_size, eval_image_size)

    images, labels = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        num_threads=FLAGS.num_preprocessing_threads,
        capacity=5 * batch_size)

    ####################
    # Define the model #
    ####################
    logits, _ = network_fn(images)

    if quantize:
      tf.contrib.quantize.create_eval_graph()

    if FLAGS.moving_average_decay:
      variable_averages = tf.train.ExponentialMovingAverage(
          FLAGS.moving_average_decay, tf_global_step)
      variables_to_restore = variable_averages.variables_to_restore(
          slim.get_model_variables())
      variables_to_restore[tf_global_step.op.name] = tf_global_step
    else:
      variables_to_restore = slim.get_variables_to_restore()

    predictions = tf.argmax(logits, 1)
    labels = tf.squeeze(labels)

    # Define the metrics:
    names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
        'Train_Accuracy': slim.metrics.streaming_accuracy(predictions, labels),
        'Train_Recall_5': slim.metrics.streaming_recall_at_k(
            logits, labels, 5),
    })

    # Print the summaries to screen.
    for name, value in names_to_values.items():
      summary_name = 'eval/%s' % name
      op = tf.summary.scalar(summary_name, value, collections=[])
      op = tf.Print(op, [value], summary_name)
      tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)

    if max_num_batches:
      num_batches = max_num_batches
    else:
      # This ensures that we make a single pass over all of the data.
      num_batches = math.ceil(dataset.num_samples / float(batch_size))

    if tf.gfile.IsDirectory(FLAGS.train_dir):
      checkpoint_path = tf.train.latest_checkpoint(FLAGS.train_dir)
    else:
      checkpoint_path = FLAGS.train_dir

    tf.logging.info('Evaluating on train_dir: %s' % FLAGS.train_dir)

    dirlist = os.listdir(FLAGS.train_dir)
    file_numbers = []
    for file_name in dirlist:
        if file_name.startswith("model.ckpt-") & file_name.endswith(".index"):
            idx = file_name.replace("model.ckpt-", "").replace(".index", "")
            file_numbers.append(int(idx))
    file_numbers.sort()
    for file_number in file_numbers[-1:]:
        file_name = "model.ckpt-%d" % file_number
        checkpoint_path = os.path.join(FLAGS.train_dir, file_name)
        tf.logging.info('Evaluating %s' % checkpoint_path)
        slim.evaluation.evaluate_once(
                master=master,
                checkpoint_path=checkpoint_path,
                logdir=FLAGS.train_dir,
                num_evals=num_batches,
                eval_op=list(names_to_updates.values()),
                variables_to_restore=variables_to_restore)
  
  t2 = time()
  print("End train_accuracy %d s" %(t2 - t1))
  sys.stdout.flush()

