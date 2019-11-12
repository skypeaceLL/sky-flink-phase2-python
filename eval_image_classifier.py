from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os

from time import time
from datetime import datetime
from nets import nets_factory
from datasets import dataset_utils
from preprocessing import preprocessing_factory
from tensorflow.contrib import slim
#from tensorflow.python.training import checkpoint_management
import sys

FLAGS = tf.app.flags.FLAGS

batch_size = FLAGS.num_validation
max_num_batches = 1
dataset_split_name = "validation"

def find_max_accuracy_checkpoint():

    if FLAGS.num_validation == 0:
        print("FLAGS.num_validation is 0, no need to validation")
        return None

    print("Begin evaluate val_accuracy %s" % format(datetime.now().isoformat()))
    t1 = time()

    preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
        preprocessing_name, is_training=False)

    labels_to_names = dataset_utils.read_label_file(FLAGS.dataset_dir)
    num_classes = len(labels_to_names)

    def decode(serialized_example):
        feature = {
            'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
            'image/format': tf.FixedLenFeature((), tf.string, default_value='jpg'),
            'image/class/label': tf.FixedLenFeature(
                [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
        }
        features = tf.parse_single_example(serialized_example, features=feature)
        # image
        image_string = features['image/encoded']
        image = tf.image.decode_jpeg(image_string, channels=3)
        image = image_preprocessing_fn(image, FLAGS.train_image_size, FLAGS.train_image_size)
        # label
        label = features['image/class/label']
        label = tf.one_hot(label, num_classes)
        return image, label

    def input_iter(filenames, batch_size, num_epochs):
        if not num_epochs:
            num_epochs = 1
        dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=FLAGS.num_readers)
        dataset = dataset.map(decode)
        dataset = dataset.repeat(num_epochs)
        dataset = dataset.batch(batch_size)
        # dataset = dataset.shuffle(buffer_size=NUM_IMAGES)
        iterator = dataset.make_one_shot_iterator()
        return iterator

    with tf.Graph().as_default() as graph:
        tf_global_step = slim.get_or_create_global_step()
        network_fn = nets_factory.get_network_fn(
            FLAGS.model_name,
            num_classes=(num_classes - FLAGS.labels_offset),
            weight_decay=FLAGS.weight_decay,
            is_training=False)

        eval_image_size = FLAGS.train_image_size

        x = tf.placeholder(tf.float32, [None, eval_image_size, eval_image_size, 3])
        y_ = tf.placeholder(tf.float32, [None, num_classes])

        logits, endpoints = network_fn(x)

        predictions_key = "Predictions"
        if FLAGS.model_name.startswith("resnet"):
            predictions_key = "predictions"
        t_prediction = endpoints[predictions_key]

        if FLAGS.moving_average_decay:
            variable_averages = tf.train.ExponentialMovingAverage(
              FLAGS.moving_average_decay, tf_global_step)
            variables_to_restore = variable_averages.variables_to_restore(
              slim.get_model_variables())
            variables_to_restore[tf_global_step.op.name] = tf_global_step
        else:
            variables_to_restore = slim.get_variables_to_restore()

        predictions = tf.argmax(t_prediction, 1, name="prediction")
        test_labels = tf.argmax(y_, 1, name="label")
        correct_prediction = tf.equal(predictions, test_labels)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")

        input_dir = []
        for i in range(5) :
            data_file = os.path.join(FLAGS.dataset_dir, "garbage_validation_0000%d-of-00005.tfrecord")%i
            input_dir.append(data_file)

        iter = input_iter(input_dir, batch_size, 1)
        next_batch = iter.get_next()

        saver = tf.train.Saver(var_list=variables_to_restore) #Same as slim.get_variables()
        init1 = tf.global_variables_initializer()
        init2 = tf.local_variables_initializer()
        with tf.Session() as sess:
            sess.run(init1)
            sess.run(init2)
            images, labels = sess.run(next_batch)

            dirlist = os.listdir(FLAGS.train_dir)
            file_numbers = []
            for file_name in dirlist:
                if file_name.startswith("model.ckpt-") & file_name.endswith(".index"):
                    idx = file_name.replace("model.ckpt-","").replace(".index","")
                    file_numbers.append(int(idx))
            file_numbers.sort()
            maxAccuracy = 0.0
            maxAccuracyCheckPoint = ""
            for file_number in file_numbers:
                if file_number<=0:
                    continue
                file_name = "model.ckpt-%d"%file_number
                checkpoint_path = os.path.join(FLAGS.train_dir, file_name)
                print('Evaluate val_accuracy on %s' % checkpoint_path)
                saver.restore(sess, checkpoint_path)
                train_accuracy = sess.run(fetches=accuracy, feed_dict={x: images, y_: labels})
                print("Val_accuracy: {0}".format(train_accuracy))
                if train_accuracy >= (maxAccuracy + 0.0000):
                    maxAccuracy = train_accuracy
                    maxAccuracyCheckPoint = checkpoint_path
            print("Max val_accuracy: %f"%maxAccuracy)
            print("maxAccuracyCheckPoint: %s"%maxAccuracyCheckPoint)
            sess.close()
            t2 = time()
            print("End val_accuracy %d s" %(t2 - t1))
            sys.stdout.flush()
            return maxAccuracyCheckPoint


