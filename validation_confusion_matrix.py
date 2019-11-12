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
import numpy as np
import sys

FLAGS = tf.app.flags.FLAGS

batch_size = FLAGS.num_validation
max_num_batches = 1
dataset_split_name = "validation"

def execute(checkpoint_path, model_no):
    if FLAGS.num_validation == 0:
        print("FLAGS.num_validation is 0, no need to validation")
        return None
    
    if checkpoint_path == None:
        checkpoint_path = tf.train.latest_checkpoint(FLAGS.train_dir)

    print("Begin validation_confusion_matrix %s" % format(datetime.now().isoformat()))
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
        y = endpoints[predictions_key]
        test_labels = tf.argmax(y_, 1, name="label")

        if FLAGS.moving_average_decay:
            variable_averages = tf.train.ExponentialMovingAverage(
              FLAGS.moving_average_decay, tf_global_step)
            variables_to_restore = variable_averages.variables_to_restore(
              slim.get_model_variables())
            variables_to_restore[tf_global_step.op.name] = tf_global_step
        else:
            variables_to_restore = slim.get_variables_to_restore()

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
            saver.restore(sess, checkpoint_path)
            images, labels = sess.run(next_batch)

            predictions = sess.run(fetches=y, feed_dict={x: images})
            predictions = np.squeeze(predictions)
            ids = sess.run(test_labels, feed_dict={y_: labels})
            errorList = []
            v_records = []
            for i in range(batch_size):
                prediction = predictions[i]
                top_k = prediction.argsort()[-5:][::-1]
                if ids[i] != top_k[0]:
                    errorList.append(str(ids[i]) + ":" + str(top_k[0]))
                v_record = str(ids[i]) + " " + labels_to_names[ids[i]] + " => "
                #print(ids[i], labels_to_names[ids[i]], "=> ", end='')
                for id in top_k:
                    human_string = labels_to_names[id]
                    score = prediction[id]
                    v_record = v_record + str(id) + ":" + human_string + "(P=" + str(score) + "), "
                    #print('%d:%s(P=%.5f), ' % (id, human_string, score), end='')
                print(v_record)
                v_records.append(v_record)
            print(errorList)
            errorid_filename = os.path.join(FLAGS.inference_dir, model_no + "_error.csv")
            print("Write file: %s ..."%errorid_filename)
            with tf.gfile.Open(errorid_filename, 'w') as f:
                for idmap in errorList:
                    f.write('%s\n' % (idmap))
            validation_record_filename = os.path.join(FLAGS.inference_dir, model_no + "_validation_record.txt")
            print("Write file: %s ..." % validation_record_filename)
            with tf.gfile.Open(validation_record_filename, 'w') as f:
                for v_rec in v_records:
                    f.write('%s\n' % (v_rec))
            sess.close()
    t2 = time()
    print("End validation_confusion_matrix %d s" %(t2 - t1))
    sys.stdout.flush()



