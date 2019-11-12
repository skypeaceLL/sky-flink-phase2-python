from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os
from datetime import datetime
from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def
from tensorflow.contrib import slim
from nets import nets_factory
from datasets import dataset_utils

FLAGS = tf.app.flags.FLAGS

def export(checkpoint_path, modelNo):

    print("Begin exporting %s" % format(datetime.now().isoformat()))
    
    saved_model_dir = "SavedModel"
    
    inference_dir = os.environ['MODEL_INFERENCE_PATH']
    export_dir = os.path.join(inference_dir, saved_model_dir, modelNo, "SavedModel")

    print("The path of saved model: %s"%export_dir)

    if tf.gfile.Exists(export_dir):
        print('Saved model folder already exist. Delete it firstly.')
        if(export_dir.endswith(saved_model_dir)):
            tf.gfile.DeleteRecursively(export_dir)

    if(checkpoint_path==None):
        checkpoint_path = tf.train.latest_checkpoint(FLAGS.train_dir)

    print("checkpoint_path: %s"%checkpoint_path)

    with tf.Graph().as_default() as graph:
        tf_global_step = slim.get_or_create_global_step()
        labels_to_names = dataset_utils.read_label_file(FLAGS.dataset_dir)
        num_classes = len(labels_to_names)
        network_fn = nets_factory.get_network_fn(
            FLAGS.model_name,
            num_classes=(num_classes - FLAGS.labels_offset),
            weight_decay=FLAGS.weight_decay,
            is_training=False)

        input_shape = [None, FLAGS.train_image_size, FLAGS.train_image_size, 3]
        input_tensor = tf.placeholder(name='input_1', dtype=tf.float32, shape=input_shape)

        predictions_key = "Predictions"
        if FLAGS.model_name.startswith("resnet"):
            logits, endpoints = network_fn(input_tensor)
            predictions_key = "predictions"
        elif FLAGS.model_name.startswith("inception"):
            logits, endpoints = network_fn(input_tensor, create_aux_logits=False)
        elif FLAGS.model_name.startswith("nasnet_mobile"):
            logits, endpoints = network_fn(input_tensor, use_aux_head=0)

        predictions = endpoints[predictions_key]

        if FLAGS.moving_average_decay:
            variable_averages = tf.train.ExponentialMovingAverage(
              FLAGS.moving_average_decay, tf_global_step)
            variables_to_restore = variable_averages.variables_to_restore(
              slim.get_model_variables())
            variables_to_restore[tf_global_step.op.name] = tf_global_step
        else:
            variables_to_restore = slim.get_variables_to_restore()

        saver = tf.train.Saver(var_list=variables_to_restore) #Same as slim.get_variables()

        init1 = tf.global_variables_initializer()
        init2 = tf.local_variables_initializer()
        with tf.Session() as sess:
            sess.run(init1)
            sess.run(init2)
            saver.restore(sess, checkpoint_path)

            #uninitialized_variables = [str(v, 'utf-8') for v in set(sess.run(tf.report_uninitialized_variables()))]
            #print(uninitialized_variables)
            #tf.graph_util.convert_variables_to_constants()

            print("Exporting saved model to: %s" % export_dir)

            prediction_signature = predict_signature_def(
                inputs={'input_1': input_tensor},
                outputs={'output': predictions})

            signature_def_map = {
                tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY: prediction_signature
            }

            builder = tf.saved_model.builder.SavedModelBuilder(export_dir)

            builder.add_meta_graph_and_variables(
                sess,
                tags = [tf.saved_model.tag_constants.SERVING],
                signature_def_map=signature_def_map,
                clear_devices=True,
                main_op=None,              #Suggest tf.tables_initializer()?
                strip_default_attrs=False) #Suggest True?
            builder.save()
            sess.close()
            print("Done exporting %s" % format(datetime.now().isoformat()))

