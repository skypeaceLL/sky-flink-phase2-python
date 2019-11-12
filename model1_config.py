from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os

def init_params():

    FLAGS = tf.app.flags.FLAGS
    _checkpoint_path = os.path.join(FLAGS.script_root_dir, "pre_train", "resnet_v1_101.ckpt")

    #####################
    # Basic Flags #
    #####################
    tf.app.flags.DEFINE_string('checkpoint_path', _checkpoint_path, 'The path to a checkpoint from which to fine-tune.')
    tf.app.flags.DEFINE_integer('max_number_of_steps', 3500, 'The maximum number of training steps.') 
    # 3680: 20 epochs, 4800: 26 epochs
    tf.app.flags.DEFINE_integer('log_every_n_steps', 30, 'The frequency with which logs are print.')
    tf.app.flags.DEFINE_integer('save_summaries_secs', 600, 'The frequency with which summaries are saved, in seconds.')
    tf.app.flags.DEFINE_integer('save_interval_secs', 68, 'The frequency with which the model is saved, in seconds.') #68
    tf.app.flags.DEFINE_integer('max_to_keep', 12, 'max number of checkpoint to keep')
    tf.app.flags.DEFINE_integer('num_readers', 4, 'The number of parallel readers that read data from the dataset.')
    tf.app.flags.DEFINE_integer('num_preprocessing_threads', 4, 'The number of threads used to create the batches.')
    tf.app.flags.DEFINE_integer('num_clones', 1, 'Number of model clones to deploy.')
    tf.app.flags.DEFINE_boolean('clone_on_cpu', True, 'Use CPUs to deploy clones.')

    tf.app.flags.DEFINE_string('dataset_name', 'garbage', 'The name of the dataset to load.')
    tf.app.flags.DEFINE_string('dataset_split_name', 'train', 'The name of the train/test split.')
    tf.app.flags.DEFINE_integer('batch_size', 32, 'The number of samples in each batch.')
    tf.app.flags.DEFINE_integer('train_image_size', 224, 'Train image size')
    tf.app.flags.DEFINE_string('model_name', 'resnet_v1_101', 'The name of the architecture to train.')
    tf.app.flags.DEFINE_string('preprocessing_name', None, 'The name of the preprocessing to use. If None, then the model_name flag is used.')

    tf.app.flags.DEFINE_integer('labels_offset', 0, 'An offset for the labels in the dataset.')

    #####################
    # Fine-Tuning Flags #
    #####################
    tf.app.flags.DEFINE_string('checkpoint_exclude_scopes', 'resnet_v1_101/logits', 'Comma-separated list of scopes of variables.')
    tf.app.flags.DEFINE_string('trainable_scopes', 'resnet_v1_101/logits', 'Comma-separated list of scopes.')
    tf.app.flags.DEFINE_boolean('ignore_missing_vars', False, 'When restoring a checkpoint would ignore missing variables.')

    #######################
    # Distribution and clone Flags #
    #######################
    tf.app.flags.DEFINE_string('master', '', 'The address of the TensorFlow master to use.')
    tf.app.flags.DEFINE_integer('worker_replicas', 1, 'Number of worker replicas.')
    tf.app.flags.DEFINE_integer('num_ps_tasks', 0, 'The number of parameter servers.')
    tf.app.flags.DEFINE_integer('task', 0, 'Task id of the replica running the training.')
    tf.app.flags.DEFINE_bool('sync_replicas', False, 'Whether or not to synchronize the replicas during training.')
    tf.app.flags.DEFINE_integer('replicas_to_aggregate', 1, 'The Number of gradients to collect before updating params.')

    ######################
    # Optimization Flags #
    ######################
    tf.app.flags.DEFINE_string('optimizer', 'adam', '"adadelta", "adagrad", "adam", "ftrl", "momentum", "sgd" or "rmsprop".')
    tf.app.flags.DEFINE_float('weight_decay', 0.00004, 'The weight decay on the model weights.') #0.00004
    tf.app.flags.DEFINE_float('adadelta_rho', 0.95, 'The decay rate for adadelta.')
    tf.app.flags.DEFINE_float('adagrad_initial_accumulator_value', 0.1, 'Starting value for the AdaGrad accumulators.')
    tf.app.flags.DEFINE_float('adam_beta1', 0.9, 'The exponential decay rate for the 1st moment estimates.')
    tf.app.flags.DEFINE_float('adam_beta2', 0.999, 'The exponential decay rate for the 2nd moment estimates.')
    tf.app.flags.DEFINE_float('opt_epsilon', 1e-8, 'Epsilon term for the optimizer.') #1.0
    tf.app.flags.DEFINE_float('ftrl_learning_rate_power', -0.5, 'The learning rate power.')
    tf.app.flags.DEFINE_float('ftrl_initial_accumulator_value', 0.1, 'Starting value for the FTRL accumulators.')
    tf.app.flags.DEFINE_float('ftrl_l1', 0.0, 'The FTRL l1 regularization strength.')
    tf.app.flags.DEFINE_float('ftrl_l2', 0.0, 'The FTRL l2 regularization strength.')
    tf.app.flags.DEFINE_float('momentum', 0.9, 'The momentum for the MomentumOptimizer and RMSPropOptimizer.')
    tf.app.flags.DEFINE_float('rmsprop_momentum', 0.9, 'Momentum.')
    tf.app.flags.DEFINE_float('rmsprop_decay', 0.9, 'Decay term for RMSProp.')
    tf.app.flags.DEFINE_integer('quantize_delay', -1,'Number of steps to start quantized training. Set to -1 would disable quantized training.')

    #######################
    # Learning Rate Flags #
    #######################
    tf.app.flags.DEFINE_string(
        'learning_rate_decay_type',
        'exponential',
        'Specifies how the learning rate is decayed. One of "fixed", "exponential",'
        ' or "polynomial"')

    tf.app.flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')  #0.01, 0.001
    tf.app.flags.DEFINE_float('end_learning_rate', 0.000001, 'The minimal end learning rate used by a polynomial decay learning rate.')#0.0001
    tf.app.flags.DEFINE_float('label_smoothing', 0.0, 'The amount of label smoothing.')
    tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.3, 'Learning rate decay factor.') #0.94, 0.3

    tf.app.flags.DEFINE_float(
        'num_steps_per_decay', 1000.0,
        'Number of epochs after which learning rate decays. Note: this flag counts '
        'epochs per clone but aggregates per sync replicas. So 1.0 means that '
        'each clone will go over full epoch individually, but replicas will go '
        'once across all replicas.')

    tf.app.flags.DEFINE_float('moving_average_decay', None, 'The decay to use for the moving average.'
        'If left as None, then moving averages are not used.') #None, 0.99
