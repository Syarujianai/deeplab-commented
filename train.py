# Copyright 2018 The TensorFlow Authors All Rights Reserved.
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
"""Training script for the DeepLab model.

See model.py for more details and usage.
"""

import six
import tensorflow as tf
from deeplab import common
from deeplab import model
from deeplab.datasets import segmentation_dataset
from deeplab.utils import input_generator
from deeplab.utils import train_utils
from deployment import model_deploy

slim = tf.contrib.slim

prefetch_queue = slim.prefetch_queue

flags = tf.app.flags

FLAGS = flags.FLAGS

# Settings for multi-GPUs/multi-replicas training.

# syaru: tf定义了tf.app.flags, 用于支持接受命令行传递参数, 相当于接受argv.
# 第一个是参数名称,第二个参数是默认值,第三个是参数描述.
flags.DEFINE_integer('num_clones', 1, 'Number of clones to deploy.')

flags.DEFINE_boolean('clone_on_cpu', False, 'Use CPUs to deploy clones.')

flags.DEFINE_integer('num_replicas', 1, 'Number of worker replicas.')

flags.DEFINE_integer('startup_delay_steps', 15,
                     'Number of training steps between replicas startup.')

flags.DEFINE_integer('num_ps_tasks', 0,
                     'The number of parameter servers. If the value is 0, then '
                     'the parameters are handled locally by the worker.')

flags.DEFINE_string('master', '', 'BNS name of the tensorflow server')

flags.DEFINE_integer('task', 0, 'The task ID.')

# Settings for logging.

flags.DEFINE_string('train_logdir', None,
                    'Where the checkpoint and logs are stored.')

flags.DEFINE_integer('log_steps', 10,
                     'Display logging information at every log_steps.')

flags.DEFINE_integer('save_interval_secs', 1200,
                     'How often, in seconds, we save the model to disk.')

flags.DEFINE_integer('save_summaries_secs', 600,
                     'How often, in seconds, we compute the summaries.')

flags.DEFINE_boolean('save_summaries_images', False,
                     'Save sample inputs, labels, and semantic predictions as images to summary.')

# Settings for training strategy.

flags.DEFINE_enum('learning_policy', 'poly', ['poly', 'step'],
                  'Learning rate policy for training.')

# Use 0.007 when training on PASCAL augmented training set, train_aug. When
# fine-tuning on PASCAL trainval set, use learning rate=0.0001.
flags.DEFINE_float('base_learning_rate', .0001,
                   'The base learning rate for model training.')

flags.DEFINE_float('learning_rate_decay_factor', 0.1,
                   'The rate to decay the base learning rate.')

flags.DEFINE_integer('learning_rate_decay_step', 2000,
                     'Decay the base learning rate at a fixed step.')

flags.DEFINE_float('learning_power', 0.9,
                   'The power value used in the poly learning policy.')

flags.DEFINE_integer('training_number_of_steps', 30000,
                     'The number of steps used for training')

flags.DEFINE_float('momentum', 0.9, 'The momentum value to use')

# When fine_tune_batch_norm=True, use at least batch size larger than 12
# (batch size more than 16 is better). Otherwise, one could use smaller batch
# size and set fine_tune_batch_norm=False.
flags.DEFINE_integer('train_batch_size', 8,
                     'The number of images in each batch during training.')

flags.DEFINE_float('weight_decay', 0.00004,
                   'The value of the weight decay for training.')

flags.DEFINE_multi_integer('train_crop_size', [513, 513],
                           'Image crop size [height, width] during training.')

flags.DEFINE_float('last_layer_gradient_multiplier', 1.0,
                   'The gradient multiplier for last layers, which is used to '
                   'boost the gradient of last layers if the value > 1.')

flags.DEFINE_boolean('upsample_logits', True,
                     'Upsample logits during training.')

# Settings for fine-tuning the network.

flags.DEFINE_string('tf_initial_checkpoint', None,
                    'The initial checkpoint in tensorflow format.')

# Set to False if one does not want to re-use the trained classifier weights.
flags.DEFINE_boolean('initialize_last_layer', True,
                     'Initialize the last layer.')

flags.DEFINE_boolean('last_layers_contain_logits_only', False,
                     'Only consider logits as last layers or not.')

flags.DEFINE_integer('slow_start_step', 0,
                     'Training model with small learning rate for few steps.')

flags.DEFINE_float('slow_start_learning_rate', 1e-4,
                   'Learning rate employed during slow start.')

# Set to True if one wants to fine-tune the batch norm parameters in DeepLabv3.
# Set to False and use small batch size to save GPU memory.
flags.DEFINE_boolean('fine_tune_batch_norm', True,
                     'Fine tune the batch norm parameters or not.')

flags.DEFINE_float('min_scale_factor', 0.5,
                   'Mininum scale factor for data augmentation.')

flags.DEFINE_float('max_scale_factor', 2.,
                   'Maximum scale factor for data augmentation.')

flags.DEFINE_float('scale_factor_step_size', 0.25,
                   'Scale factor step size for data augmentation.')

# For `xception_65`, use atrous_rates = [12, 24, 36] if output_stride = 8, or
# rates = [6, 12, 18] if output_stride = 16. For `mobilenet_v2`, use None. Note
# one could use different atrous_rates/output_stride during training/evaluation.
flags.DEFINE_multi_integer('atrous_rates', None,
                           'Atrous rates for atrous spatial pyramid pooling.')

flags.DEFINE_integer('output_stride', 16,
                     'The ratio of input to output spatial resolution.')

# Dataset settings.
flags.DEFINE_string('dataset', 'pascal_voc_seg',
                    'Name of the segmentation dataset.')

flags.DEFINE_string('train_split', 'train',
                    'Which split of the dataset to be used for training')

flags.DEFINE_string('dataset_dir', None, 'Where the dataset reside.')


def _build_deeplab(inputs_queue, outputs_to_num_classes, ignore_label):
  """Builds a clone of DeepLab.

  Args:
    inputs_queue: A prefetch queue for images and labels.
    outputs_to_num_classes: A map from output type to the number of classes.
      For example, for the task of semantic segmentation with 21 semantic
      classes, we would have outputs_to_num_classes['semantic'] = 21.
    ignore_label: Ignore label.

  Returns:
    A map of maps from output_type (e.g., semantic prediction) to a
      dictionary of multi-scale logits names to logits. For each output_type,
      the dictionary has keys which correspond to the scales and values which
      correspond to the logits. For example, if `scales` equals [1.0, 1.5],
      then the keys would include 'merged_logits', 'logits_1.00' and
      'logits_1.50'.
  """
  samples = inputs_queue.dequeue()

  # add name to input and label nodes so we can add to summary
  # syaru: tf.identity(samples['image']): transform to tensor(ops) 
  samples[common.IMAGE] = tf.identity(samples[common.IMAGE], name = common.IMAGE) # syaru: common.IMAGE = 'image'
  samples[common.LABEL] = tf.identity(samples[common.LABEL], name = common.LABEL) #        common.LABEL = 'label'

  model_options = common.ModelOptions(
      outputs_to_num_classes=outputs_to_num_classes,
      crop_size=FLAGS.train_crop_size,
      atrous_rates=FLAGS.atrous_rates,
      output_stride=FLAGS.output_stride)
  # syaru: model.multi_scale_logits(): Gets the logits for multi-scale inputs.
  # The returned logits are all downsampled (due to max-pooling layers)
  # for both training and evaluation.
  outputs_to_scales_to_logits = model.multi_scale_logits(
      samples[common.IMAGE],              # syaru: images: A tensor of size [batch, height, width, channels].
      model_options=model_options,        # model_options: A ModelOptions instance to configure models.
      image_pyramid=FLAGS.image_pyramid,  # image_pyramid: Input image scales for multi-scale feature extraction.
      weight_decay=FLAGS.weight_decay,
      is_training=True,
      fine_tune_batch_norm=FLAGS.fine_tune_batch_norm)

  # add name to graph node so we can add to summary
  outputs_to_scales_to_logits[common.OUTPUT_TYPE][model._MERGED_LOGITS_SCOPE] = tf.identity( # syaru: common.OUTPUT_TYPE = 'semantic'
    outputs_to_scales_to_logits[common.OUTPUT_TYPE][model._MERGED_LOGITS_SCOPE],             # model._MERGED_LOGITS_SCOPE = 'merged_logits'
    name = common.OUTPUT_TYPE
  )

  for output, num_classes in six.iteritems(outputs_to_num_classes): # syaru: six.iteritems(): 迭代输出字典的键值(outputs_to_num_classes is a dict)
    train_utils.add_softmax_cross_entropy_loss_for_each_scale(      # deeplab.utils.train_utils: Adds softmax cross entropy loss for logits of each scale
        outputs_to_scales_to_logits[output],                        # scales_to_logits: A map from logits names for different scales to logits. 
        samples[common.LABEL],                                      # The logits have shape [batch, logits_height, logits_width, num_classes]. 
        num_classes,                                                # labels: Groundtruth labels with shape [batch, image_height, image_width, 1].
        ignore_label,
        loss_weight=1.0,
        upsample_logits=FLAGS.upsample_logits,
        scope=output)

  return outputs_to_scales_to_logits


def main(unused_argv):
  # syaru: Sets the threshold(入口) for what messages will be logged. 加上这句才能输出训练过程的log.
  tf.logging.set_verbosity(tf.logging.INFO) 
  # Set up deployment (i.e., multi-GPUs and/or multi-replicas).
  # syaru: models/research/slim/deployment/model_deploy.DeploymentConfig(object)
  config = model_deploy.DeploymentConfig(
      num_clones=FLAGS.num_clones,
      clone_on_cpu=FLAGS.clone_on_cpu,
      replica_id=FLAGS.task,
      num_replicas=FLAGS.num_replicas,
      num_ps_tasks=FLAGS.num_ps_tasks)

  # Split the batch across GPUs.
  assert FLAGS.train_batch_size % config.num_clones == 0, (
      'Training batch size not divisble by number of clones (GPUs).')

  clone_batch_size = int(FLAGS.train_batch_size / config.num_clones)

  # Get dataset-dependent information.
  """
  syaru: deeplab/datasets/segmentation_dataset.get_dataset()
  Gets an instance of slim Dataset.
  Args:
    dataset_name: Dataset name.
    split_name: A train/val Split name.
    dataset_dir: The directory of the dataset sources.
  """
  dataset = segmentation_dataset.get_dataset(
      FLAGS.dataset, FLAGS.train_split, dataset_dir=FLAGS.dataset_dir)

  tf.gfile.MakeDirs(FLAGS.train_logdir)                    # sayru: FLAGS.train_logdir = "pascal_voc_seg/exp/train_on_trainval_set/train"
  tf.logging.info('Training on %s set', FLAGS.train_split) #        FLAGS.train_split = "trainval"

  with tf.Graph().as_default() as graph:
    with tf.device(config.inputs_device()):                    # syaru: deeplab/utils/input_generator.get(): This functions gets the dataset split for semantic segmentation.
      samples = input_generator.get(                           # Returns: A dictionary of batched Tensors for semantic segmentation.
          dataset,                                             # Args: dataset: An instance of slim Dataset.
          FLAGS.train_crop_size,                               #       train_crop_size: 如果定义了crop_size，那么在train时会对大于crop_size的图片进行随机裁剪
          clone_batch_size,
          min_resize_value=FLAGS.min_resize_value,
          max_resize_value=FLAGS.max_resize_value,
          resize_factor=FLAGS.resize_factor,
          min_scale_factor=FLAGS.min_scale_factor,             # syaru: min_scale_factor: 'Minmum scale factor for data augmentation.'
          max_scale_factor=FLAGS.max_scale_factor,             # min_scale_factor: 'Maximum scale factor for data augmentation.'
          scale_factor_step_size=FLAGS.scale_factor_step_size, # scale_factor_step_size: 'Scale factor step size for data augmentation.'(from minmum to maximum)
          dataset_split=FLAGS.train_split,
          is_training=True,
          model_variant=FLAGS.model_variant)
      # syaru: /tensorflow/contrib/slim/python/slim/data/prefetch_queue.py
      inputs_queue = prefetch_queue.prefetch_queue(            # tensors: A list or dictionary of `Tensors` to enqueue in the buffer.
          samples, capacity=128 * config.num_clones)           # capacity: An integer. The maximum number of elements in the queue.

    # Create the global step on the device storing the variables.
    with tf.device(config.variables_device()):
      global_step = tf.train.get_or_create_global_step()

      # Define the model and create clones.
      """
      syaru: 
      models/research/slim/deployment/model_deploy.create_clones():
      The `model_fn(*args, **kwargs)` function is called `config.num_clones` times to create the model clones.
      (and one or several clones are deployed on different GPUs and one or several replicas of such clones.)
      Then it return the scope and device in a namedtuple `Clone(outputs, scope, device)`.

      Args:
      config: A DeploymentConfig object.
      model_fn: A callable. Called as `model_fn(*args, **kwargs)`
      args: Optional list of arguments to pass to `model_fn`.
      kwargs: Optional list of keyword arguments to pass to `model_fn`..
      Returns:
      A list of namedtuples `Clone`.

      Note: it is assumed that any loss created by `model_fn` is collected at
      the tf.GraphKeys.LOSSES collection.

      To recover the losses, summaries or update_ops created by the clone use:
      ```python
      losses = tf.get_collection(tf.GraphKeys.LOSSES, clone.scope)
      summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, clone.scope)
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, clone.scope)
      ```
      """
      model_fn = _build_deeplab
      model_args = (inputs_queue, {
          common.OUTPUT_TYPE: dataset.num_classes
      }, dataset.ignore_label)
      clones = model_deploy.create_clones(config, model_fn, args=model_args)

      # Gather update_ops from the first clone. These contain, for example,
      # the updates for the batch_norm variables created by model_fn.
      first_clone_scope = config.clone_scope(0)
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, first_clone_scope)

    # Gather initial summaries.
    summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

    # Add summaries for model variables.
    for model_var in slim.get_model_variables():
      summaries.add(tf.summary.histogram(model_var.op.name, model_var))

    # Add summaries for images, labels, semantic predictions
    if FLAGS.save_summaries_images:
        summary_image = graph.get_tensor_by_name(                                # syaru: get_tensor_by_name(name): return tensor by specifily 'name'.
            ('%s/%s:0' % (first_clone_scope, common.IMAGE)).strip('/'))          # str.strip (): is used to remove the specified characters at the front/end of the string (the default is space).
        summaries.add(tf.summary.image('samples/%s' % common.IMAGE, summary_image))

        summary_label = tf.cast(graph.get_tensor_by_name(
            ('%s/%s:0' % (first_clone_scope, common.LABEL)).strip('/')),
            tf.uint8)
        summaries.add(tf.summary.image('samples/%s' % common.LABEL, summary_label))

        predictions = tf.cast(tf.expand_dims(tf.argmax(graph.get_tensor_by_name( # syaru: tf.argmax(axis=3)
            ('%s/%s:0' % (first_clone_scope, common.OUTPUT_TYPE)).strip('/')),
            3), -1), tf.uint8)
        summaries.add(tf.summary.image('samples/%s' % common.OUTPUT_TYPE, predictions))

    # Add summaries for losses.
    for loss in tf.get_collection(tf.GraphKeys.LOSSES, first_clone_scope):
      summaries.add(tf.summary.scalar('losses/%s' % loss.op.name, loss))

    # Build the optimizer based on the device specification.
    with tf.device(config.optimizer_device()):
      # syaru: train_utils.get_model_learning_rate():
      #        Computes the model's learning rate for different learning policy("step" and "poly").
      learning_rate = train_utils.get_model_learning_rate(
          FLAGS.learning_policy, FLAGS.base_learning_rate,
          FLAGS.learning_rate_decay_step, FLAGS.learning_rate_decay_factor,
          FLAGS.training_number_of_steps, FLAGS.learning_power,
          FLAGS.slow_start_step, FLAGS.slow_start_learning_rate)
      optimizer = tf.train.MomentumOptimizer(learning_rate, FLAGS.momentum)
      summaries.add(tf.summary.scalar('learning_rate', learning_rate))

    startup_delay_steps = FLAGS.task * FLAGS.startup_delay_steps
    for variable in slim.get_model_variables():
      summaries.add(tf.summary.histogram(variable.op.name, variable))

    with tf.device(config.variables_device()):
      # syaru: Compute clone losses and gradients for the given list of `Clones`.
      total_loss, grads_and_vars = model_deploy.optimize_clones(
          clones, optimizer)
      total_loss = tf.check_numerics(total_loss, 'Loss is inf or nan.')
      summaries.add(tf.summary.scalar('total_loss', total_loss))

      # Modify the gradients for biases and last layer variables.
      """
      syaru: 
      For the task of semantic segmentation, the models are
      usually fine-tuned from the models trained on the task of image
      classification. To fine-tune the models, we usually set larger (e.g.,
      10 times larger) learning rate for the parameters of last layer.
      
      deeplab/model/model.get_extra_layer_scopes():
      Returns: A list of scopes for extra layers. 

      deeplab/utils/train_utils.get_model_gradient_multipliers():
      Returns: The gradient multiplier map with variables as key, and multipliers as value.
      """ 
      last_layers = model.get_extra_layer_scopes(FLAGS.last_layers_contain_logits_only)
      grad_mult = train_utils.get_model_gradient_multipliers(
          last_layers, FLAGS.last_layer_gradient_multiplier)
      if grad_mult:
        grads_and_vars = slim.learning.multiply_gradients(
            grads_and_vars, grad_mult)

      # Create gradient update op.
      grad_updates = optimizer.apply_gradients(
          grads_and_vars, global_step=global_step)
      update_ops.append(grad_updates)
      # syaru: tf.identity()和tf.group()均可将语句变为操作(ops). 
      #        (我们需要`optimizer.apply_gradients`后才计算`total_loss`(as 'train_op')，而tf.control_dependencies()适用于tf.ops)
      #        And `update_ops = tf.get_collection(..)` only return a list of variables.
      update_op = tf.group(*update_ops)
      with tf.control_dependencies([update_op]):
        train_tensor = tf.identity(total_loss, name='train_op')

    # Add the summaries from the first clone. These contain the summaries
    # created by model_fn and either optimize_clones() or _gather_clone_loss().
    summaries |= set(
        tf.get_collection(tf.GraphKeys.SUMMARIES, first_clone_scope))

    # Merge all summaries together.
    summary_op = tf.summary.merge(list(summaries))

    # Soft placement allows placing on CPU ops without GPU implementation.
    # syaru: set gpu_options
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    session_config = tf.ConfigProto(
        allow_soft_placement=True, log_device_placement=False, gpu_options=gpu_options)

    # Start the training.
    # syaru: /tensorflow/contrib/slim/python/slim/learning.py
    # train_utils.get_model_init_fn(): Gets the function initializing model variables from a checkpoint.
    slim.learning.train(
        train_tensor,
        logdir=FLAGS.train_logdir,
        log_every_n_steps=FLAGS.log_steps,
        master=FLAGS.master,
        number_of_steps=FLAGS.training_number_of_steps,
        is_chief=(FLAGS.task == 0),
        session_config=session_config,
        startup_delay_steps=startup_delay_steps, # syaru:
        init_fn=train_utils.get_model_init_fn(   # `init_fn`: An optional callable to be executed after `init_op` is called. The
            FLAGS.train_logdir,                  # callable must accept one argument, the session being initialized.
            FLAGS.tf_initial_checkpoint,
            FLAGS.initialize_last_layer,
            last_layers,
            ignore_missing_vars=True),
        summary_op=summary_op,
        save_summaries_secs=FLAGS.save_summaries_secs,
        save_interval_secs=FLAGS.save_interval_secs)

# syaru: mark_flag_as_required先按字面意思理解，标记必需的flag
# 详见Add mark_flag_as_required functions to make the APIs compatible with python-gflags.(tensorflow/tensorflow-#11195)
if __name__ == '__main__':
  flags.mark_flag_as_required('train_logdir')
  flags.mark_flag_as_required('tf_initial_checkpoint')
  flags.mark_flag_as_required('dataset_dir')
  tf.app.run() # syaru: run(...): Runs the program with an optional 'main' function and 'argv' list.
