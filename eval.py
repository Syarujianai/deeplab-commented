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
"""Evaluation script for the DeepLab model.

See model.py for more details and usage.
"""

import six
import math
import tensorflow as tf
from deeplab import common
from deeplab import model
from deeplab.datasets import segmentation_dataset
from deeplab.utils import input_generator

slim = tf.contrib.slim

flags = tf.app.flags

FLAGS = flags.FLAGS

flags.DEFINE_string('master', '', 'BNS name of the tensorflow server')

# Settings for log directories.

flags.DEFINE_string('eval_logdir', None, 'Where to write the event logs.')

flags.DEFINE_string('checkpoint_dir', None, 'Directory of model checkpoints.')

# Settings for evaluating the model.

flags.DEFINE_integer('eval_batch_size', 1,
                     'The number of images in each batch during evaluation.')

flags.DEFINE_multi_integer('eval_crop_size', [513, 513],
                           'Image crop size [height, width] for evaluation.')

flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                     'How often (in seconds) to run evaluation.')

# For `xception_65`, use atrous_rates = [12, 24, 36] if output_stride = 8, or
# rates = [6, 12, 18] if output_stride = 16. For `mobilenet_v2`, use None. Note
# one could use different atrous_rates/output_stride during training/evaluation.
flags.DEFINE_multi_integer('atrous_rates', None,
                           'Atrous rates for atrous spatial pyramid pooling.')

flags.DEFINE_integer('output_stride', 16,
                     'The ratio of input to output spatial resolution.')

# Change to [0.5, 0.75, 1.0, 1.25, 1.5, 1.75] for multi-scale test.
flags.DEFINE_multi_float('eval_scales', [1.0],
                         'The scales to resize images for evaluation.')

# Change to True for adding flipped images during test.
flags.DEFINE_bool('add_flipped_images', False,
                  'Add flipped images for evaluation or not.')

# Dataset settings.

flags.DEFINE_string('dataset', 'pascal_voc_seg',
                    'Name of the segmentation dataset.')

flags.DEFINE_string('eval_split', 'val',
                    'Which split of the dataset used for evaluation')

flags.DEFINE_string('dataset_dir', None, 'Where the dataset reside.')

flags.DEFINE_integer('max_number_of_evaluations', 0,
                     'Maximum number of eval iterations. Will loop '
                     'indefinitely upon nonpositive values.')


def main(unused_argv):
  tf.logging.set_verbosity(tf.logging.INFO)
  # Get dataset-dependent information.
  dataset = segmentation_dataset.get_dataset(
      FLAGS.dataset, FLAGS.eval_split, dataset_dir=FLAGS.dataset_dir)

  tf.gfile.MakeDirs(FLAGS.eval_logdir)
  tf.logging.info('Evaluating on %s set', FLAGS.eval_split)

  with tf.Graph().as_default():
    samples = input_generator.get(
        dataset,
        FLAGS.eval_crop_size,
        FLAGS.eval_batch_size,
        min_resize_value=FLAGS.min_resize_value,
        max_resize_value=FLAGS.max_resize_value,
        resize_factor=FLAGS.resize_factor,
        dataset_split=FLAGS.eval_split,
        is_training=False,
        model_variant=FLAGS.model_variant)

    model_options = common.ModelOptions(
        outputs_to_num_classes={common.OUTPUT_TYPE: dataset.num_classes},
        crop_size=FLAGS.eval_crop_size,
        atrous_rates=FLAGS.atrous_rates,
        output_stride=FLAGS.output_stride)

    if tuple(FLAGS.eval_scales) == (1.0,): # syaru: `(1.0,)`, it's a tupple!
      tf.logging.info('Performing single-scale test.')
      """ 
      syaru: 
      Returns: A dictionary with keys specifying the output_type (e.g., semantic
      prediction) and values storing Tensors representing predictions (argmax
      over channels). Each prediction has size [batch, height, width].
      """
      predictions = model.predict_labels(samples[common.IMAGE], model_options,
                                         image_pyramid=FLAGS.image_pyramid)
    else:
      tf.logging.info('Performing multi-scale test.')
      predictions = model.predict_labels_multi_scale(
          samples[common.IMAGE],
          model_options=model_options,
          eval_scales=FLAGS.eval_scales,
          add_flipped_images=FLAGS.add_flipped_images)
    # syaru: common.OUTPUT_TYPE = 'semantic'
    #        common.LABEL='label'
    # `tf.reshape(..., shape=[-1])`: -1自动匹配reshape成一维，下述操作将predictions和labels从[batch, image_height, image_width, 1]平铺成一维
    # `tf.not_equal(x, y, name=None)`: Returns a Tensor of type bool, which represents the truth value of (x != y) element-wise.
    predictions = predictions[common.OUTPUT_TYPE]
    predictions = tf.reshape(predictions, shape=[-1])
    labels = tf.reshape(samples[common.LABEL], shape=[-1])
    weights = tf.to_float(tf.not_equal(labels, dataset.ignore_label))

    # Set ignore_label regions to label 0, because metrics.mean_iou requires
    # range of labels = [0, dataset.num_classes]. Note the ignore_label regions
    # are not evaluated since the corresponding regions contain weights = 0.
    # syaru :label等于ignore_label的样本将不参与Loss计算，并且反向传播时梯度直接置0. 
    # 为了符合tf.metrics.mean_iou的range([0, dataset.num_classes]), 利用`tf.where(input, a, b)`实现.
    # `tf.where(input, a, b)`: 其中a，b均为尺寸一致的tensor，作用是将a中对应input中true的位置的元素值不变，其余元素进行替换.
    # `tf.zeros_like(labels)`: 生成与labels shape相同的的全零tensor，作为参数pass进tf.where将labels中值为255(dataset.ignore_label=255)的替换为0.
    labels = tf.where(
        tf.equal(labels, dataset.ignore_label), tf.zeros_like(labels), labels)

    predictions_tag = 'miou'
    for eval_scale in FLAGS.eval_scales:
      predictions_tag += '_' + str(eval_scale)
    if FLAGS.add_flipped_images:
      predictions_tag += '_flipped'

    # Define the evaluation metric.
    # syaru: With regard to segmentation task, in `tensorflow/python/ops/metrics_impl.py`, IOU is defined as follows:
    # IOU = true_positive / (true_positive + false_positive + false_negative).
    # 其实很好理解, segmentation task相当于dense prediction, prediction/labels都平铺成一维, false negative是两者都黑的部分, 不参与交并比的计算. 
    metric_map = {}
    metric_map[predictions_tag] = tf.metrics.mean_iou(
        predictions, labels, dataset.num_classes, weights=weights)

    # syaru: `tensorflow/contrib/ops/metrics/python/ops/metric_ops.py`
    # Pairing metric names with their associated value and update ops (when the list of metrics is long) -> (value_tensor, update_op).
    # metric name: 'miou_1.0' (only one name in this ocasion), value and update ops: tf.metrics.mean_iou(predictions, labels, dataset.num_classes, weights=weights)
    # Returns: A `dictionary` from `metric names to value ops` and a `dictionary` from `metric names to update ops`.
    # In this ocasion, it will returns `dict1={'miou_1.0': value_tensor}, dict2={'miou_1.0': update_op}`
    metrics_to_values, metrics_to_updates = (
        tf.contrib.metrics.aggregate_metric_map(metric_map))

    # syaru: key: `metric_name, value: `metric_value`
    for metric_name, metric_value in six.iteritems(metrics_to_values):
      slim.summaries.add_scalar_summary(
          metric_value, metric_name, print_summary=True)

    # syaru: 'eval_batch_size'= 1, dataset.num_samples为`eval`验证集的图片数
    num_batches = int(
        math.ceil(dataset.num_samples / float(FLAGS.eval_batch_size)))

    tf.logging.info('Eval num images %d', dataset.num_samples)
    tf.logging.info('Eval batch size %d and num batch %d',
                    FLAGS.eval_batch_size, num_batches)

    # syaru: 命令行传入的max_number_of_evaluations=1, num_eval_iters!=None时, evaluation_loop只会迭代一次
    # `tensorflow/contrib/slim/python/slim/evaluation.py`
    num_eval_iters = None
    if FLAGS.max_number_of_evaluations > 0:
      num_eval_iters = FLAGS.max_number_of_evaluations
    slim.evaluation.evaluation_loop(
        master=FLAGS.master,
        checkpoint_dir=FLAGS.checkpoint_dir,
        logdir=FLAGS.eval_logdir,
        num_evals=num_batches,
        eval_op=list(metrics_to_updates.values()),
        max_number_of_evaluations=num_eval_iters,
        eval_interval_secs=FLAGS.eval_interval_secs)

# syaru: 当前模块名, 被直接运行时模块名为 __main__, 被导入时则为__name__.
# if __name__ == '__main__': 当模块被直接运行时, 以下代码块将被运行, 当模块是被导入时, 代码块不被运行.
if __name__ == '__main__':
  flags.mark_flag_as_required('checkpoint_dir')
  flags.mark_flag_as_required('eval_logdir')
  flags.mark_flag_as_required('dataset_dir')
  tf.app.run() # run(...): Runs the program with an optional 'main' function and 'argv' list.
