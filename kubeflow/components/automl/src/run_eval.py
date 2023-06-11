import copy
import os
from absl import flags

import tensorflow as tf

from object_detection import model_lib_v2
from object_detection import model_lib
from object_detection import inputs

from server_api import ServerApi

MODEL_BUILD_UTIL_MAP = model_lib.MODEL_BUILD_UTIL_MAP
flags.DEFINE_string('namespace', None, 'Account namespace')
flags.DEFINE_string('model_version', None, 'model version id')
flags.DEFINE_string('host_url', None, 'Backend url')
flags.DEFINE_string('auth_token', None, 'Backend auth token')
flags.DEFINE_string('auth_header_key', None, 'Backend auth header key')
flags.DEFINE_string(
    'model_dir', None, 'Path to output model directory '
    'where event and checkpoint files will be written.')
flags.DEFINE_string('pipeline_config_path', None, 'Path to pipeline config '
                    'file.')
flags.DEFINE_float('score_increment', 0.05, 'Increment of minimum score threshold')
flags.DEFINE_string(
    'optimization', 'high-accuracy', 'model to use'
)
flags.DEFINE_string(
    'train_method', 'cloud', 'training method to use'
)
FLAGS = flags.FLAGS

def run_eval(model_dir, pipeline_config_path, optimization, min_score_threshold, train_method):
  def get_config_override(min_score_threshold):
    config_template = """
      model{{
        faster_rcnn {{
          second_stage_post_processing {{
            batch_non_max_suppression {{
              score_threshold: {score}
            }}
          }}
        }}
      }}
      eval_config {{
        min_score_threshold: {score}
      }}
    """
    if FLAGS.optimization == 'faster-prediction' or train_method == 'edge':
      config_template = """
        model{{
          ssd {{
            post_processing {{
              batch_non_max_suppression {{
                score_threshold: {score}
              }}
            }}
          }}
        }}
        eval_config {{
          min_score_threshold: {score}
        }}
      """
    config_override = config_template.format(score=min_score_threshold)
    return config_override

  def _run_eval(min_score_threshold):
    config_override = get_config_override(min_score_threshold)
    get_configs_from_pipeline_file = MODEL_BUILD_UTIL_MAP[
        'get_configs_from_pipeline_file']
    merge_external_params_with_configs = MODEL_BUILD_UTIL_MAP[
        'merge_external_params_with_configs']

    configs = get_configs_from_pipeline_file(
        pipeline_config_path, config_override=config_override)
    model_config = configs['model']
    train_input_config = configs['train_input_config']
    eval_config = configs['eval_config']
    eval_input_configs = configs['eval_input_configs']
    eval_on_train_input_config = copy.deepcopy(train_input_config)
    # eval_on_train_input_config.sample_1_of_n_examples = (
    #     sample_1_of_n_eval_on_train_examples)
    # if override_eval_num_epochs and eval_on_train_input_config.num_epochs != 1:
    #   tf.logging.warning('Expected number of evaluation epochs is 1, but '
    #                     'instead encountered `eval_on_train_input_config'
    #                     '.num_epochs` = '
    #                     '{}. Overwriting `num_epochs` to 1.'.format(
    #                         eval_on_train_input_config.num_epochs))
    #   eval_on_train_input_config.num_epochs = 1

    detection_model = MODEL_BUILD_UTIL_MAP['detection_model_fn_base'](
        model_config=model_config, is_training=True)

    # Create the inputs.
    eval_inputs = []
    for eval_input_config in eval_input_configs:
      next_eval_input = inputs.eval_input(
          eval_config=eval_config,
          eval_input_config=eval_input_config,
          model_config=model_config,
          model=detection_model)
      eval_inputs.append((eval_input_config.name, next_eval_input))

    # if eval_index is not None:
    #   eval_inputs = [eval_inputs[eval_index]]

    global_step = tf.compat.v2.Variable(
        0, trainable=False, dtype=tf.compat.v2.dtypes.int64)
    latest_checkpoint = tf.train.latest_checkpoint(model_dir)
    ckpt = tf.compat.v2.train.Checkpoint(
        step=global_step, model=detection_model)

    ckpt.restore(latest_checkpoint).expect_partial()

    eval_name, eval_input = eval_inputs[0]
    summary_writer = tf.compat.v2.summary.create_file_writer(
        os.path.join(model_dir, 'eval{}'.format(min_score_threshold), eval_name))
    with summary_writer.as_default():
      return model_lib_v2.eager_eval_loop(
          detection_model,
          configs,
          eval_input,
          use_tpu=False,
          global_step=global_step)

  return _run_eval(min_score_threshold)

def main(unused_argv):
  flags.mark_flag_as_required('model_dir')
  flags.mark_flag_as_required('pipeline_config_path')
  flags.mark_flag_as_required('namespace')
  flags.mark_flag_as_required('model_version')
  flags.mark_flag_as_required('host_url')
  flags.mark_flag_as_required('auth_token')
  flags.mark_flag_as_required('auth_header_key')
  # flags.mark_flag_as_required('train_method')
  inc = FLAGS.score_increment
  score_thresh = inc
  performance = []

  def normalize_float(val):
    # return val if val != 0.0 else 0
    return float(val)

  while score_thresh < 1:
    metrics = run_eval(
      FLAGS.model_dir,
      FLAGS.pipeline_config_path,
      FLAGS.optimization,
      score_thresh,
      FLAGS.train_method
    )
    print('=====METRICS: {}'.format(metrics))
    performance.append({
      'scoreThreshold': normalize_float(score_thresh),
      'precision': normalize_float(metrics['DetectionBoxes_Precision/mAP']),
      'recall': normalize_float(metrics['DetectionBoxes_Recall/AR@100']),
    })
    score_thresh = score_thresh + inc
  print('====performance: {}', performance)
  # server_api = ServerApi(FLAGS.host_url, FLAGS.auth_token, FLAGS.auth_header_key)
  # server_api.save_model_performance(FLAGS.namespace, FLAGS.model_version, performance)

if __name__ == '__main__':
  tf.compat.v1.app.run()

"""
python run_eval.py \
  --pipeline_config_path=gs://wizyvision-dev-automl/train_job/tf2_manual/train_pipeline/faster_rcnn_pipeline.config \
  --model_dir=gs://wizyvision-dev-automl/train_job/tf2_manual/gpu8_training_not_docker_multi_gpux8_v2_high_mem \
  --score_increment=0.4 \
  --host_url=https://auth.wizdam.xyz \
  --auth_token=cDOXnwraQRO220VgaEA2uRFrKUXymlFlAcUL4F \
  --auth_header_key=wizydam-dev-api-token \
  --model_version=9 \
  --namespace=carlo_test_wsli_3 \
  --optimization=high-accuracy
  --train_method=edge
"""
