import argparse
import os

import make_label_map as label_map_util
import make_pipeline_config
import prep_retrain_artifacts
import tensorflow as tf

def is_reuse_prev_version(params):
  return params.prev_model_dir != 'None'

def main(params):
  label_map_path = os.path.join(params.output_dir, 'label_map.pbtxt')
  label_map = label_map_util.make_label_map(params.label_map_path, label_map_path)
  num_classes = len(label_map.index)
  num_steps = params.num_steps
  if is_reuse_prev_version(params):
    latest_checkpoint = tf.train.latest_checkpoint(params.prev_model_dir)
    last_step = int(latest_checkpoint.split('ckpt-')[-1])
    num_steps = last_step + params.num_steps
    prep_retrain_artifacts.copy_ckpt(params.prev_model_dir, params.model_dir)

  make_pipeline_config.write({
    'num_classes': num_classes,
    'fine_tune_checkpoint': params.fine_tune_checkpoint,
    'num_steps': num_steps,
    'label_map_path': label_map_path,
    'train_path': params.train_path,
    'eval_count': int(params.eval_count),
    'eval_path': params.eval_path,
    'pipeline_template': params.pipeline_template,
    'model_name': params.model_name,
    'is_retrain': False,
    'batch_size': params.batch_size,
    'optimization': params.optimization

  }, params.pipeline_config_path)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--label-map-path', dest='label_map_path', required=True)
  parser.add_argument('--output-dir', dest='output_dir', required=True)
  parser.add_argument('--train-path', dest='train_path', required=True)
  parser.add_argument('--eval-count', dest='eval_count', required=True)
  parser.add_argument('--eval-path', dest='eval_path', required=True)
  parser.add_argument('--num-steps', dest='num_steps', default=20000, type=int)
  parser.add_argument('--optimization', dest='optimization', default='high-accuracy')
  parser.add_argument('--train-method', dest='train_method', default='cloud')
  parser.add_argument('--train-bucket', dest='train_bucket', required=True)
  parser.add_argument('--model-dir', dest='model_dir')
  parser.add_argument('--prev-model-dir', dest='prev_model_dir', default="None")
  parser.add_argument('--pipeline-config-path', dest='pipeline_config_path', default="None")
  parser.add_argument('--batch-size', dest='batch_size', default="None", type=int)

  kwargs, _ = parser.parse_known_args()

  config_output = 'faster_rcnn_pipeline.config'
  pipeline_template = 'gs://{}/static/pipeline/fine_tune_checkpoints/tensorflow2/faster_rcnn_inception_resnet_v2_640x640_coco17_tpu-8/pipeline.config'.format(kwargs.train_bucket)
  fine_tune_checkpoint = 'gs://{}/static/pipeline/fine_tune_checkpoints/tensorflow2/faster_rcnn_inception_resnet_v2_640x640_coco17_tpu-8/checkpoint/ckpt-0'.format(kwargs.train_bucket)
  model_name = 'faster_rcnn'
  if kwargs.train_method == 'edge':
    config_output = 'ssd_mobilenet_v2_fpnlite'
    pipeline_template = 'gs://{}/static/pipeline/templates/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8.config'.format(kwargs.train_bucket)
    fine_tune_checkpoint = 'gs://{}/static/pipeline/fine_tune_checkpoints/tensorflow2/ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8/checkpoint/ckpt-0'.format(kwargs.train_bucket)
    model_name = 'ssd_mobilenet'
  elif kwargs.optimization == 'faster-prediction':
    config_output = 'ssd_resnet101_v1_fpn.config'
    pipeline_template = 'gs://{}/static/pipeline/templates/tensorflow2/ssd_resnet101_v1_fpn_640x640_coco17_tpu-8.config'.format(kwargs.train_bucket)
    fine_tune_checkpoint = 'gs://{}/static/pipeline/fine_tune_checkpoints/tensorflow2/ssd_resnet101_v1_fpn_640x640_coco17_tpu-8/checkpoint/ckpt-0'.format(kwargs.train_bucket)
    model_name = 'ssd'

  # use pipeline_template from params if will need to reuse the previous checkpoint
  if is_reuse_prev_version(kwargs):
    pipeline_template = kwargs.pipeline_config_path

  kwargs.fine_tune_checkpoint = fine_tune_checkpoint
  kwargs.pipeline_template = pipeline_template
  kwargs.model_name = model_name


  kwargs.pipeline_config_path = os.path.join(kwargs.output_dir, config_output)
  main(kwargs)

  print('pipeline_config_path: {}'.format(kwargs.pipeline_config_path))

  with open("/tmp/pipeline_config_path.txt", 'w') as f:
    f.write(kwargs.pipeline_config_path)


  """
    python prep_artifacts.py \
      --output-dir=gs://wizyvision-dev-automl/train_job/tf2_manual/train_pipeline \
      --label-map-path=gs://wizyvision-dev-automl/train_job/tf2_manual/dataset/tag_map.csv \
      --train-path=gs://wizyvision-dev-automl/train_job/tf2_manual/records/train-?????-of-?????.tfrecords \
      --eval-path=gs://wizyvision-dev-automl/train_job/tf2_manual/records/val-?????-of-?????.tfrecords \
      --eval-count=10 \
      --num-steps=100 \
      --train-bucket=wizyvision-dev-automl \
      --batch-size=8 \
      --train-method=edge

    outputs:
      pipeline_config_path
  """
