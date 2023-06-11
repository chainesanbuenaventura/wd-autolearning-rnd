import argparse
from google.cloud import storage
import pandas as pd
import sklearn.model_selection as model_selection
import make_pipeline_config
import tensorflow as tf

client = storage.Client()

def copy_ckpt(model_dir, output_dir):
  source_bucket = client.bucket(model_dir.split('/')[2])
  output_bucket = client.bucket(output_dir.split('/')[2])
  source_prefix = model_dir.replace('gs://{}/'.format(source_bucket.name), '')
  output_prefix = output_dir.replace('gs://{}/'.format(output_bucket.name), '')
  blobs = client.list_blobs(source_bucket, prefix=source_prefix)
  for blob in blobs:
    if "/eval" not in blob.name:
      new_name = blob.name.replace(source_prefix, output_prefix)
      source_bucket.copy_blob(blob, output_bucket, new_name=new_name)

def main(kwargs):
  optimization = kwargs.optimization
  output_dir = kwargs.output_dir
  pipeline_config_path = kwargs.pipeline_config_path

  prev_model_dir = kwargs.prev_model_dir
  new_model_dir = output_dir + '/model'

  config_output = 'ssd_inception_v2.config' if optimization == 'faster-prediction' else 'faster_rcnn_pipeline.config'
  if kwargs.train_method == 'edge':
    config_output = 'ssd_mobilenet_v2_fpnlite'

  pipeline_config_output = "{}/pipeline/{}".format(output_dir, config_output)

  latest_checkpoint = tf.train.latest_checkpoint(prev_model_dir)
  last_step = int(latest_checkpoint.split('ckpt-')[-1])
  num_steps = last_step + kwargs.num_steps

  copy_ckpt(prev_model_dir, new_model_dir)


  make_pipeline_config.write({
    'num_steps': num_steps,
    'is_retrain': True,
    'pipeline_template': pipeline_config_path,
    'batch_size': kwargs.batch_size,
    'optimization': kwargs.optimization
  }, pipeline_config_output)
  print(pipeline_config_output)

  return new_model_dir, pipeline_config_output

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--output-dir', dest='output_dir', required=True)
  parser.add_argument('--pipeline-config-path', dest='pipeline_config_path', required=True)
  parser.add_argument('--prev-model-dir', dest='prev_model_dir', required=True)
  parser.add_argument('--num-steps', dest='num_steps', required=True, type=int)
  parser.add_argument('--optimization', dest='optimization', required=True)
  parser.add_argument('--train-method', dest='train_method', required=True)
  parser.add_argument('--batch-size', dest='batch_size', required=True, type=int)

  kwargs, _ = parser.parse_known_args()
  new_model_dir, pipeline_config_output = main(kwargs)

  with open("/tmp/pipeline_config_path.txt", "w") as f:
    f.write(pipeline_config_output)
  with open("/tmp/model_dir.txt", "w") as f:
    f.write(new_model_dir)

"""
  python prep_retrain_artifacts.py \
    --output-dir=gs://wd-model-rnd-0000/train_job/retrain_aerialmodeltest__m1__v5 \
    --pipeline-config-path=gs://wd-model-rnd-0000/train_job/aerialmodeltest__m1__v5/pipeline/faster_rcnn_pipeline.config \
    --prev-model-dir=gs://wd-model-rnd-0000/train_job/aerialmodeltest__m1__v5/model \
    --num-steps=332 \
    --optimization=high-accuracy \
    --batch-size=8 \
    --train-method=edge
"""
