import argparse
import io
import logging
import os
import re
import csv
from io import StringIO

import tensorflow as tf
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import schema_utils
from tensorflow_transform.coders import example_proto_coder

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions

from PIL import Image

def run(params, pipeline_args):
  def create_raw_metadata():
    raw_data_schema = {
      'image/height': tf.io.FixedLenFeature([], tf.int64),
      'image/width': tf.io.FixedLenFeature([], tf.int64),
      'image/filename': tf.io.FixedLenFeature([], tf.string),
      'image/source_id': tf.io.FixedLenFeature([], tf.string),
      'image/encoded': tf.io.FixedLenFeature([], tf.string),
      'image/format': tf.io.FixedLenFeature([], tf.string),
      'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
      'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
      'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
      'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
      'image/object/class/text': tf.io.VarLenFeature(tf.string),
      'image/object/class/label': tf.io.VarLenFeature(tf.int64),
    }

        # create dataset_metadata given raw_schema
    raw_metadata = dataset_metadata.DatasetMetadata(
      schema_utils.schema_from_feature_spec(raw_data_schema))

    return raw_metadata

  def read_image(path):
    with tf.io.gfile.GFile(path, 'rb') as fid:
        encoded_jpg = fid.read()
    image = Image.open(io.BytesIO(encoded_jpg))
    height = image.height # Image height
    width = image.width # Image width
    image_format = image.format
    filename = path.split('/')[-1]
    return encoded_jpg, height, width, image_format, filename

  def make_record(row):
    result = {
      "path": row.get('path'),
      "xmins": [float(x) for x in row.get('xmins').split(';')] if row.get('xmins') is not None else [],
      "ymins": [float(x) for x in row.get('ymins').split(';')] if row.get('ymins') is not None else [],
      "xmaxs": [float(x) for x in row.get('xmaxs').split(';')] if row.get('xmaxs') is not None else [],
      "ymaxs": [float(x) for x in row.get('ymaxs').split(';')] if row.get('ymaxs') is not None else [],
      "labels": row.get('labels').split(';') if row.get('labels') is not None else [],
      "labelIds": [int(x) for x in row.get('labelIds').split(';')]
    }
    encoded_jpg, height, width, image_format, filename = read_image(result.get('path'))

    classes = result.get('labelIds')
    labels_bytes = [label.encode() for label in result.get('labels')]

    formatted = {
      'image/height': height,
      'image/width': width,
      'image/filename': filename.encode(),
      'image/source_id': filename.encode(),
      'image/encoded': encoded_jpg,
      'image/format': image_format.encode(),
      'image/object/bbox/xmin': result.get('xmins'),
      'image/object/bbox/xmax': result.get('xmaxs'),
      'image/object/bbox/ymin': result.get('ymins'),
      'image/object/bbox/ymax': result.get('ymaxs'),
      'image/object/class/text': labels_bytes,
      'image/object/class/label': classes,
    }
    return formatted

  def string_to_dict(string_input):
    values = list(csv.reader(StringIO(string_input)))[0]
    row = dict(
      zip(('path', 'name', 'labels', 'labelIds', 'xmins', 'xmaxs', 'ymins', 'ymaxs'), values))
    print(row)
    return row

  def read_csv(pipeline, dataset_dir, step):
    file_path = os.path.join(dataset_dir, '{}.csv'.format(step))
    raw_data = (
      pipeline
      | '{} - Read from csv'.format(step) >> beam.io.ReadFromText(file_path, skip_header_lines=1)
      | '{} - String to dict'.format(step) >> beam.Map(string_to_dict)
      | '{} - Make record item'.format(step) >> beam.Map(make_record)
    )
    return raw_data

  def write_tfrecords(transformed_data, location, step):
    return (
      transformed_data
      | '{} - Write Transformed Data'.format(step) >> beam.io.tfrecordio.WriteToTFRecord(
          file_path_prefix="{}".format(location),
          file_name_suffix=".tfrecords",
          coder=example_proto_coder.ExampleProtoCoder(create_raw_metadata().schema)
      )
    )
    ### START PIPELINE
  pipeline_options = PipelineOptions(flags=pipeline_args)

  p = beam.Pipeline(options=pipeline_options)

  output_paths = known_args.output_paths
  for step in ['test', 'train', 'val']:
    records = read_csv(p, params.dataset_dir, step)
    write_tfrecords(records, output_paths.get(step), step)

  result = p.run()
  result.wait_until_finish()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--output-dir', dest='output_dir', required=True)
  parser.add_argument('--project', dest='project', required=True)
  parser.add_argument('--dataset-dir', dest='dataset_dir', required=True)

  known_args, pipeline_args = parser.parse_known_args()
  pipeline_args.append('--project')
  pipeline_args.append(known_args.project)

  output_paths = {
    "train": os.path.join(known_args.output_dir, 'train'),
    "test": os.path.join(known_args.output_dir, 'test'),
    "val": os.path.join(known_args.output_dir, 'val'),
  }
  known_args.output_paths = output_paths
  run(known_args, pipeline_args)


  train_path = output_paths.get('train') + '-?????-of-?????.tfrecords'
  val_path = output_paths.get('val') + '-?????-of-?????.tfrecords'
  test_path = output_paths.get('test') + '-?????-of-?????.tfrecords'

  print("train_path: {}".format(train_path))
  print("val_path: {}".format(val_path))
  print("test_path: {}".format(test_path))

  with open("/tmp/train_path.txt", "w") as f:
    f.write(train_path)
  with open("/tmp/val_path.txt", "w") as f:
    f.write(val_path)
  with open("/tmp/test_path.txt", "w") as f:
    f.write(test_path)


  """
    python make_tfrecords.py \
      --project=wizydam-dev \
      --output-dir=gs://wizyvision-dev-automl/train_job/tf2_manual/records \
      --dataset-dir=gs://wizyvision-dev-automl/train_job/tf2_manual/dataset/split

    outputs:
      train_path: gs://wizyvision-dev-automl/train_job/tf2_manual/records/train-?????-of-?????.tfrecords
      val_path: gs://wizyvision-dev-automl/train_job/tf2_manual/records/val-?????-of-?????.tfrecords
      test_path: gs://wizyvision-dev-automl/train_job/tf2_manual/records/test-?????-of-?????.tfrecords
  """
