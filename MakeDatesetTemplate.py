import io
import logging
import os

import tensorflow as tf
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import schema_utils
from tensorflow_transform.coders import example_proto_coder

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions, StaticValueProvider, SetupOptions;
from apache_beam.runners.dataflow.native_io import iobase as dataflow_io
from apache_beam.io.gcp import bigquery_tools

from PIL import Image

class MakeDatasetOptions(PipelineOptions):
  @classmethod
  def _add_argparse_args(cls, parser):
    # Use add_value_provider_argument for arguments to be templatable
    # Use add_argument as usual for non-templatable arguments
    parser.add_value_provider_argument(
        '--bq_dataset',
        default='autolearningRND.bottlestTest2',
        help='dataset.table where to read the dataset')

    parser.add_value_provider_argument(
        '--output_dir',
        # default='example_output',
        default='gs://wd-model-rnd-0000/test_bottles/records',
        # gs://wd-model-rnd-0000/cloud/test_bottles/records
        help='Cloud storage path to write the tf-records')
    parser.add_value_provider_argument(
        '--temp_dir',
        default='gs://wd-model-rnd-0000/test_bottles/tmp',
        # default='example_output/tmp',
        help='Cloud storage path to write the temporary files')

# class BQSourceQueryBuilder(beam.io.BigQuerySource):
#   def __init__(self, dataset_table, step):
#     self.dataset_table = dataset_table
#     self.step = step
#     self.validate = False
#     self.flatten_results = True
#     self.coder = bigquery_tools.RowAsDictJsonCoder()
#     self.kms_key = None
#     self.use_legacy_sql = False
#     self.table_reference = None
#     self.query = self.get_source_query('autolearningRND.bottlestTest2', 'train')

#   def get_source_query(self, dataset_table, step):
#     query = """
#       SELECT
#         path,
#         type,
#         STRING_AGG(CAST(label as STRING), ';') as labels,
#         STRING_AGG(CAST(xmin as STRING), ';') as xmins,
#         STRING_AGG(CAST(ymin as STRING), ';') as ymins,
#         STRING_AGG(CAST(xmax as STRING), ';') as xmaxs,
#         STRING_AGG(CAST(ymax as STRING), ';') as ymaxs,
#       FROM `{}`
#       GROUP BY path,type
#     """.format(dataset_table)

#     source_query = 'SELECT * FROM ({}) WHERE type = "{}"'.format(query, step.upper())
#     return source_query

  # def reader(self, test_bigquery_client=None):
  #   source_query = self.get_source_query(self.dataset_table.get(), self.step)
  #   self.query = source_query
  #   return bigquery_tools.BigQueryReader(
  #       source=self,
  #       test_bigquery_client=test_bigquery_client,
  #       use_legacy_sql=self.use_legacy_sql,
  #       flatten_results=self.flatten_results,
        # kms_key=self.kms_key)

def run():
  labelMap = {
    'unknown_bottle': 1,
    'volvic': 2,
    'sports_cap': 3,
    'fiji': 4,
    'wilkins': 5,
    'summit': 6,
    'evian': 7,
    'nature_spring': 8,
    'red_bull': 9,
    'vittel': 10,
    'st_george': 11,
  }

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
  def get_source_query(dataset_table, step):
    query = """
      SELECT
        path,
        type,
        STRING_AGG(CAST(label as STRING), ';') as labels,
        STRING_AGG(CAST(xmin as STRING), ';') as xmins,
        STRING_AGG(CAST(ymin as STRING), ';') as ymins,
        STRING_AGG(CAST(xmax as STRING), ';') as xmaxs,
        STRING_AGG(CAST(ymax as STRING), ';') as ymaxs,
      FROM `{}`
      GROUP BY path,type
    """.format(dataset_table)

    source_query = 'SELECT * FROM ({}) WHERE type = "{}"'.format(query, step.upper())
    return source_query

  def read_image(path):
    with tf.io.gfile.GFile(path, 'rb') as fid:
        encoded_jpg = fid.read()
    image = Image.open(io.BytesIO(encoded_jpg))
    height = image.height # Image height
    width = image.width # Image width
    image_format = image.format
    filename = path.split('/')[-1]
    return encoded_jpg, height, width, image_format, filename

  def prep_bq_row(bq_row):
    result = {
      "path": bq_row.get('path'),
      "xmins": [float(x) for x in bq_row.get('xmins').split(';')] if bq_row.get('xmins') is not None else [],
      "ymins": [float(x) for x in bq_row.get('ymins').split(';')] if bq_row.get('ymins') is not None else [],
      "xmaxs": [float(x) for x in bq_row.get('xmaxs').split(';')] if bq_row.get('xmaxs') is not None else [],
      "ymaxs": [float(x) for x in bq_row.get('ymaxs').split(';')] if bq_row.get('ymaxs') is not None else [],
      "labels": bq_row.get('labels').split(';') if bq_row.get('labels') is not None else []
    }
    print('===bq:::{}'.format(result))
    encoded_jpg, height, width, image_format, filename = read_image(result.get('path'))

    classes = [labelMap.get(label) for label in result.get('labels')]
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

  def read_from_bq(pipeline, bq_dataset, step):
    source_query = get_source_query(bq_dataset, step)
    raw_data = (
      pipeline
      | '{} - Read Data from BigQuery'.format(step) >> beam.io.Read(
          beam.io.BigQuerySource(query=source_query, use_standard_sql=True))
      | '{} - Format Data'.format(step) >> beam.Map(prep_bq_row)
    )
    return raw_data

  def write_tfrecords(transformed_data, location, step):
    # # transformed_data, transformed_metadata = transformed_dataset
    return (
      transformed_data
      | '{} - Write Transformed Data'.format(step) >> beam.io.tfrecordio.WriteToTFRecord(
          file_path_prefix="{}/{}".format(location, step),
          file_name_suffix=".tfrecords",
          coder=example_proto_coder.ExampleProtoCoder(create_raw_metadata().schema)
      )
    )
    ### START PIPELINE
  pipeline_options = PipelineOptions()
  pipeline_options.view_as(SetupOptions).save_main_session = True

  p = beam.Pipeline(options=pipeline_options)

  # dataset_options = pipeline_options.view_as(MakeDatasetOptions)

  # output_dir = dataset_options.output_dir
  # bq_dataset = dataset_options.bq_dataset

  # step = 'train'
  # transformed_train_data = read_from_bq(p, bq_dataset, step)
  # write_tfrecords(transformed_train_data, output_dir, step)

  for step in ['train', 'validation', 'test']:
    transformed_train_data = read_from_bq(p, 'autolearningRND.bottlestTest2', step)
    write_tfrecords(transformed_train_data, 'gs://wd-model-rnd-0000/cloud/test_bottles2/records', step)

  result = p.run()
  result.wait_until_finish()

if __name__ == '__main__':
  logger = logging.getLogger().setLevel(logging.INFO)
  run()
