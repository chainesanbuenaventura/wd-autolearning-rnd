import io
import os

import tensorflow as tf
import apache_beam as beam
import tensorflow_transform as tft

from tensorflow_transform.beam import impl
from tensorflow_transform.tf_metadata import dataset_metadata
from tensorflow_transform.tf_metadata import schema_utils
from tensorflow_transform.coders import example_proto_coder
from tensorflow_transform.tf_metadata import metadata_io
from tensorflow_transform.beam.tft_beam_io import transform_fn_io
from PIL import Image

PROJECT = 'wizydam-dev' # change to your proect_Id
BUCKET = 'wd-model-rnd-0000' # change to your bucket name
REGION = 'europe-west1' # change to your region
ROOT_DIR = 'babyweight_tft' # directory where the output is stored locally or on GCS

RUN_LOCAL = True # if True, the DirectRunner is used, else DataflowRunner
DATA_SIZE = 100 #

os.environ['PROJECT'] = PROJECT
os.environ['BUCKET'] = BUCKET
os.environ['REGION'] = REGION
os.environ['ROOT_DIR'] = ROOT_DIR
os.environ['RUN_LOCAL'] = str(RUN_LOCAL)

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

def get_source_query(step):

  query = """
  SELECT
    path,
    type,
    STRING_AGG(CAST(label as STRING), ';') as labels,
    STRING_AGG(CAST(xmin as STRING), ';') as xmins,
    STRING_AGG(CAST(ymin as STRING), ';') as ymins,
    STRING_AGG(CAST(xmax as STRING), ';') as xmaxs,
    STRING_AGG(CAST(ymax as STRING), ';') as ymaxs,
  FROM `autolearningRND.bottlestTest2`
  GROUP BY path,type
  """

  source_query = 'SELECT * FROM ({}) WHERE type = "{}"'.format(query, step.upper())
  print(source_query)
  return source_query

def prep_bq_row(bq_row):
  result = {
    "path": bq_row.get('path'),
    "xmins": [float(x) for x in bq_row.get('xmins').split(';')] if bq_row.get('xmins') is not None else [],
    "ymins": [float(x) for x in bq_row.get('ymins').split(';')] if bq_row.get('ymins') is not None else [],
    "xmaxs": [float(x) for x in bq_row.get('xmaxs').split(';')] if bq_row.get('xmaxs') is not None else [],
    "ymaxs": [float(x) for x in bq_row.get('ymaxs').split(';')] if bq_row.get('ymaxs') is not None else [],
    "labels": bq_row.get('labels').split(';') if bq_row.get('labels') is not None else []
  }
  with tf.io.gfile.GFile(result.get('path'), 'rb') as fid:
    encoded_jpg = fid.read()

  image = Image.open(io.BytesIO(encoded_jpg))
  height = image.height # Image height
  width = image.width # Image width
  image_format = image.format
  filename = result.get('path').split('/')[-1]

  classes = [labelMap.get(label) for label in result.get('labels')]
  labels_bytes = [label.encode() for label in result.get('labels')]

  for i, label_id in enumerate(classes):
    if label_id is None:
        print('====label id is none')
        print(result.get('labels')[i])


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

def read_from_bq(pipeline, step):
  source_query = get_source_query(step)
  raw_data = (
    pipeline
    | '{} - Read Data from BigQuery'.format(step) >> beam.io.Read(
        beam.io.BigQuerySource(query=source_query, use_standard_sql=True))
    | '{} - Clean up Data'.format(step) >> beam.Map(prep_bq_row)
  )
  return raw_data

def write_tfrecords(transformed_data, location, step):
  # # transformed_data, transformed_metadata = transformed_dataset
  (
    transformed_data
    | '{} - Write Transformed Data'.format(step) >> beam.io.tfrecordio.WriteToTFRecord(
        file_path_prefix=os.path.join(location,'{}'.format(step)),
        file_name_suffix=".tfrecords",
        coder=example_proto_coder.ExampleProtoCoder(create_raw_metadata().schema)
    )
  )

def run_transformation_pipeline(args):
  pipeline_options = beam.pipeline.PipelineOptions(flags=[], **args)

  temporary_dir = args['temp_dir']
  transformed_data_location = args['transformed_data_location']

  with beam.Pipeline('DirectRunner', options=pipeline_options) as pipeline:
    with impl.Context(temporary_dir):
      # Preprocess train data
      # step = 'test'
      # # Read raw train data from BQ
      # raw_test_dataset = read_from_bq(pipeline, step)
      # print('====raw_test_dataset')
      # write_tfrecords(raw_test_dataset, transformed_data_location, step)

      # step = 'eval'
      # raw_eval_dataset = read_from_bq(pipeline, step)
      # print('====raw_eval_dataset')
      # write_tfrecords(raw_eval_dataset, transformed_data_location, step)

      step = 'train'
      raw_train_dataset = read_from_bq(pipeline, step)
      print('====raw_train_dataset')
      write_tfrecords(raw_train_dataset, transformed_data_location, step)

OUTPUT_DIR = os.path.join('example_output')
TRANSFORMED_DATA_DIR = os.path.join(OUTPUT_DIR,'transformed')
TEMP_DIR = os.path.join('example_output', 'tmp')

run_transformation_pipeline({
  'transformed_data_location':  TRANSFORMED_DATA_DIR,
  'temp_dir': TEMP_DIR,

  'project': PROJECT,
  'region': REGION,
  'staging_location': os.path.join(OUTPUT_DIR, 'staging'),
  'temp_location': os.path.join(OUTPUT_DIR, 'tmp'),
  'worker_machine_type': 'n1-standard-1',
  'requirements_file': 'requirements.txt',
})


try:
    tf.gfile.DeleteRecursively(TRANSFORMED_DATA_DIR)
    # tf.gfile.DeleteRecursively(TRANSFORM_ARTEFACTS_DIR)
    tf.gfile.DeleteRecursively(TEMP_DIR)
    print('previous transformation files deleted!')
except:
    pass
