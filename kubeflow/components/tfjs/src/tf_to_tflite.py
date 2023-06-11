import argparse
import os
import tensorflow as tf

from google.cloud import storage
from smart_open import open

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--input-dir', dest='input_dir', required=True)
  parser.add_argument('--output-file', dest='output_file', required=True)
  parser.add_argument('--svcacct-name', dest='svcacct_name', required=True)
  known_args, _ = parser.parse_known_args()

  input_dir = known_args.input_dir
  output_file = known_args.output_file

  os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/app/{}'.format(known_args.svcacct_name)

  converter = tf.lite.TFLiteConverter.from_saved_model(input_dir) # path to the SavedModel directory
  tflite_model = converter.convert()

  with open(output_file, 'wb') as fout:
    fout.write(tflite_model)

