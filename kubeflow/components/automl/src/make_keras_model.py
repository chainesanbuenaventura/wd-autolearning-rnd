import argparse
import os

from google.cloud import storage

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--input-dir', dest='input_dir', required=True)
  parser.add_argument('--output-dir', dest='output_dir', required=True)
  parser.add_argument('--svcacct-name', dest='svcacct_name', required=True)
  known_args, _ = parser.parse_known_args()

  input_dir = known_args.input_dir
  unique_name = input_dir.split('/')[4]

  unique_folder = '/tmp/{}'.format(unique_name)
  os.system('mkdir {}'.format(unique_folder))
  model_json = '{}/saved_model.json'.format(unique_folder)
  open(model_json, 'w').close()
  weights_bin = '{}/model.weights.bin'.format(unique_folder)
  open(weights_bin, 'w').close()

  os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/app/{}'.format(known_args.svcacct_name)
  client = storage.Client()
  bucket = client.get_bucket(input_dir.split('/')[2])

  source_prefix = '/'.join(input_dir.split('/')[3:])
  blob_model_json = bucket.get_blob('{}/model.json'.format(source_prefix))
  blob_model_json.download_to_filename(model_json)
  blob_weights_bin = bucket.get_blob('{}/model.weights.bin'.format(source_prefix))
  blob_weights_bin.download_to_filename(weights_bin)

  os.system('tensorflowjs_converter --input_format=tfjs_layers_model --output_format=keras_saved_model --skip_op_check --strip_debug_ops=True {} {}'.format(model_json, known_args.output_dir))

  os.system("rm -rf {}".format(unique_folder))
