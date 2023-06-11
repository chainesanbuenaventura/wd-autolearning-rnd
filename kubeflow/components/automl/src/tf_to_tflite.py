import argparse
import os
import tensorflow as tf
import numpy as np

# from google.cloud import storage
from smart_open import open


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--pipeline-config-path', dest='pipeline_config_path', required=True)
  parser.add_argument('--trained-checkpoint-prefix', dest='trained_checkpoint_prefix', required=True)
  parser.add_argument('--graph-def-path', dest='graph_def_path', required=True)
  parser.add_argument('--output-file', dest='output_file', required=True)
  parser.add_argument('--optimization', dest='optimization', required=True)
  parser.add_argument('--train-method', dest='train_method', default='cloud')
  # parser.add_argument('--svcacct-name', dest='svcacct_name', required=True)
  known_args, _ = parser.parse_known_args()

  pipeline_config_path = known_args.pipeline_config_path
  trained_checkpoint_prefix = known_args.trained_checkpoint_prefix
  output_file = known_args.output_file
  graph_def_path = known_args.graph_def_path

  if known_args.optimization == 'faster-prediction' or known_args.optimization == 'balanced':
    cmd = "python -m object_detection.export_tflite_graph_tf2 " \
      "--pipeline_config_path {pipeline_config_path} " \
      "--trained_checkpoint_dir {trained_checkpoint_prefix} " \
      "--output_directory {graph_def_path}".format(
        pipeline_config_path=pipeline_config_path,
        trained_checkpoint_prefix=trained_checkpoint_prefix,
        graph_def_path=graph_def_path,
      )
    print(cmd)
    os.system(cmd)

    graph_def_file = graph_def_path + '/tflite_graph.pb'

    if (known_args.train_method == 'edge'):
      converter = tf.lite.TFLiteConverter.from_saved_model(f"{graph_def_path}/saved_model")
      converter.enable_v1_converter = True
      converter.graph_def_file = f'{graph_def_path}/saved_model.pb'
      converter.allow_custom_ops = True
      converter.enable_v1_converter = True
      converter.input_shapes=1,300,300,3
      converter.output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3'
      converter.mean_values=128
      converter.std_dev_values=128
      converter.change_concat_input_ranges=False
      converter.post_training_quantize = True
      # converter.output_file = output_file

      if (known_args.optimization == 'faster-prediction'):
        converter.inference_type='QUANTIZED_UINT8'
      else:
        converter.quantize_to_float16 = True
        # converter.target_spec.supported_types = [tf.float16]
      tflite_quant_model = converter.convert()
      with tf.io.gfile.GFile(output_file, "wb") as f:
        f.write(tflite_quant_model)
    else:
      convert_cmd = "!tflite_convert --saved_model_dir={graph_def_path}/saved_model --output_file={output_file}".format(
        graph_def_path=graph_def_path,
        output_file=output_file
      )
    # converter.allow_custom_ops = True
    # # converter.inference_type = "float32"
    # tflite_model = converter.convert()

    # with open(output_file, 'wb') as fout:
    #   fout.write(tflite_model)
    print(output_file)
    with open('/tmp/tflite_path.txt', 'w') as f:
      f.write(output_file)
  else:
    with open('/tmp/tflite_path.txt', 'w') as f:
      f.write('')

"""
  python tf_to_tflite.py \
      --pipeline-config-path
      --trained-checkpoint-prefix
      --graph-def-path
      --output-file
      --optimization=faster-prediction
"""
