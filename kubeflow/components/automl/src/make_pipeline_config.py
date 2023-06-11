import argparse

import tensorflow as tf
from google.protobuf import text_format
from object_detection.protos import pipeline_pb2


def parse_arguments():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('pipeline')
    parser.add_argument('output')
    return parser.parse_args()


def write(config, output_path):
  pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
  pipeline_template = config['pipeline_template']

  with tf.io.gfile.GFile(pipeline_template, 'r') as f:
    proto_str = f.read()
    text_format.Merge(proto_str, pipeline_config)

  pipeline_config.train_config.num_steps = config['num_steps']
  pipeline_config.train_config.batch_size = config['batch_size']

  if not config['is_retrain']:
    model_name = config['model_name']

    if model_name == 'ssd':
      pipeline_config.model.ssd.num_classes = config['num_classes']
    elif (model_name == 'ssd_mobilenet' and config['optimization'] != 'high-accuracy'):
      pipeline_config.model.ssd.num_classes = config['num_classes']
      pipeline_config.graph_rewriter.quantization.delay = round(config['num_steps'] / 2)
      pipeline_config.graph_rewriter.quantization.weight_bits = 8
      pipeline_config.graph_rewriter.quantization.activation_bits = 8

    elif model_name == 'ssd_mobilenet' and config['optimization'] == 'high-accuracy':
      pass

    else:
      pipeline_config.model.faster_rcnn.num_classes = config['num_classes']

    pipeline_config.train_config.fine_tune_checkpoint = config['fine_tune_checkpoint']
    pipeline_config.train_input_reader.label_map_path = config['label_map_path']
    pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [config['train_path']]

    pipeline_config.eval_config.num_examples = config['eval_count']
    pipeline_config.eval_config.num_visualizations = 1

    pipeline_config.eval_input_reader[0].label_map_path = config['label_map_path']
    pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [config['eval_path']]

  config_text = text_format.MessageToString(pipeline_config)

  with tf.io.gfile.GFile(output_path, "wb") as f:
    f.write(config_text)

if __name__ == '__main__':
  main()
