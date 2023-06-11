import argparse
import os

# import eval_reader
import tensorflow as tf
from google.protobuf import text_format
from object_detection import exporter_lib_v2
from object_detection.protos import pipeline_pb2

faster_rcnn_config_override =  " \
            model{ \
              faster_rcnn { \
                second_stage_post_processing { \
                  batch_non_max_suppression { \
                    max_total_detections: 100 \
                    max_detections_per_class: 100 \
                  } \
                } \
              } \
            }"
ssd_config_override = " \
            model{ \
              ssd { \
                post_processing { \
                  batch_non_max_suppression { \
                    max_total_detections: 100 \
                    max_detections_per_class: 100 \
                  } \
                } \
              } \
            }"

# def find_nearest_ckpt_at_step(step, checkpoint_dir):
#   checkpoints = tf.train.get_checkpoint_state(checkpoint_dir).all_model_checkpoint_paths
#   nearest = None
#   paths = list(checkpoints)
#   paths.reverse()
#   for path in paths:
#     if nearest == None:
#       ckpt_step = int(path.split('ckpt-')[-1])
#       if ckpt_step <= step:
#         print(ckpt_step, step)
#         nearest = path
#   return nearest

# def find_best_ckpt(checkpoint_dir):
#   eval_metrics = eval_reader.read_eval_metrics(os.path.join(checkpoint_dir, 'eval_0'))
#   best_step, best_val = eval_reader.find_best(eval_metrics, 'DetectionBoxes_Precision/mAP')
#   best_ckpt = find_nearest_ckpt_at_step(best_step, checkpoint_dir)
#   print('best_ckpt:::{}'.format(best_ckpt))
#   return best_ckpt

def export_checkpoint(params):
  config_override = faster_rcnn_config_override
  if params.optimization == 'faster-prediction' or params.train_method == 'edge':
    config_override = ssd_config_override

  # best_checkpoint = find_best_ckpt(params.checkpoint_dir)

  pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
  with tf.io.gfile.GFile(params.pipeline_config_path, 'r') as f:
    text_format.Merge(f.read(), pipeline_config)
  text_format.Merge(config_override, pipeline_config)

  input_shape = None
  input_type = params.input_type
  # exporter_lib_v2.export_inference_graph(
  #     FLAGS.input_type, pipeline_config, FLAGS.trained_checkpoint_dir,
  #     FLAGS.output_directory, FLAGS.use_side_inputs, FLAGS.side_input_shapes,
  #     FLAGS.side_input_types, FLAGS.side_input_names)
  exporter_lib_v2.export_inference_graph(
      input_type, pipeline_config, params.checkpoint_dir,
      params.output_dir)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--checkpoint-dir', dest='checkpoint_dir', required=True)
  parser.add_argument('--output-dir', dest='output_dir', required=True)
  parser.add_argument('--pipeline-config-path', dest='pipeline_config_path', required=True)
  parser.add_argument('--optimization', dest='optimization', default='high-accuracy')
  parser.add_argument('--train-method', dest='train_method', default='cloud')
  parser.add_argument('--input-type', dest='input_type')
  known_args, _ = parser.parse_known_args()

  export_checkpoint(known_args)

  saved_model_path = os.path.join(known_args.output_dir, 'saved_model')
  print('saved_model_path:{}'.format(saved_model_path))

  with open("/tmp/saved_model_path.txt", 'w') as f:
    f.write(saved_model_path)
#   pipeline_config_path

"""
  python export_model.py \
    --pipeline-config-path=gs://wizyvision-dev-automl/train_job/tf2_manual/train_pipeline/faster_rcnn_pipeline.config \
    --checkpoint-dir=gs://wizyvision-dev-automl/train_job/tf2_manual/gpu8_training_not_docker_multi_gpux8_v2_high_mem/ \
    --output-dir=gs://wizyvision-dev-automl/train_job/tf2_manual/export_manual/
    --input-type=image_tensor
    --train-method=edge
"""
