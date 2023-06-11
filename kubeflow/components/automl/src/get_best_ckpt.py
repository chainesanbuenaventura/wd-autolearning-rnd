import argparse
import os

import eval_reader
import tensorflow as tf

def find_nearest_ckpt_at_step(step, checkpoint_dir):
  checkpoints = tf.train.get_checkpoint_state(checkpoint_dir).all_model_checkpoint_paths
  paths = list(checkpoints)
  paths.reverse()
  nearest = paths[0]
  for path in paths:
    if nearest == None:
      ckpt_step = int(path.split('ckpt-')[-1])
      if ckpt_step <= step:
        print(ckpt_step, step)
        nearest = path
  return nearest

def find_best_ckpt(checkpoint_dir):
  eval_metrics = eval_reader.read_eval_metrics(os.path.join(checkpoint_dir, 'eval_0'))
  best_step, best_val = eval_reader.find_best(eval_metrics, 'DetectionBoxes_Precision/mAP')
  best_ckpt = find_nearest_ckpt_at_step(best_step, checkpoint_dir)
  print('best_ckpt:::{}'.format(best_ckpt))
  return best_ckpt

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--checkpoint-dir', dest='checkpoint_dir', required=True)
  known_args, _ = parser.parse_known_args()

  # best_checkpoint = find_best_ckpt(known_args.checkpoint_dir)
  best_checkpoint = known_args.checkpoint_dir

  print('best_checkpoint:{}'.format(best_checkpoint))

  with open('/best_checkpoint.txt', 'w') as f:
    f.write(best_checkpoint)

"""
  python get_best_ckpt.py \
    --pipeline-config-path=gs://wizyvision-dev-automl/train_job/aerialmodeltest_1_40_manual_5/pipeline/faster_rcnn_pipeline.config \
    --checkpoint-dir=gs://wizyvision-dev-automl/train_job/aerialmodeltest_1_40_manual_5/model \
    --output-dir=gs://wizyvision-dev-automl/train_job/aerialmodeltest_1_40_manual_5/export

"""
