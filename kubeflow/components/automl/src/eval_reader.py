import collections
import os

import tensorflow as tf
from tensorflow.io import gfile

_EVENT_FILE_GLOB_PATTERN = 'events.out.tfevents.*'

def get_summaries(eval_dir):
  """Yields `tensorflow.Event` protos from event files in the eval dir.

  Args:
    eval_dir: Directory containing summary files with eval metrics.

  Yields:
    `tensorflow.Event` object read from the event files.
  """
  if gfile.exists(eval_dir):
    for event_file in gfile.glob(
        os.path.join(eval_dir, _EVENT_FILE_GLOB_PATTERN)):
      for event in tf.train.summary_iterator(event_file):
        yield event



def read_eval_metrics(eval_dir):
  """Helper to read eval metrics from eval summary files.

  Args:
    eval_dir: Directory containing summary files with eval metrics.

  Returns:
    A `dict` with global steps mapping to `dict` of metric names and values.
  """
  eval_metrics_dict = collections.defaultdict(dict)
  for event in get_summaries(eval_dir):
    # print('=====event:{}'.format(event))
    if not event.HasField('summary'):
      continue
    metrics = {}
    for value in event.summary.value:
      if value.HasField('simple_value'):
        metrics[value.tag] = value.simple_value
    if metrics:
      eval_metrics_dict[event.step].update(metrics)
  return collections.OrderedDict(
      sorted(eval_metrics_dict.items(), key=lambda t: t[0]))

def find_best(eval_metrics, metric_name):
  best_val = None
  best_step = None
  for step, metrics in eval_metrics.items():
    val = metrics[metric_name]
    if best_val is None or best_val < val:
      best_val = val
      best_step = step
  return best_step, best_val

