import tensorflow as tf
from google.protobuf.json_format import MessageToJson
counter = 0
for example in tf.compat.v1.python_io.tf_record_iterator("RuntimeValueProvider(option: output_dir, type: str, default_value: 'example_output')/train-00000-of-00001.tfrecords"):
  if counter == 0:
    jsonMessage = MessageToJson(tf.train.Example.FromString(example))
    print(jsonMessage)
  counter = counter + 1
