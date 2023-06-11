import pandas as pd
from object_detection.protos.string_int_label_map_pb2 import StringIntLabelMap, StringIntLabelMapItem
from google.protobuf import text_format
import tensorflow as tf

def convert_classes(classes, start=1):
  msg = StringIntLabelMap()
  for index, row in classes.iterrows():
    msg.item.append(StringIntLabelMapItem(id=row['labelMapId'], name=row['name']))

  text = str(text_format.MessageToBytes(msg, as_utf8=True), 'utf-8')
  return text


def make_label_map(path, output_path):
  df = pd.read_csv(path)
  txt = convert_classes(df)
  with tf.io.gfile.GFile(output_path, "w") as f:
    f.write(txt)

  return df
