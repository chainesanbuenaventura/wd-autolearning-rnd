import argparse

import tensorflow.keras
import tensorflow as tf

KERAS_MODEL = ''

# decode image bytes to image_tensor
# then resize the image to [224, 224] to fit it on the mobilenetV2 model
def preproc_image(image_bytes):
  image_tensor = tf.image.decode_jpeg(image_bytes,
                                       channels=3)
  image_bytes = tf.image.resize(
    image_tensor, [224, 224], method=tf.image.ResizeMethod.BILINEAR, preserve_aspect_ratio=False,
    antialias=False, name=None
  )
  normalized = (image_bytes / 127) - 1
  return normalized

# this will override the input and output signature of the model
@tf.function()
def my_module_encoder(image_bytes):
  global KERAS_MODEL

  inputs = {
    # the key will be the name of the input when doing prediction in AI Platform
    'image_bytes': image_bytes,
  }
  # on AI Platform, they support batch size
  # to support it on our model, we use tf.map_fn to map on the instances
  resized = tf.map_fn(
    preproc_image,
    elems=inputs['image_bytes'],
    dtype=tf.float32,
    parallel_iterations=32,
    back_prop=False)
  outputs = {
    'predictions': KERAS_MODEL(resized)
  }
  return outputs

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--input-dir', dest='input_dir', required=True)
  parser.add_argument('--output-dir', dest='output_dir', required=True)
  known_args, _ = parser.parse_known_args()

  KERAS_MODEL  = tensorflow.keras.models.load_model(known_args.input_dir)

  # save the output and override the signatures
  tf.saved_model.save(
    KERAS_MODEL,
    known_args.output_dir,
    signatures=my_module_encoder.get_concrete_function(
        image_bytes=tf.TensorSpec(shape=[1], dtype=tf.string)
    ),
    options=None
  )
