import os

import kfp.dsl as dsl
import kfp.gcp as gcp
import kfp.components as comp

TFJS_IMAGE = ''
SVCACCT_NAME = ''

@dsl.pipeline(
  name="WizyML Studio conversion pipeline",
  description="Conversion to different ML formats"
)
def convert_pipeline(
  input_dir="gs://wizyvision-dev-automl/train_job/leemanual__m18__v87/export/saved_model",
  output_prefix="gs://wizyvision-dev-automl/train_job/leemanual__m18__v87/export",
  host_url="https://wizydam-dev.appspot.com",
  auth_header="wizydam-dev-api-token",
  auth_token="cDOXnwraQRO220VgaEA2uRFrKUXymlFlAcUL4F",
  namespace="leemanual",
  model_version=87
):
  global TFJS_IMAGE
  global SVCACCT_NAME

  output_tf = '{}/tf'.format(output_prefix)
  make_keras_model = dsl.ContainerOp(name='make_keras_model',
    image=TFJS_IMAGE,
    command=[
        "python", "/app/make_keras_model.py"],
    arguments=[
      "--input-dir", input_dir,
      "--output-dir", output_tf,
      "--svcacct-name", SVCACCT_NAME
    ],
    file_outputs={}).set_memory_request('2G').set_memory_limit('2G').apply(gcp.use_gcp_secret('user-gcp-sa'))

  output_savedmodel = '{}/saved_model'.format(output_prefix)
  to_aiplatform_format = dsl.ContainerOp(name='to_aiplatform_format',
    image=TFJS_IMAGE,
    command=[
        "python", "/app/to_aiplatform_format.py"],
    arguments=[
      "--input-dir", output_tf,
      "--output-dir", output_savedmodel
    ],
    file_outputs={}).set_memory_request('2G').set_memory_limit('2G').apply(gcp.use_gcp_secret('user-gcp-sa')).after(make_keras_model)

  output_tflite = '{}/tflite'.format(output_prefix)
  to_tflite = dsl.ContainerOp(name='to_tflite',
    image=TFJS_IMAGE,
    command=["python", "/app/tf_to_tflite.py"],
    arguments=[
      "--input-dir={}".format(output_tf),
      "--output-file={}/model.tflite".format(output_tflite),
      "--svcacct-name", SVCACCT_NAME,
    ],
    file_outputs={}).set_memory_request('2G').set_memory_limit('2G').apply(gcp.use_gcp_secret('user-gcp-sa')).after(to_aiplatform_format)


  dsl.ContainerOp(name='update_server',
    image=TFJS_IMAGE,
    command=[
        "python", "/app/update_server.py"],
    arguments=[
      "--host-url", host_url,
      "--auth-token", auth_token,
      "--auth-header", auth_header,
      "--namespace", namespace,
      "--model-version", model_version,
      "--tf-url", output_savedmodel,
      "--tflite-url", output_tflite
    ]).set_memory_request('200M').set_memory_limit('200M').apply(gcp.use_gcp_secret('user-gcp-sa')).after(to_tflite)

def build(tfjs_image=None, svcacct_name=None):
  global TFJS_IMAGE
  global SVCACCT_NAME

  TFJS_IMAGE = tfjs_image
  SVCACCT_NAME = svcacct_name

  import kfp.compiler as compiler
  compiler.Compiler().compile(convert_pipeline,  'dist/convert-pipeline.tar.gz')
