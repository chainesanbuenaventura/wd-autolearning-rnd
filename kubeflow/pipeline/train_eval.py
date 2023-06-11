import kfp.dsl as dsl
import kfp.gcp as gcp
import kfp.components as comp

train_comp_url = 'https://raw.githubusercontent.com/carlobambo/pipelines/e3e59fa8c7688714dea22ff6d951abee6da01f28/components/gcp/ml_engine/train/component.yaml'
mlengine_train_op = comp.load_component_from_url(train_comp_url)

def train_and_eval(
  components_image=None,
  project=None,
  host_url=None,
  auth_token=None,
  auth_header=None,
  model_version=None,
  namespace=None,
  train_bucket=None,
  eval_score_increment=None,
  optimization=None,
  train_method=None,
  num_steps=None,
  output_bucket=None,
  job_name=None,
  pipeline_config_path=None,
  run_after=None,
  train_data=[],
  early_stopping_max_steps=None,
  early_stopping_min_steps=None,
  object_detect_train_image=None,
):
  job_dir = "gs://{}/train_job/{}/model".format(output_bucket, job_name)
  train_args = {
    "args": [
      "--model_dir=gs://{}/train_job/{}/model".format(output_bucket, job_name),
      "--pipeline_config_path={}".format(pipeline_config_path),
      "--early_stopping_enabled=1",
      "--early_stopping_max_steps={}".format(early_stopping_max_steps),
      "--early_stopping_min_steps={}".format(early_stopping_min_steps),
    ],
    # --master-machine-type n1-highmem-16 \
    # --master-accelerator count=8,type=nvidia-tesla-k80 \
    # --master-image-uri eu.gcr.io/wizydam-dev/tf/object_detection2_training:latest \
    "scaleTier": "{}".format("CUSTOM"),
    "masterType": "n1-highmem-8",
    "masterConfig": {
      "imageUri": "{}".format(object_detect_train_image),
      "acceleratorConfig": {
        "count": 2,
        "type": "nvidia-tesla-p100"
      },
    },
  }

  train = mlengine_train_op(
    project_id=project,
    training_input=train_args,
    job_id_prefix=job_name,
    # package_uris=[
    #   "gs://{}/static/codes/object_detection/dist/object_detection-0.1.tar.gz".format(train_bucket),
    #   "gs://{}/static/codes/object_detection/slim/dist/slim-0.1.tar.gz".format(train_bucket),
    #   "gs://{}/static/codes/object_detection/pycocotools/pycocotools-2.0.tar.gz".format(train_bucket),
    # ],
    region="europe-west1",
    job_dir=job_dir,
  ).apply(gcp.use_gcp_secret('user-gcp-sa')).after(run_after)

  add_job_id = dsl.ContainerOp(name='update_start_training_data_with_job_id',
    image=components_image,
    command=[
        "python", "/app/update_training_data.py"],
    arguments=[
      "--host-url", host_url,
      "--auth-token", auth_token,
      "--auth-header", auth_header,
      "--model-version", model_version,
      "--namespace", namespace,
      "--new-training-step", 'startTrainingJob',
      "--training-step", 'startTrainingJob',

      ## createDataset payload

      "numSteps", num_steps,
      "pipelineConfigPath", pipeline_config_path,
      "jobDir", job_dir,
      "optimization", optimization,
      "jobId", train.outputs['job_id'],
    ] + train_data).apply(gcp.use_gcp_secret('user-gcp-sa')).after(train)

  add_eval_data = dsl.ContainerOp(name='update_eval_data',
    image=components_image,
    command=[
        "python", "/app/update_training_data.py"],
    arguments=[
      "--host-url", host_url,
      "--auth-token", auth_token,
      "--auth-header", auth_header,
      "--model-version", model_version,
      "--namespace", namespace,
      "--new-training-step", 'evaluate',
      "--training-step", 'evaluate',

      ## createDataset payload
    ]).apply(gcp.use_gcp_secret('user-gcp-sa')).after(add_job_id)



  export = dsl.ContainerOp(name='export_model',
    image=components_image,
    command=[
        "python", "/app/export_model.py"],
    arguments=[
      "--pipeline-config-path", pipeline_config_path,
      "--checkpoint-dir", job_dir,
      "--input-type", "encoded_image_string_tensor",
      "--output-dir", "gs://{}/train_job/{}/export".format(output_bucket, job_name),
      "--train-dir", train.outputs['job_dir'],
      "--optimization", optimization,
      "--train-method", train_method,
    ],
    file_outputs={
      "saved_model_path": "/tmp/saved_model_path.txt",
    }).set_memory_request('6G').set_memory_limit('6G').apply(gcp.use_gcp_secret('user-gcp-sa')).after(add_eval_data)

  export_image_tensor = dsl.ContainerOp(name='export_image_tensor_model',
    image=components_image,
    command=[
        "python", "/app/export_model.py"],
    arguments=[
      "--pipeline-config-path", pipeline_config_path,
      "--checkpoint-dir", job_dir,
      "--input-type", "image_tensor",
      "--output-dir", "gs://{}/train_job/{}/export_image_tensor".format(output_bucket, job_name),
      "--train-dir", train.outputs['job_dir'],
      "--optimization", optimization,
      "--train-method", train_method
    ],
    file_outputs={
      "saved_model_path": "/tmp/saved_model_path.txt",
    }).set_memory_request('6G').set_memory_limit('6G').apply(gcp.use_gcp_secret('user-gcp-sa')).after(export)

  to_tflite = dsl.ContainerOp(name='to_tflite',
    image=components_image,
    command=[
      "python",
      "/app/tf_to_tflite.py"
    ],
    arguments=[
      "--pipeline-config-path", pipeline_config_path,
      "--trained-checkpoint-prefix", "gs://{}/train_job/{}/export_image_tensor/checkpoint".format(output_bucket, job_name),
      "--graph-def-path", "gs://{}/train_job/{}/tf_lite_graph".format(output_bucket, job_name),
      "--output-file", "gs://{}/train_job/{}/tflite/model.tflite".format(output_bucket, job_name),
      "--optimization", optimization,
      "--train-method", train_method,
    ],
    file_outputs={
      "tflite_path": "/tmp/tflite_path.txt",
    }).set_memory_request('6G').set_memory_limit('6G').apply(gcp.use_gcp_secret('user-gcp-sa')).after(export_image_tensor)

  convert_tfjs = dsl.ContainerOp(name='convert_tfjs',
    image=components_image,
    command=["tensorflowjs_converter"],
    arguments=[
      "--control_flow_v2=False",
      "--input_format=tf_saved_model",
      "--saved_model_tags=serve",
      "--signature_name=serving_default",
      "--skip_op_check",
      "--strip_debug_ops=True",
      "--weight_shard_size_bytes=4194304",
      export_image_tensor.outputs['saved_model_path'], # input saved_model.pb format
      "gs://{}/train_job/{}/tfjs".format(output_bucket, job_name) # tfjs model output_dir
    ]).set_memory_request('6G').set_memory_limit('6G').apply(gcp.use_gcp_secret('user-gcp-sa')).after(to_tflite)

  eval_job = dsl.ContainerOp(name='run_eval',
    image=components_image,
    command=[
        "python", "/app/run_eval.py"],
    arguments=[
      "--pipeline_config_path={}".format(pipeline_config_path),
      "--model_dir=gs://{}/train_job/{}/eval".format(output_bucket, job_name),
      "--run_once=True".format(),
      "--checkpoint_dir=gs://{}/train_job/{}/export".format(output_bucket, job_name),
      "--score_increment={}".format(eval_score_increment),
      "--host_url={}".format(host_url),
      "--auth_token={}".format(auth_token),
      "--auth_header_key={}".format(auth_header),
      "--model_version={}".format(model_version),
      "--namespace={}".format(namespace),
      "--optimization", optimization,
      "--train_method", train_method
    ]).set_memory_request('6G').set_memory_limit('6G').apply(gcp.use_gcp_secret('user-gcp-sa')).after(convert_tfjs)

  return dsl.ContainerOp(name='update_done_data',
    image=components_image,
    command=[
        "python", "/app/update_training_data.py"],
    arguments=[
      "--host-url", host_url,
      "--auth-token", auth_token,
      "--auth-header", auth_header,
      "--model-version", model_version,
      "--namespace", namespace,
      "--new-training-step", 'done',
      "--training-step", 'done',

      ## createDataset payload
      "deploymentUri", export.outputs['saved_model_path'],
      "tfjsUri", "gs://{}/train_job/{}/tfjs".format(output_bucket, job_name),
      "tfliteUri", to_tflite.outputs['tflite_path']
    ]).after(eval_job).apply(gcp.use_gcp_secret('user-gcp-sa'))
