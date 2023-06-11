import kfp.dsl as dsl
import kfp.gcp as gcp
import kfp.components as comp
import train_eval

import os
# train_comp_url = 'https://raw.githubusercontent.com/kubeflow/pipelines/c0fb46aa1b32733e6cd0d4680bc192b0c3584cff/components/gcp/ml_engine/train/component.yaml'
train_comp_url = 'https://raw.githubusercontent.com/carlobambo/pipelines/e3e59fa8c7688714dea22ff6d951abee6da01f28/components/gcp/ml_engine/train/component.yaml'
mlengine_train_op = comp.load_component_from_url(train_comp_url)
COMPONENTS_IMAGE = ''
OBJECT_DETECT_TRAIN_IMAGE = ''
TRAIN_BUCKET = ''

@dsl.pipeline(
  name="WizyVision dev auto ML pipeline",
  description="Auto ML desc"
)
def automl_pipeline(
    host_url="https://auth.wizdam.xyz",
    auth_token="cDOXnwraQRO220VgaEA2uRFrKUXymlFlAcUL4F",
    auth_header="wizydam-dev-api-token",
    project="wizydam-dev",
    model_version=1,
    namespace="aerialmodeltest",
    output_bucket=TRAIN_BUCKET,
    job_name="aerialmodeltest_1_40_manual",
    optimization="high-accuracy",
    train_method="cloud",
    num_steps=5,
    eval_score_increment=0.05,
    prev_model_dir="None",
    pipeline_config_path="None",

    early_stopping_max_steps=30000,
    early_stopping_min_steps=20000,
):
  global COMPONENTS_IMAGE
  global OBJECT_DETECT_TRAIN_IMAGE
  global TRAIN_BUCKET
  dsl.ContainerOp(name='update_create_dataset_data',
    image=COMPONENTS_IMAGE,
    command=[
        "python", "/app/update_training_data.py"],
    arguments=[
      "--host-url", host_url,
      "--auth-token", auth_token,
      "--auth-header", auth_header,
      "--model-version", model_version,
      "--namespace", namespace,
      "--new-training-step", 'createDataset',
      "--training-step", 'createDataset',

      ## createDataset payload
      "bucket", output_bucket,
      "outputDir", "train_job/{}/dataset".format(job_name),
    ]).set_memory_request('200M').set_memory_limit('200M').apply(gcp.use_gcp_secret('user-gcp-sa'))

  make_dataset = dsl.ContainerOp(name='make_dataset',
    image=COMPONENTS_IMAGE,
    command=[
        "python", "/app/make_csv_dataset.py"],
    arguments=[
      "--host-url", host_url,
      "--auth-token", auth_token,
      "--auth-header", auth_header,
      "--model-version", model_version,
      "--namespace", namespace,
      "--bucket", output_bucket,
      "--output-dir", "train_job/{}/dataset".format(job_name),
    ],
    file_outputs={
      "dataset_dir": "/tmp/dataset_dir.txt",
      "tag_map_path": "/tmp/tag_map_path.txt",
    }).set_memory_request('200M').set_memory_limit('200M').apply(gcp.use_gcp_secret('user-gcp-sa'))

  split_dataset = dsl.ContainerOp(name='split_dataset',
    image=COMPONENTS_IMAGE,
    command=[
        "python", "/app/split_dataset.py"],
    arguments=[
      "--bucket", output_bucket,
      "--csv-folder", make_dataset.outputs['dataset_dir'],
      "--tag-map-path", make_dataset.outputs['tag_map_path'],
      "--output-dir", "gs://{}/train_job/{}/split".format(output_bucket, job_name),
    ],
    file_outputs={
      "train_path": "/tmp/train_path.txt",
      "val_path": "/tmp/val_path.txt",
      "test_path": "/tmp/test_path.txt",
      "split_dir": "/tmp/split_dir.txt",
      "val_size": "/tmp/val_size.txt",
      "train_size": "/tmp/train_size.txt",
      "test_size": "/tmp/test_size.txt",
    }).set_memory_request('200M').set_memory_limit('200M').apply(gcp.use_gcp_secret('user-gcp-sa'))

  dsl.ContainerOp(name='update_create_tfrecord_data',
    image=COMPONENTS_IMAGE,
    command=[
        "python", "/app/update_training_data.py"],
    arguments=[
      "--host-url", host_url,
      "--auth-token", auth_token,
      "--auth-header", auth_header,
      "--model-version", model_version,
      "--namespace", namespace,
      "--new-training-step", 'createTfRecord',
      "--training-step", 'createTfRecord',

      ## createDataset payload
      "outputDir", "gs://{}/train_job/{}/records".format(output_bucket, job_name),
      "project", project,
      "datasetDir", split_dataset.outputs['split_dir'],
    ]).apply(gcp.use_gcp_secret('user-gcp-sa'))

  make_tfrecords = dsl.ContainerOp(name='make_tf_records',
    image=COMPONENTS_IMAGE,
    command=[
        "python", "/app/make_tfrecords.py"],
    arguments=[
      "--output-dir", "gs://{}/train_job/{}/records".format(output_bucket, job_name),
      "--project", project,
      "--dataset-dir", split_dataset.outputs['split_dir'],
    ],
    file_outputs={
      "train_path": "/tmp/train_path.txt",
      "val_path": "/tmp/val_path.txt",
      "test_path": "/tmp/test_path.txt",
    }).set_memory_request('1G').set_memory_limit('1G').apply(gcp.use_gcp_secret('user-gcp-sa'))

  prep_artifacts = dsl.ContainerOp(name='prep_artifacts',
    image=COMPONENTS_IMAGE,
    command=[
        "python", "/app/prep_artifacts.py"],
    arguments=[
      "--project", project,
      "--label-map-path", make_dataset.outputs['tag_map_path'],
      "--output-dir", "gs://{}/train_job/{}/pipeline".format(output_bucket, job_name),
      "--train-path", make_tfrecords.outputs['train_path'],
      "--eval-path", make_tfrecords.outputs['val_path'],
      "--eval-count", split_dataset.outputs['val_size'],
      "--optimization", optimization,
      "--train-method", train_method,
      "--num-steps", num_steps,
      "--train-bucket", TRAIN_BUCKET,
      "--prev-model-dir", prev_model_dir,
      "--pipeline-config-path", pipeline_config_path,
      "--model-dir", "gs://{}/train_job/{}/model".format(output_bucket, job_name),
      "--batch-size", 8
    ],
    file_outputs={
      "pipeline_config_path": "/tmp/pipeline_config_path.txt",
    }).set_memory_request('500M').set_memory_limit('500M').apply(gcp.use_gcp_secret('user-gcp-sa'))

  dsl.ContainerOp(name='update_start_training_data',
    image=COMPONENTS_IMAGE,
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
      "trainPath", make_tfrecords.outputs['train_path'],
      "evalPath", make_tfrecords.outputs['val_path'],
      "evalCount", split_dataset.outputs['val_size'],
      "pipelineConfigPath", prep_artifacts.outputs['pipeline_config_path'],
      "jobDir", "gs://{}/train_job/{}/model".format(output_bucket, job_name),
      "optimization", optimization,
    ]).apply(gcp.use_gcp_secret('user-gcp-sa'))

  train_eval.train_and_eval(
    components_image=COMPONENTS_IMAGE,
    object_detect_train_image=OBJECT_DETECT_TRAIN_IMAGE,
    project=project,
    host_url=host_url,
    auth_token=auth_token,
    auth_header=auth_header,
    model_version=model_version,
    namespace=namespace,
    train_bucket=TRAIN_BUCKET,
    eval_score_increment=eval_score_increment,
    optimization=optimization,
    output_bucket=output_bucket,
    num_steps=num_steps,
    job_name=job_name,
    pipeline_config_path=prep_artifacts.outputs['pipeline_config_path'],
    run_after=prep_artifacts,
    train_method=train_method,
    train_data=[
      "trainPath", make_tfrecords.outputs['train_path'],
      "evalPath", make_tfrecords.outputs['val_path'],
      "evalCount", split_dataset.outputs['val_size'],
    ],
    early_stopping_max_steps=early_stopping_max_steps,
    early_stopping_min_steps=early_stopping_min_steps,
  )

def build(train_bucket=None, components_image=None, object_detect_train_image=None):
  global COMPONENTS_IMAGE
  global OBJECT_DETECT_TRAIN_IMAGE
  global TRAIN_BUCKET

  TRAIN_BUCKET = train_bucket
  OBJECT_DETECT_TRAIN_IMAGE = object_detect_train_image
  COMPONENTS_IMAGE = components_image

  import kfp.compiler as compiler
  compiler.Compiler().compile(automl_pipeline,  'dist/automl-pipeline.tar.gz')

# if __name__ == '__main__':
#   import kfp.compiler as compiler
#   compiler.Compiler().compile(automl_pipeline,  'automl-pipeline.tar.gz')
