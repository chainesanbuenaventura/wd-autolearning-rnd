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
  name="WizyVision dev retrain pipeline",
  description="Retrain desc"
)
def retrain_pipeline(
    host_url="https://auth.wizdam.xyz",
    auth_token="cDOXnwraQRO220VgaEA2uRFrKUXymlFlAcUL4F",
    auth_header="wizydam-dev-api-token",
    project="wizydam-dev",
    namespace="aerialmodeltest",
    output_bucket=TRAIN_BUCKET,

    job_name="retrain_aerial_model_test1",
    model_version=1,
    num_steps=5,
    pipeline_config_path=None,
    prev_model_dir=None,
    optimization="high-accuracy",
    eval_score_increment=0.05,
    train_method='cloud',
    early_stopping_max_steps=30000,
    early_stopping_min_steps=20000,
):
  global COMPONENTS_IMAGE
  global OBJECT_DETECT_TRAIN_IMAGE
  global TRAIN_BUCKET
  prep_artifacts = dsl.ContainerOp(name='prep_artifacts',
    image=COMPONENTS_IMAGE,
    command=[
        "python", "/app/prep_retrain_artifacts.py"],
    arguments=[
      "--project", project,
      "--output-dir", "gs://{}/train_job/{}".format(output_bucket, job_name),
      "--optimization", optimization,
      "--num-steps", num_steps,
      "--prev-model-dir", prev_model_dir,
      "--pipeline-config-path", pipeline_config_path,
      "--batch-size", 8,
      "--train-method", train_method
    ],
    file_outputs={
      "pipeline_config_path": "/tmp/pipeline_config_path.txt",
      "model_dir": "/tmp/model_dir.txt",
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
    train_data=[],
    early_stopping_max_steps=early_stopping_max_steps,
    early_stopping_min_steps=early_stopping_min_steps,
    train_method=train_method
  )

# if __name__ == '__main__':
#   import kfp.compiler as compiler
#   compiler.Compiler().compile(retrain_pipeline, 'retrain-pipeline.tar.gz')
def build(train_bucket=None, components_image=None, object_detect_train_image=None):
  global COMPONENTS_IMAGE
  global OBJECT_DETECT_TRAIN_IMAGE
  global TRAIN_BUCKET

  TRAIN_BUCKET = train_bucket
  OBJECT_DETECT_TRAIN_IMAGE = object_detect_train_image
  COMPONENTS_IMAGE = components_image

  import kfp.compiler as compiler
  compiler.Compiler().compile(retrain_pipeline,  'dist/retrain-automl-pipeline.tar.gz')



