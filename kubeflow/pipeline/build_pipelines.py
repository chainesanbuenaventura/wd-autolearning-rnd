import build_automl_pipeline
import build_convert_pipeline
import build_retrain_pipeline

if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--env', dest='env', required=True)

  kwargs, _ = parser.parse_known_args()

  if kwargs.env == 'development':
    train_bucket = 'wizyvision-dev-automl'
    project = 'wizydam-dev'
    region = 'eu'
    svcacct_name = 'wizydam-dev-adabcf9be9a2.json'
    tag = ':latest'
    tfjs_tag = 'latest'

  elif kwargs.env == 'production':
    train_bucket = 'wizyvision-automl'
    project = 'wizdam-prod'
    region = 'eu'
    tag = ':v2'
    tfjs_tag = ':v0.17.1'
    svcacct_name = 'wizdam-prod-0965371627f5.json'

  components_image = '{region}.gcr.io/{project}/object_detection/pipeline_components{tag}'.format(
    region=region,
    project=project,
    tag=tag
  )
  object_detect_train_image = '{region}.gcr.io/{project}/object_detection/train{tag}'.format(
    region=region,
    project=project,
    tag=tag
  )
  tfjs_image = '{region}.gcr.io/{project}/wd-automl/tfjs{tfjs_tag}'.format(
    region=region,
    project=project,
    tfjs_tag=tfjs_tag,
  )

  build_automl_pipeline.build(
    train_bucket=train_bucket,
    components_image=components_image,
    object_detect_train_image=object_detect_train_image,
  )
  build_retrain_pipeline.build(
    train_bucket=train_bucket,
    components_image=components_image,
    object_detect_train_image=object_detect_train_image,
  )
  build_convert_pipeline.build(
    tfjs_image=tfjs_image,
    svcacct_name=svcacct_name
  )
