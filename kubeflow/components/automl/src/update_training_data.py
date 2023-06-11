import argparse

from server_api import ServerApi



def update(server_api, namespace, version_id, step_data, new_step):
  server_api.update_training_data(namespace, version_id, step_data, new_step=new_step)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--host-url', dest='host_url', required=True)
  parser.add_argument('--auth-token', dest='auth_token', required=True)
  parser.add_argument('--auth-header', dest='auth_header_key', required=True)
  parser.add_argument('--model-version', dest='model_version', required=True)
  parser.add_argument('--namespace', dest='namespace', required=True)
  parser.add_argument('--training-step', dest='training_step', required=True)
  parser.add_argument('--new-training-step', dest='new_training_step', default=None)

  params, unknown = parser.parse_known_args()

  step_data = dict(zip(unknown[:-1:2],unknown[1::2]))

  host_url = params.host_url
  auth_token= params.auth_token
  header_key = params.auth_header_key
  namespace = params.namespace
  server_api = ServerApi(host_url, auth_token, header_key)
  training_data = {}
  training_data[params.training_step] = step_data

  update(server_api, namespace, params.model_version, training_data, params.new_training_step)

  """
    python update_training_data.py \
      --host-url=https://auth.wizdam.xyz \
      --auth-token=cDOXnwraQRO220VgaEA2uRFrKUXymlFlAcUL4F \
      --auth-header=wizydam-dev-api-token \
      --model-version=5 \
      --namespace=aerialmodeltest \
      --training-step=startTrainingJob \
      --new-training-step=done \
      numSteps 2000 \
      trainPath gs://wd-model-rnd-0000/train_job/aerialmodeltest__m1__v5/records/train-?????-of-?????.tfrecords \
      evalPath gs://wd-model-rnd-0000/train_job/aerialmodeltest__m1__v5/records/val-?????-of-?????.tfrecords \
      evalCount 8 \
      pipelineConfigPath gs://wd-model-rnd-0000/train_job/aerialmodeltest__m1__v5/pipeline/faster_rcnn_pipeline.config \
      jobDir gs://wd-model-rnd-0000/train_job/aerialmodeltest__m1__v5/model \
      optimization higher-accuracy
  """
