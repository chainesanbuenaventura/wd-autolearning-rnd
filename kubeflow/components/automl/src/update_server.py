import argparse
import requests
import json

def get_version(url, headers, params):
  r = requests.get(
    url=url,
    headers=headers,
    params=params,
    verify=False
  )
  r.raise_for_status()
  json_data = r.json()
  return json_data['data']

def update_version(url, headers, data):
  r = requests.put(
    url=url,
    headers=headers,
    verify=False,
    data=json.dumps(data)
  )
  json_data = r.json()
  return json_data

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--host-url', dest='host_url', required=True)
  parser.add_argument('--auth-token', dest='auth_token', required=True)
  parser.add_argument('--auth-header', dest='auth_header_key', required=True)
  parser.add_argument('--namespace', dest='namespace', required=True)
  parser.add_argument('--model-version', dest='model_version', required=True)
  parser.add_argument('--tf-url', dest='tf_url', required=True)
  parser.add_argument('--tflite-url', dest='tflite_url', required=True)
  known_args, _ = parser.parse_known_args()

  version_info = get_version(
    '{}/api/v1/server/ml/versions/{}'.format(known_args.host_url, known_args.model_version),
    {
      'content-type': 'application/json',
      known_args.auth_header_key: known_args.auth_token
    },
    {
      'namespace': known_args.namespace,
    }
  )

  version_info['trainingData']['done']['deploymentUri'] = known_args.tf_url;
  version_info['trainingData']['done']['tfliteUri'] = known_args.tflite_url;
  update_version(
    '{}/api/v1/server/ml/versions/{}'.format(known_args.host_url, known_args.model_version),
    {
      'content-type': 'application/json',
      known_args.auth_header_key: known_args.auth_token
    },
    {
      'namespace': known_args.namespace,
      'trainingData': version_info['trainingData'],
      'trainingStatus': 'done'
    }
  )
