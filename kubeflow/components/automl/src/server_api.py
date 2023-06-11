import requests
import json
import datetime

class ServerApi():
  def __init__(self, host_url, auth_token, auth_header_key):
    self.host_url = host_url
    self.auth_token = auth_token
    self.auth_header_key = auth_header_key

  def get_headers(self):
    headers = {
      'content-type': 'application/json'
    }
    headers[self.auth_header_key] = self.auth_token
    return headers

  def get_url(self, path):
    url = self.host_url + path
    return url

  def get_request(self, path, params, raise_when_failed=True):
    url = self.get_url(path)
    headers = self.get_headers()
    r = requests.get(
      url=url,
      params=params,
      headers=headers,
      verify=False
    )
    r.raise_for_status()
    # extracting data in json format
    json_data = r.json()
    return json_data
  def put_request(self, path, data, params = None):
    url = self.get_url(path)
    headers = self.get_headers()
    r = requests.put(
      url=url,
      params=params,
      headers=headers,
      verify=False,
      data=json.dumps(data)
    )
    # r.raise_for_status()
    # extracting data in json format
    json_data = r.json()
    return json_data

  def get_files(self, params, limit, offset):
    path =  '/api/v1/server/ml/files'
    request_params = {
      'filter[versionId]': params.model_version,
      'filter[namespace]': params.namespace,
      'filter[hasTag]': 'true',
      'namespace': params.namespace,

      'page[limit]': limit,
      'page[offset]': offset
    }
    json_data = self.get_request(path, request_params)
    files = json_data.get('data').get('files')
    total = json_data.get('data').get('total')
    return files, total

  def get_tags(self, params):
    json_data = self.get_request('/api/v1/server/ml/tags?', params)
    tags = json_data.get('data').get('tags')
    return tags

  def update_training_data(self, namespace, id, stepData, new_step = None):
    path = '/api/v1/server/ml/versions/{}'.format(id)
    request_params = { 'namespace': namespace, 'request_id': 1 }
    response = self.get_request(path, request_params)
    version = response.get('data')
    new_training_data = version['trainingData'].copy()
    new_training_data.update(stepData)

    # update dates
    now = datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S%z')
    if new_step is not None:
      prev_step = version['trainingStatus']
      if new_step in new_training_data:
        started_at = now
        if new_step in version['trainingData']:
          if 'startedAt' in version['trainingData'][new_step]:
            started_at = version['trainingData'][new_step]['startedAt']
        new_training_data[new_step].update({
          'startedAt': started_at,
        })
      # update end at if changed
      if prev_step != new_step:
        if prev_step in new_training_data:
          new_training_data[prev_step].update({
            'endedAt': now,
          })

    update_payload = {
      'namespace': namespace,
      'trainingData': new_training_data,
    }
    if new_step is not None:
      update_payload['trainingStatus'] = new_step
    print('update_payload:{}'.format(update_payload))
    print('new_step:{}'.format(new_step))
    response = self.put_request(path, update_payload, params=request_params)

