import argparse
import requests
from server_api import ServerApi
# from smart_open import open
import csv
import os
import tempfile
from datetime import datetime, timedelta
from google.cloud import storage

def write_csv(blob, columns, rows):
  with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_csv:
    writer = csv.DictWriter(temp_csv, fieldnames = columns)
    writer.writeheader()
    writer.writerows(rows)
    temp_csv.close()
    blob.upload_from_filename(temp_csv.name)
    os.unlink(temp_csv.name)

def main(params):
  server_api = ServerApi(params.host_url, params.auth_token, params.auth_header_key)
  def get_files(limit, offset, tag_map={}):
    # extracting data in json format
    files, total = server_api.get_files(params, limit, offset)
    formatted = []
    for f in files:
      labels = []
      labelIds = []
      xmins = []
      xmaxs = []
      ymaxs = []
      ymins = []
      for tag in f.get('tags'):
        for ann in tag.get('annotations'):
          xmin = min(ann.get('boundingPoly'), key=lambda x:x['x'])['x']
          xmax = max(ann.get('boundingPoly'), key=lambda x:x['x'])['x']

          ymin = min(ann.get('boundingPoly'), key=lambda y:y['y'])['y']
          ymax = max(ann.get('boundingPoly'), key=lambda y:y['y'])['y']
          labels.append(tag.get('name'))
          labelIds.append('{}'.format(tag_map[tag.get('id')]))
          xmins.append('{}'.format(xmin))
          xmaxs.append('{}'.format(xmax))
          ymaxs.append('{}'.format(ymax))
          ymins.append('{}'.format(ymin))
      formatted.append({
        'path': f.get('url'),
        'name': f.get('name'),
        'labels': ';'.join(labels),
        'labelIds': ';'.join(labelIds),
        'xmins': ';'.join(xmins),
        'xmaxs': ';'.join(xmaxs),
        'ymins': ';'.join(ymins),
        'ymaxs': ';'.join(ymaxs)
      })
    return formatted, total

  client = storage.Client()
  bucket = client.get_bucket(params.bucket)
  tags = server_api.get_tags({
    'namespace': params.namespace,
    'filter[namespace]': params.namespace,
    'filter[versionId]': params.model_version
  })
  tag_map = { x.get('id'): x.get('labelMapId') for x in tags}
  tag_map_blob = bucket.blob(params.tag_map_path)
  write_csv(tag_map_blob, ["id", "labelMapId", "name"], tags)

  page_size = 200

  def get_and_write_files(limit, offset, page):
    files, total = get_files(page_size, offset, tag_map=tag_map)
    blob = bucket.blob(os.path.join(params.csv_path, 'files-{}.csv'.format(page)))
    write_csv(blob, ["path", "name", "labels", "labelIds", "xmins", "xmaxs", "ymins", "ymaxs"], files)

    if total > limit + offset:
      page = page + 1
      get_and_write_files(limit, offset + limit, page)
  get_and_write_files(page_size, 0, 1)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--host-url', dest='host_url', required=True)
  parser.add_argument('--auth-token', dest='auth_token', required=True)
  parser.add_argument('--auth-header', dest='auth_header_key', required=True)
  parser.add_argument('--model-version', dest='model_version', required=True)
  parser.add_argument('--namespace', dest='namespace', required=True)
  parser.add_argument('--bucket', dest='bucket', required=True)
  parser.add_argument('--output-dir', dest='output_dir', required=True)

  known_args, pipeline_args = parser.parse_known_args()
  known_args.csv_path = os.path.join(known_args.output_dir, 'all')
  known_args.tag_map_path = os.path.join(known_args.output_dir, 'tag_map.csv')
  main(known_args)
  dataset_dir = os.path.join('gs://', known_args.bucket, known_args.csv_path)
  tag_map_path = os.path.join('gs://', known_args.bucket, known_args.tag_map_path)
  print("dataset_dir: {}".format(dataset_dir))
  print("tag_map_path: {}".format(tag_map_path))

  with open("/tmp/dataset_dir.txt", "w") as f:
    f.write(dataset_dir)
  with open("/tmp/tag_map_path.txt", "w") as f:
    f.write(tag_map_path)

"""
  Testing in cli
  python make_csv_dataset.py \
    --host-url=https://auth.wizdam.xyz \
    --auth-token=cDOXnwraQRO220VgaEA2uRFrKUXymlFlAcUL4F \
    --auth-header=wizydam-dev-api-token \
    --model-version=25 \
    --namespace=aerialmodeltest \
    --bucket=wizyvision-dev-automl \
    --output-dir=train_job/tf2_manual/dataset

  outputs:
    dataset_dir: gs://wd-model-rnd-0000/train_job/dummy-model-v1/dataset/all
    tag_map_path: gs://wd-model-rnd-0000/train_job/dummy-model-v1/dataset/tag_map.csv


  gsutil cp -r gs://wizyvision-dev-automl/train_job/tf2_manual/dataset/all/* /mnt/data/dataset/all/
  gsutil cp gs://wizyvision-dev-automl/train_job/tf2_manual/dataset/tag_map.csv /mnt/dataset/tag_map.csv
"""
