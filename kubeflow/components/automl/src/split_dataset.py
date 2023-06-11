import argparse
from io import BytesIO, StringIO
import os

from google.cloud import storage
import pandas as pd
import sklearn.model_selection as model_selection

client = storage.Client()

def list_csvs(bucket, folder):
  blobs = client.list_blobs(bucket, prefix=folder.replace('gs://{}/'.format(bucket), ''))
  return list(blobs)

def upload_csv(bucket, path, pd):
  gcs_bucket = client.get_bucket(bucket)
  blob = gcs_bucket.blob(path)
  csv_data = pd.to_csv(index=False, encoding='utf-8')
  blob.upload_from_string(csv_data, content_type="text/csv")

def split_dataset(params):
  columns = ['path', 'name', 'labels', 'labelIds', 'xmins', 'xmaxs', 'ymins', 'ymaxs']
  df = pd.DataFrame(columns=columns)
  csvs = list_csvs(params.bucket, params.csv_folder)
  for f in csvs:
    file_contents = f.download_as_string()

  # Append it to our DataFrame.
    string = BytesIO(file_contents)
    df = df.append(pd.read_csv(string))
  df = df.reset_index(drop=True)
  train, test_val = model_selection.train_test_split(
      df,
      train_size=0.70,
      test_size=0.30,
      random_state=101)
  val, test = model_selection.train_test_split(
    test_val,
    train_size=0.5,
    test_size=0.5,
    random_state=101
  )
  output_path = params.output_dir.replace('gs://{}/'.format(params.bucket), '')
  upload_csv(params.bucket, os.path.join(output_path, 'train.csv'), train)
  upload_csv(params.bucket, os.path.join(output_path, 'val.csv'), val)
  upload_csv(params.bucket, os.path.join(output_path, 'test.csv'), test)

  paths = {
    "train": os.path.join(params.output_dir, 'train.csv'),
    "val": os.path.join(params.output_dir, 'val.csv'),
    "test": os.path.join(params.output_dir, 'test.csv')
  }
  sizes = {
    "val": str(len(val.index)),
    "train": str(len(train.index)),
    "test": str(len(test.index))
  }
  return paths, sizes

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--output-dir', dest='output_dir', required=True)
  parser.add_argument('--csv-folder', dest='csv_folder', required=True)
  parser.add_argument('--tag-map-path', dest='tag_map_path', required=True)
  parser.add_argument('--bucket', dest='bucket', required=True)

  known_args, _ = parser.parse_known_args()
  paths, sizes = split_dataset(known_args)
  print("train_path: {}".format(paths.get('train')))
  print("test_path: {}".format(paths.get('test')))
  print("val_path: {}".format(paths.get('val')))
  print("val_size: {}".format(sizes.get('val')))
  print("train_size: {}".format(sizes.get('train')))
  print("test_size: {}".format(sizes.get('test')))
  print("split_dir: {}".format(known_args.output_dir))
  # write output
  with open("/tmp/train_path.txt", "w") as f:
    f.write(paths.get('train'))
  with open("/tmp/test_path.txt", "w") as f:
    f.write(paths.get('test'))
  with open("/tmp/val_path.txt", "w") as f:
    f.write(paths.get('val'))
  with open("/tmp/val_size.txt", "w") as f:
    f.write(sizes.get('val'))
  with open("/tmp/train_size.txt", "w") as f:
    f.write(sizes.get('train'))
  with open("/tmp/test_size.txt", "w") as f:
    f.write(sizes.get('test'))
  with open("/tmp/split_dir.txt", "w") as f:
    f.write(known_args.output_dir)

  """
    python split_dataset.py \
      --bucket=wizyvision-dev-automl \
      --csv-folder=gs://wizyvision-dev-automl/train_job/tf2_manual/dataset/all \
      --tag-map-path=gs://wizyvision-dev-automl/train_job/tf2_manual/dataset/tag_map.csv \
      --output-dir=gs://wizyvision-dev-automl/train_job/tf2_manual/dataset/split

    outputs:
      train_job/dummy-model-v1/dataset/all
      train_path: gs://wizyvision-dev-automl/train_job/tf2_manual/dataset/split/train.csv
      test_path: gs://wizyvision-dev-automl/train_job/tf2_manual/dataset/split/test.csv
      val_path: gs://wizyvision-dev-automl/train_job/tf2_manual/dataset/split/val.csv
      val_size: 5
      train_size: 23
      test_size: 6
      split_dir: gs://wizyvision-dev-automl/train_job/tf2_manual/dataset/split
  """
