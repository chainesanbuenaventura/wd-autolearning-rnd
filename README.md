#SETUP
1. python3 -m venv .env
2. source .env/bin/activate
3. pip install -r requirements.text


Upload template
python -m examples.mymodule \
  --runner DataflowRunner \
  --project YOUR_PROJECT_ID \
  --staging_location gs://wd-model-rnd-0000/staging \
  --temp_location gs://wd-model-rnd-0000/temp \
  --template_location gs://wd-model-rnd-0000/templates/MakeDatasetTemplate

  python MakeDatesetTemplate.py \
    --project wizydam-dev \
    --staging_location gs://wd-model-rnd-0000/staging \
    --temp_location gs://wd-model-rnd-0000/temp \
    --parameters bq_dataset="autolearningRND.bottlestTest2",output_dir="gs://wd-model-rnd-0000/test_bottles/records",temp_dir="gs://wd-model-rnd-0000/test_bottles/tmp"

python MakeDatesetTemplate.py \
  --requirements_file requirements_new.txt \
  --project wizydam-dev \
  --runner DataflowRunner \
  --staging_location gs://wd-model-rnd-0000/staging \
  --temp_location gs://wd-model-rnd-0000/temp \
  --template_location gs://wd-model-rnd-0000/dataflow_templates/MakeDatasetTemplate
  --extra_package tfx-0.21.2-py3-none-any.whl


python MakeDatesetTemplate.py \
  --setup_file ./setup.py \
  --project wizydam-dev \
  --runner DataflowRunner \
  --staging_location gs://wd-model-rnd-0000/staging \
  --temp_location gs://wd-model-rnd-0000/temp \
  --template_location gs://wd-model-rnd-0000/dataflow_templates/MakeDatasetTemplate



gcloud ml-engine jobs submit training object_detection_`date +%m_%d_%Y_%H_%M_%S` \
    --runtime-version 1.15 \
    --python-version 2.7 \
    --job-dir=gs://wd-model-rnd-0000/object_detection/models27/ \
    --packages gs://wd-model-rnd-0000/codes/object_detection/dist/object_detection-0.1.tar.gz,gs://wd-model-rnd-0000/codes/object_detection/slim/dist/slim-0.1.tar.gz,gs://wd-model-rnd-0000/codes/object_detection/pycocotools/pycocotools-2.0.tar.gz \
    --module-name object_detection.model_main \
    --region europe-west1 \
    --config ./train.yaml \
    -- \
    --model_dir=gs://wd-model-rnd-0000/object_detection/models27/ \
    --pipeline_config_path=gs://wd-model-rnd-0000/kubeflow/test-4/artifacts/tfrecords/faster_rcnn_pipeline.config


python export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path gs://wd-model-rnd-0000/kubeflow/test-4/artifacts/tfrecords/faster_rcnn_pipeline.config \
    --trained_checkpoint_prefix gs://wd-model-rnd-0000/object_detection/models27/model.ckpt-20 \
    --output_directory gs://wd-model-rnd-0000/object_detection/models27/image_tensor \


gsutil cp -r /Users/carlo/Projects/machine-learning/bottles/bottles_model/outputs/cloud_training2/saved_model/ gs://wd-model-rnd-0000/object_detection/models27/encoded_image_tensor

gcloud ml-engine versions create encoded_image_tensor --model wizdam_demo_bottles --origin=gs://wd-model-rnd-0000/object_detection/models27/encoded_image_tensor --runtime-version=1.15 --python-version=2.7
