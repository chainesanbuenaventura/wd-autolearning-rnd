1. Install docker
2. Authenticate in google cloud container registry
  1. gcloud auth login
  2. gcloud auth application default login
  3. gcloud auth configure-docker


RUNNING

docker run -it \
  --mount src=`pwd`,target=/src,type=bind \
  eu.gcr.io/wizydam-dev/wd-automl/preproc:v0.11.0-rc0

gsutil cp templates/tensorflow2/faster_rcnn_resnet50_v1_800x1333_coco17_gpu-8.config gs://wizyvision-dev-automl/static/pipeline/templates/tensorflow2/

gsutil cp gs://wizyvision-dev-automl/static/pipeline/fine_tune_checkpoints/tensorflow2/faster_rcnn_resnet50_v1_800x1333_coco17_gpu


/Users/carlo/Projects/wizdam/tensorflow/data/fine_tune_checkpoints/faster_rcnn_resnet50_v1_800x1333_coco17_gpu-8

gsutil cp -r /Users/carlo/Projects/wizdam/tensorflow/data/fine_tune_checkpoints/faster_rcnn_resnet50_v1_800x1333_coco17_gpu-8/* gs://wizyvision-dev-automl/static/pipeline/fine_tune_checkpoints/tensorflow2/faster_rcnn_resnet50_v1_800x1333_coco17_gpu/
