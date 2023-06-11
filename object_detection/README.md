Base docker files for training

Build a docker file for object_detection2 with `object_detection.model_main_tf2`
as an entry point
```
OD_TRAIN_IMAGE=eu.gcr.io/wizydam-dev/tf/object_detection2_training:latest
docker build -f object_detection/tf2/train/Dockerfile -t ${OD_TRAIN_IMAGE}
```
