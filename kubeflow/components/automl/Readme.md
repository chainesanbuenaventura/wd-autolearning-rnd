## Build
export PROJECT=wizydam-dev
export DEPLOYMENT_NAME=wd-automl
export VERSION_TAG=v0.12.0-rc.0
export DOCKER_IMAGE_NAME=eu.gcr.io/${PROJECT}/${DEPLOYMENT_NAME}/pipeline-components:${VERSION_TAG}

docker build ./ -t ${DOCKER_IMAGE_NAME} -f ./Dockerfile

## RUN
export ARTIFACTS_DIR=~/Projects/wizdam/autolearning/kubeflow/components/bq2tfrecord/artifacts
docker run -it \
  -v ${PWD}/src:/src -v $HOME/.config/gcloud:/home/tensorflow/.config/gcloud \
  -v /Users/carlo/Projects/wizdam/autolearning/container_data:/mnt/data \
  ${DOCKER_IMAGE_NAME} bash


## Push
docker push ${DOCKER_IMAGE_NAME}



## BUILD PROD
export PROJECT=wizdam-prod
export DEPLOYMENT_NAME=wd-automl
export VERSION_TAG=v0.12.0
export DOCKER_IMAGE_NAME=eu.gcr.io/${PROJECT}/${DEPLOYMENT_NAME}/pipeline-components:${VERSION_TAG}

docker build ./ -t ${DOCKER_IMAGE_NAME} -f ./Dockerfile
