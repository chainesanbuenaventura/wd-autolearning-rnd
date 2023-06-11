export PROJECT=wizydam-dev
export DEPLOYMENT_NAME=object_detection
export DEV_VERSION=latest
export DOCKER_IMAGE_NAME=eu.gcr.io/${PROJECT}/${DEPLOYMENT_NAME}/pipeline_components:${DEV_VERSION}

docker build ./ -t ${DOCKER_IMAGE_NAME} -f ./Dockerfile
docker push ${DOCKER_IMAGE_NAME}
