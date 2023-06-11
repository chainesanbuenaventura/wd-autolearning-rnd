export PROD_PROJECT=wizdam-prod
export PROD_DEPLOYMENT_NAME=wd-automl
export PROD_VERSION=v1.0.0
export PROD_DOCKER_IMAGE_NAME=eu.gcr.io/${PROD_PROJECT}/${PROD_DEPLOYMENT_NAME}/pipeline-components:${PROD_VERSION}

docker build ./ -t ${PROD_DOCKER_IMAGE_NAME} -f ./Dockerfile
docker push ${PROD_DOCKER_IMAGE_NAME}
