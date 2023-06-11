export PROJECT=wizydam-dev
export DEPLOYMENT_NAME=wd-automl
export DEV_VERSION=latest
export TFJS_IMAGE=eu.gcr.io/${PROJECT}/${DEPLOYMENT_NAME}/tfjs:${DEV_VERSION}

docker build ./ -t ${TFJS_IMAGE} -f ./Dockerfile
docker push ${TFJS_IMAGE}



# docker run -it \
#   -v ${PWD}/src:/src -v $HOME/.config/gcloud:/home/tensorflow/.config/gcloud \
#   -v /Users/carlo/Projects/wizdam/autolearning/container_data:/mnt/data \
#   ${TFJS_IMAGE} bash


# tensorflowjs_converter \
#   --control_flow_v2=False \
#   --input_format=tf_saved_model \
#   --saved_model_tags=serve \
#   --signature_name=serving_default \
#   --skip_op_check --strip_debug_ops=True \
#   --weight_shard_size_bytes=4194304 \
#   /mnt/data/tfjs_car/export/saved_model /mnt/data/tfjs_car/tfjs_model
