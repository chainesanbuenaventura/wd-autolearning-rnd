python upload_pipeline.py --package=autolearning-pipeline.tar.gz --host=669cafb082532673-dot-europe-west1.pipelines.googleusercontent.com

## build
docker build ./ -t wv-automl-pipeline-builder

## RUN
docker run --rm -it \
  -v ${PWD}:/src \
  wv-automl-pipeline-builder bash
