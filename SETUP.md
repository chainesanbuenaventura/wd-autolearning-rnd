# copy static codes
`export AUTO_ML_BUCKET=gs://wizyvision-dev-automl` for dev
`export AUTO_ML_BUCKET=gs://wizyvision-automl` for prod
on models/research
gsutil cp dist/object_detection-0.1.tar.gz $AUTO_ML_BUCKET/static/codes/object_detection/dist/
gsutil cp /tmp/pycocotools/pycocotools-2.0.tar.gz $AUTO_ML_BUCKET/static/codes/object_detection/pycocotools/
gsutil cp slim/dist/slim-0.1.tar.gz $AUTO_ML_BUCKET/static/codes/object_detection/slim/dist/

# copy pipeline config templates
gsutil cp -r templates/ $AUTO_ML_BUCKET/static/pipeline/templates/


# copy from dev to prod
gsutil cp -r gs://wizyvision-dev-automl/static/codes/* gs://wizyvision-automl/static/codes/*
