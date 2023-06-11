FROM eu.gcr.io/wizydam-dev/object_detection/base:latest
WORKDIR /app

COPY requirements.txt /assets/requirements.txt
RUN pip install -r /assets/requirements.txt
# RUN pip install tensorflowjs[wizard]
# RUN pip install google-cloud-storage
# RUN pip install requests==2.23.0
# RUN pip install smart_open[gcp]

RUN pip uninstall opencv-python -y
USER root
RUN pip uninstall opencv-python -y
RUN pip install opencv-python-headless
USER tensorflow

COPY src /app
