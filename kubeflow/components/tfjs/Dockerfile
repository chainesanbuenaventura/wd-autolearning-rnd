FROM tensorflow/tensorflow:2.2.0-gpu

RUN pip install tensorflowjs[wizard]
RUN pip install google-cloud-storage
RUN pip install requests==2.23.0
RUN pip install smart_open[gcp]
COPY src /app
