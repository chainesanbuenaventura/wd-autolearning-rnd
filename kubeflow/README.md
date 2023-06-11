export PROJECT_ID="wizydam-dev"
export CLUSTER="cluster-1"
export ZONE="europe-west1-b"
# Configure kubectl to connect with the cluster
gcloud container clusters get-credentials "$CLUSTER" --zone "$ZONE" --project "$PROJECT_ID"

export SA_NAME=kubeflow-pipeline
export NAMESPACE=default

gcloud iam service-accounts create $SA_NAME \
  --display-name $SA_NAME --project "$PROJECT_ID"
