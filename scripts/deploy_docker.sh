#!/bin/bash

# Create fresh dockerfile
mlflow models generate-dockerfile --model-uri models:/Embedder_NN@champion --env-manager local -d api

# Build the image
cd api
docker build . -t mlops_project

# Stop existing container (if any)
docker stop mlops-project || true
docker container prune -f

# Start a new container
docker run -d -p 5152:8080 --name mlops-project mlops_project &

# Push image to dockerhub
docker tag mlops_project vbazilevich/mlops_project
docker push vbazilevich/mlops_project:latest