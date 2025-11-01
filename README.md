# Iris Classification Model Deployment on GKE

## Overview
This project demonstrates deploying a machine learning model (Iris classifier) to Google Kubernetes Engine (GKE) using:
- Docker container
- Kubernetes Deployment & Service
- MLflow model registry
- LoadBalancer exposed API endpoint

## Architecture
Flow:
MLFlow (model registry) → FastAPI Model API (Docker) → Kubernetes Deployment → LoadBalancer Service → External public URL

## Kubernetes Concepts Explained

### Docker Container
Docker container is a single isolated runtime instance of an application — example:
```
docker run my-image
```
This starts one container process.

### Kubernetes Pod
Pod is the smallest deployable unit in Kubernetes. Kubernetes does NOT run containers directly — it schedules **Pods**.

Pod wraps 1 or more containers (usually 1):

| Docker Container | Kubernetes Pod |
|------------------|----------------|
| Single container runtime | Group of 1+ containers |
| Created by Docker Engine | Created & managed by K8s |
| No built-in orchestration | Kubernetes manages scaling/restart |

**Think: Pod = container + IP + restart policy + metadata**

## Deployment Steps (Short)

### 1. Build docker image
```
docker build -t gcr.io/<project>/iris-api:v1 .
docker push gcr.io/<project>/iris-api:v1
```

### 2. Apply Kubernetes manifests
```
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
```

### 3. Check pods + service
```
kubectl get pods
kubectl get svc
```

### 4. Test Prediction
```
curl "http://<EXTERNAL_IP>/predict?sepal_length=5.1&sepal_width=3.5&petal_length=1.4&petal_width=0.2"
```

Example expected output:
```
{"prediction":0}
```

## Final Deliverables in Screencast
1. Show GKE cluster creation
2. Show deployment manifest + image
3. Show service external IP
4. Show working `curl` result
5. Explain Pod vs Container

This completes the assignment.
