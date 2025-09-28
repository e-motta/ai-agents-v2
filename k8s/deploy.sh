#!/bin/bash

# Simple deployment script for AI Agents Kubernetes manifests
set -e

echo "🚀 Deploying AI Agents Application to Kubernetes..."

# Check if kubectl is available
if ! command -v kubectl &> /dev/null; then
    echo "❌ kubectl is not installed or not in PATH"
    exit 1
fi

# Check if kubectl can connect to cluster
if ! kubectl cluster-info &> /dev/null; then
    echo "❌ Cannot connect to Kubernetes cluster"
    exit 1
fi

echo "✅ Kubernetes cluster connection verified"

# Deploy Ingress NGINX
echo "🔧 Deploying Ingress NGINX..."
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.2.1/deploy/static/provider/cloud/deploy.yaml

# Note: Secrets are now managed via secrets.yaml file
echo "📋 Using secrets from secrets.yaml file"

# Deploy ConfigMap and Secrets
echo "⚙️ Deploying Configuration and Secrets..."
kubectl apply -f configmap.yaml
kubectl apply -f secrets.yaml

# Create Docker images
# IMPORTANT: This step assumes a local Kubernetes environment (e.g., Minikube, Docker Desktop)
# where the cluster can access locally built images. For other environments,
# you must push these images to a container registry and update the Kubernetes manifests.
docker build -t ai-agents-backend:latest ../backend
docker build -t ai-agents-frontend:latest ../frontend

# Deploy PVC
echo "📦 Deploying PVC..."
kubectl apply -f backend/pvc.yaml

# Wait for the PVC to be provisioned and bound
echo "⏳ Waiting for PVC to be ready..."
kubectl wait --for=jsonpath='{.status.phase}'=Bound pvc/backend-pvc --timeout=300s

# Build the vector index using a Job
echo "🔨 Building vector index..."
kubectl apply -f backend/index-builder-job.yaml

# Wait for the index builder job to complete
echo "⏳ Waiting for index builder job to complete..."
kubectl wait --for=condition=complete job/index-builder-job --timeout=600s

# Check if the job succeeded
if kubectl get job index-builder-job -o jsonpath='{.status.conditions[?(@.type=="Complete")].status}' | grep -q "True"; then
    echo "✅ Index builder job completed successfully"
    echo "📊 Index contains $(kubectl logs -l app=index-builder | grep 'documents_count=' | tail -1 | sed 's/.*documents_count=\([0-9]*\).*/\1/') documents"
else
    echo "⚠️  Index builder job failed, but continuing with deployment"
    echo "   The index will be built on-demand when needed"
fi

# Deploy Redis (other services depend on it)
echo "📦 Deploying Redis..."
kubectl apply -f redis/configmap.yaml
kubectl apply -f redis/statefulset.yaml
kubectl apply -f redis/service.yaml

# Wait for Redis to be ready
echo "⏳ Waiting for Redis to be ready..."
kubectl wait --for=condition=ready pod -l app=redis --timeout=300s

# Deploy Backend
echo "🔧 Deploying Backend..."
kubectl apply -f backend/deployment.yaml
kubectl apply -f backend/service.yaml

# Deploy Frontend
echo "🌐 Deploying Frontend..."
kubectl apply -f frontend/deployment.yaml
kubectl apply -f frontend/service.yaml

# Wait for deployments to be ready
echo "⏳ Waiting for deployments to be ready..."
kubectl wait --for=condition=available deployment/backend-deployment --timeout=300s
kubectl wait --for=condition=available deployment/frontend-deployment --timeout=300s

# Deploy Ingress
echo "🌐 Deploying Ingress..."
kubectl apply -f ingress-local.yaml

echo "✅ Deployment completed successfully!"
echo ""
echo "📋 Deployment Summary:"
echo "  - Redis StatefulSet: $(kubectl get statefulset redis-statefulset -o jsonpath='{.status.readyReplicas}')/$(kubectl get statefulset redis-statefulset -o jsonpath='{.spec.replicas}') ready"
echo "  - Backend Deployment: $(kubectl get deployment backend-deployment -o jsonpath='{.status.readyReplicas}')/$(kubectl get deployment backend-deployment -o jsonpath='{.spec.replicas}') ready"
echo "  - Frontend Deployment: $(kubectl get deployment frontend-deployment -o jsonpath='{.status.readyReplicas}')/$(kubectl get deployment frontend-deployment -o jsonpath='{.spec.replicas}') ready"
echo ""
echo "🔍 To check status:"
echo "  kubectl get pods"
echo "  kubectl get services"
echo "  kubectl get ingress"
echo ""
echo "📝 To view logs:"
echo "  kubectl logs -l app=backend"
echo "  kubectl logs -l app=frontend"
echo "  kubectl logs -l app=redis"
echo "  kubectl logs -l app=index-builder  # Index builder job logs"
echo ""
echo "🔐 To verify secrets:"
echo "  kubectl get secrets"
echo "  kubectl describe secret openai-secret"
echo "  kubectl describe secret redis-secret"
echo ""
echo "📝 To update secrets:"
echo "  Edit k8s/secrets.yaml with base64-encoded values"
echo "  Run: kubectl apply -f secrets.yaml"
echo ""
echo "🔨 To rebuild the index:"
echo "  kubectl delete job index-builder-job"
echo "  kubectl apply -f backend/index-builder-job.yaml"
echo "  kubectl wait --for=condition=complete job/index-builder-job --timeout=600s"