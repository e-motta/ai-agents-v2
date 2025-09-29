#!/bin/bash

# Complete cleanup script for AI Agents Kubernetes manifests
set -e

echo "🧹 Cleaning up AI Agents Application from Kubernetes..."

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

# Delete application-specific components first
# The --ignore-not-found=true flag prevents errors if the resource is already gone

echo "🌐 Removing Ingress..."
kubectl delete -f ingress-local.yaml --ignore-not-found=true

echo "🌐 Removing Frontend..."
kubectl delete -f frontend/service.yaml --ignore-not-found=true
kubectl delete -f frontend/deployment.yaml --ignore-not-found=true

echo "🔧 Removing Backend..."
kubectl delete -f backend/service.yaml --ignore-not-found=true
kubectl delete -f backend/deployment.yaml --ignore-not-found=true

echo "🔨 Removing index builder job..."
kubectl delete -f backend/index-builder-job.yaml --ignore-not-found=true

echo "💾 Removing PersistentVolumeClaim (PVC)..."
kubectl delete -f backend/pvc.yaml --ignore-not-found=true

echo "📦 Removing Redis..."
kubectl delete -f redis/service.yaml --ignore-not-found=true
kubectl delete -f redis/statefulset.yaml --ignore-not-found=true
kubectl delete -f redis/configmap.yaml --ignore-not-found=true

echo "⚙️ Removing Configuration and Secrets..."
kubectl delete -f configmap.yaml --ignore-not-found=true
kubectl delete -f secrets.yaml --ignore-not-found=true

# echo "🔧 Removing Ingress NGINX Controller..."
# # This removes the core Ingress controller installed by the deploy script
# kubectl delete -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.2.1/deploy/static/provider/cloud/deploy.yaml --ignore-not-found=true

echo ""
echo "✅ Cleanup completed successfully!"
echo "Note: The underlying Persistent Volume for the PVC might still exist depending on its Reclaim Policy."
echo ""
echo "🔍 To verify cleanup, the following commands should return 'No resources found':"
echo "  kubectl get pods -A"
echo "  kubectl get services"
echo "  kubectl get ingress"
echo "  kubectl get pvc"
echo "  kubectl get jobs"
echo "  kubectl get secrets"
echo "  kubectl get configmaps"