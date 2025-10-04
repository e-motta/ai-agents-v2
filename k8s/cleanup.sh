#!/bin/bash

# Complete cleanup script for AI Agents Kubernetes manifests
set -e

# Parse command line arguments
REMOVE_PVC=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --remove-pvc)
            REMOVE_PVC=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --remove-pvc    Remove PersistentVolumeClaim (PVC) during cleanup"
            echo "  -h, --help      Show this help message"
            echo ""
            echo "By default, PVCs are NOT removed to preserve data."
            echo "Use --remove-pvc flag to explicitly remove PVCs."
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

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

# Conditionally remove PVC based on command line option
if [ "$REMOVE_PVC" = true ]; then
    echo "💾 Removing PersistentVolumeClaim (PVC)..."
    kubectl delete -f backend/pvc.yaml --ignore-not-found=true
else
    echo "💾 Skipping PersistentVolumeClaim (PVC) removal (use --remove-pvc to remove)"
fi

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

if [ "$REMOVE_PVC" = true ]; then
    echo "Note: PVC has been removed. The underlying Persistent Volume might still exist depending on its Reclaim Policy."
else
    echo "Note: PVC has been preserved. Use --remove-pvc flag to remove it if needed."
fi

echo ""
echo "🔍 To verify cleanup, the following commands should return 'No resources found':"
echo "  kubectl get pods -A"
echo "  kubectl get services"
echo "  kubectl get ingress"
if [ "$REMOVE_PVC" = true ]; then
    echo "  kubectl get pvc"
else
    echo "  kubectl get pvc  # (PVC may still exist)"
fi
echo "  kubectl get jobs"
echo "  kubectl get secrets"
echo "  kubectl get configmaps"