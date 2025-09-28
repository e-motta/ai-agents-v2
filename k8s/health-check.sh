#!/bin/bash

# Health check script for AI Agents Kubernetes deployments
set -e

echo "🏥 AI Agents Application Health Check"
echo "======================================"

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
echo ""

# Function to check resource status
check_resource() {
    local resource_type=$1
    local resource_name=$2
    local namespace=${3:-default}

    echo "🔍 Checking $resource_type: $resource_name"

    if kubectl get $resource_type $resource_name -n $namespace &> /dev/null; then
        local status=$(kubectl get $resource_type $resource_name -n $namespace -o jsonpath='{.status.phase}' 2>/dev/null || echo "Unknown")
        local ready=$(kubectl get $resource_type $resource_name -n $namespace -o jsonpath='{.status.readyReplicas}' 2>/dev/null || echo "N/A")
        local desired=$(kubectl get $resource_type $resource_name -n $namespace -o jsonpath='{.spec.replicas}' 2>/dev/null || echo "N/A")

        if [[ "$status" == "Running" ]] || [[ "$ready" == "$desired" ]] || [[ "$status" == "Bound" ]]; then
            echo "  ✅ Status: $status (Ready: $ready/$desired)"
        else
            echo "  ⚠️  Status: $status (Ready: $ready/$desired)"
        fi
    else
        echo "  ❌ Resource not found"
    fi
    echo ""
}

# Check ConfigMaps and Secrets
echo "📋 Configuration Resources"
echo "-------------------------"
check_resource "configmap" "app-config"
check_resource "configmap" "redis-config"
check_resource "secret" "openai-secret"
check_resource "secret" "redis-secret"
check_resource "secret" "backend-tls-secret"
check_resource "secret" "frontend-tls-secret"

# Check Redis
echo "📦 Redis Resources"
echo "------------------"
check_resource "statefulset" "redis-statefulset"
check_resource "service" "redis-service"
check_resource "pvc" "redis-data-redis-statefulset-0"

# Check Backend
echo "🔧 Backend Resources"
echo "-------------------"
check_resource "deployment" "backend-deployment"
check_resource "service" "backend-service"
check_resource "pvc" "backend-pvc"
check_resource "ingress" "backend-ingress"

# Check Frontend
echo "🌐 Frontend Resources"
echo "--------------------"
check_resource "deployment" "frontend-deployment"
check_resource "service" "frontend-service"
check_resource "ingress" "frontend-ingress"

# Check Pod Status
echo "🚀 Pod Status"
echo "-------------"
kubectl get pods -o wide

echo ""
echo "🔗 Service Endpoints"
echo "--------------------"
kubectl get endpoints

echo ""
echo "🌍 Ingress Status"
echo "-----------------"
kubectl get ingress

echo ""
echo "💾 Persistent Volumes"
echo "--------------------"
kubectl get pv,pvc

echo ""
echo "📊 Resource Usage"
echo "-----------------"
kubectl top pods 2>/dev/null || echo "Metrics server not available"

echo ""
echo "🔍 Recent Events"
echo "----------------"
kubectl get events --sort-by='.lastTimestamp' | tail -10

echo ""
echo "✅ Health check completed!"
echo ""
echo "💡 Troubleshooting Tips:"
echo "  - Check pod logs: kubectl logs <pod-name>"
echo "  - Describe resources: kubectl describe <resource-type> <resource-name>"
echo "  - Check service connectivity: kubectl port-forward svc/<service-name> <local-port>:<service-port>"