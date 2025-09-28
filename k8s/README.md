# Kubernetes Manifests

This directory contains the essential Kubernetes manifests needed to deploy the AI Agents application.

## Prerequisites

Before deploying these manifests, ensure you have:

1. **Kubernetes Cluster** (v1.20+)
2. **kubectl** configured to access your cluster
3. **Docker Images** built and pushed to a registry:
   - `ai-agents-backend:latest`
   - `ai-agents-frontend:latest`
4. **Ingress Controller** (nginx-ingress recommended)

## Quick Start

```bash
# Deploy everything with one command
./deploy.sh
```

## Configuration

### Environment Variables

Update the following in the deployment files:

### Secrets

Update the secrets in `secrets.yaml`:

```bash
# Encode your OpenAI API key
echo -n "your_openai_api_key" | base64
```

## Storage

### Persistent Volumes

- **Backend**: 10Gi PVC for vector store data
- **Redis**: 5Gi PVC for Redis persistence

### Storage Classes

- Uses `standard` storage class (modify as needed)

## Health Checks

All deployments include:

- **Liveness probes** to detect unhealthy containers
- **Readiness probes** to ensure containers are ready to serve traffic
- **Resource limits** to prevent resource starvation

## Scaling

### Horizontal Scaling

```bash
# Scale backend
kubectl scale deployment backend-deployment --replicas=5

# Scale frontend
kubectl scale deployment frontend-deployment --replicas=3
```

### Vertical Scaling

Modify resource requests/limits in deployment files.

## Troubleshooting

### Common Issues

1. **Pod Startup Failures**

   ```bash
   kubectl describe pod <pod-name>
   kubectl logs <pod-name>
   ```

2. **Service Connectivity**

   ```bash
   kubectl get services
   kubectl get endpoints
   ```

3. **Ingress Issues**

   ```bash
   kubectl describe ingress
   kubectl get ingress
   ```

4. **Storage Problems**
   ```bash
   kubectl get pvc
   kubectl describe pvc <pvc-name>
   ```

### Health Checks

```bash
# Run comprehensive health check
./health-check.sh
```

This script will check:

- All resource statuses (pods, services, ingress, PVCs)
- Configuration and secrets
- Resource usage and events
- Service connectivity

```bash
# Check pod status
kubectl get pods

# Check service endpoints
kubectl get endpoints

# Check ingress status
kubectl get ingress
```

## Customization

### Resource Requirements

Modify `resources.requests` and `resources.limits` in deployment files.

### Environment Variables

Add or modify environment variables in deployment files.

### Storage

Adjust PVC sizes and storage classes as needed.

### Networking

Configure ingress and services as needed.

## Production Considerations

1. **High Availability**: Deploy across multiple nodes
2. **Backup**: Implement Redis backup strategy
3. **Monitoring**: Set up comprehensive monitoring and alerting
4. **Security**: Review security configurations
5. **Performance**: Tune resource limits based on usage
6. **Updates**: Implement rolling update strategies

## Cleanup

### Option 1: Simple Cleanup Script (Recommended)

```bash
# Remove everything with one command
./cleanup.sh
```
