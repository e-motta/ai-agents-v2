#!/bin/sh

# Generate runtime configuration based on environment variables
cat > /usr/share/nginx/html/config.js << EOF
// Runtime configuration for the frontend
// This file is dynamically generated at container startup
window.APP_CONFIG = {
  API_BASE_URL: '${VITE_API_BASE_URL:-/api/v1}',
  BASE_PATH: '${DOCKER_COMPOSE:+}'
};
EOF

# Create a dynamic index.html that loads config.js from the correct path
if [ "${DOCKER_COMPOSE}" = "true" ]; then
    # For docker-compose, config.js is at root
    sed 's|src="/config.js"|src="/config.js"|g' /usr/share/nginx/html/index.html > /usr/share/nginx/html/index.html.tmp
else
    # For k8s, config.js is at /frontend/config.js
    sed 's|src="/config.js"|src="/frontend/config.js"|g' /usr/share/nginx/html/index.html > /usr/share/nginx/html/index.html.tmp
    # Also create the config.js in the frontend subdirectory for k8s
    mkdir -p /usr/share/nginx/html/frontend
    cp /usr/share/nginx/html/config.js /usr/share/nginx/html/frontend/config.js
fi
mv /usr/share/nginx/html/index.html.tmp /usr/share/nginx/html/index.html

# Generate nginx configuration based on environment
if [ "${DOCKER_COMPOSE}" = "true" ]; then
    # For docker-compose, use the default nginx config (root path)
    echo "Using docker-compose nginx configuration (root path)"
    cp /etc/nginx/conf.d/default.conf /etc/nginx/conf.d/default.conf.bak
else
    # For k8s, we need to handle /frontend/ prefix
    echo "Using k8s nginx configuration (/frontend/ prefix)"
    cat > /etc/nginx/conf.d/default.conf << 'NGINXEOF'
server {
    listen 80;
    server_name localhost;

    # Root directory for the static files
    root /usr/share/nginx/html;
    index index.html index.htm;

    # Enable gzip compression
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_proxied expired no-cache no-store private auth;
    gzip_types text/plain text/css text/xml text/javascript application/javascript application/xml+rss application/json;

    # Handle static assets with proper caching
    location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg|woff|woff2|ttf|eot)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
        try_files $uri =404;
    }

    # This is the crucial part for SPAs
    location / {
        try_files $uri $uri/ /index.html;
    }

    # Health check endpoint
    location /health {
        access_log off;
        return 200 "healthy\n";
        add_header Content-Type text/plain;
    }
}
NGINXEOF
fi

# Create nginx cache directories
mkdir -p /var/cache/nginx/client_temp /var/cache/nginx/proxy_temp /var/cache/nginx/fastcgi_temp /var/cache/nginx/uwsgi_temp /var/cache/nginx/scgi_temp

# Create a custom nginx configuration that runs as root
cat > /etc/nginx/nginx.conf << 'NGINXEOF'
user root;
worker_processes  auto;

error_log  /var/log/nginx/error.log notice;
pid        /var/run/nginx.pid;

events {
    worker_connections  1024;
}

http {
    include       /etc/nginx/mime.types;
    default_type  application/octet-stream;

    log_format  main  '$remote_addr - $remote_user [$time_local] "$request" '
                      '$status $body_bytes_sent "$http_referer" '
                      '"$http_user_agent" "$http_x_forwarded_for"';

    access_log  /var/log/nginx/access.log  main;

    sendfile        on;
    #tcp_nopush     on;

    keepalive_timeout  65;

    #gzip  on;

    include /etc/nginx/conf.d/*.conf;
}
NGINXEOF

# Start nginx
exec nginx -g "daemon off;"