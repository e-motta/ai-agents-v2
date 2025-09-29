# ---- Builder Stage ----
# Consolidate dependency installation and building into one stage
FROM node:24.9.0-alpine3.21 AS builder
WORKDIR /app

# Copy package files and install dependencies
COPY package*.json ./
RUN npm ci

# Copy the rest of the source code
COPY . .

# Accept build argument for docker-compose
ARG DOCKER_COMPOSE=false
ENV DOCKER_COMPOSE=$DOCKER_COMPOSE

# Build the application
RUN npm run build

# ---- Production Stage ----
# Use a lean nginx server for the final image
FROM nginx:1.25-alpine AS runner

# Copy only the built assets from the builder stage
COPY --from=builder /app/dist /usr/share/nginx/html

# Copy custom nginx config and startup script if necessary
# COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY start.sh /start.sh
RUN chmod +x /start.sh

# Create a simple health check endpoint
RUN echo "OK" > /usr/share/nginx/html/health

EXPOSE 80

# Start nginx
CMD ["/start.sh"]