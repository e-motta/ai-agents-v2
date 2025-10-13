# AI Agents System

[![CI Pipeline](https://github.com/e-motta/ai-agents-v2/actions/workflows/ci.yml/badge.svg)](https://github.com/e-motta/ai-agents-v2/actions/workflows/ci.yml)

A multi-agent AI system built with FastAPI, React, and Redis, designed to handle mathematical computations and knowledge-based queries.

## 🏗️ Architecture Overview

The system consists of three main components working together:

### 1. **Router Agent** 🧭

- **Purpose**: Intelligent query classification and routing
- **Function**: Analyzes incoming user queries and routes them to appropriate specialized agents
- **Decision Logic**: Routes queries to agents: `MathAgent`, `KnowledgeAgent`; or handles `UnsupportedLanguage`, `Error`
- **Security**: Implements prompt injection detection and malicious content filtering

### 2. **Specialized Agents** 🤖

- **MathAgent**: Handles mathematical expressions, calculations, and numerical problems
- **KnowledgeAgent**: Provides InfinitePay service information from indexed documentation
- **Response Conversion**: Router Agent converts raw agent responses into conversational format

### 3. **Infrastructure Components** ⚙️

- **FastAPI Backend**: RESTful API with structured logging and error handling
- **React Frontend**: Modern chat interface with conversation management
- **Redis**: Conversation history storage and session management
- **Vector Store**: ChromaDB for knowledge agent document indexing

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose
- kubectl (for Kubernetes deployment)
- Poetry (for backend development)
- OpenAI API key

### Environment Setup

Create a `.env` file in the `backend/` directory:

```bash
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini

# Redis Configuration
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=
REDIS_CONVERSATION_TTL=86400
REDIS_SOCKET_CONNECT_TIMEOUT=5
REDIS_SOCKET_TIMEOUT=5

# Environment
ENVIRONMENT=production
```

## 🐳 Running Locally with Docker

### 1. Clone the project

```bash
# Clone the repository
git clone https://github.com/e-motta/ai-agents-v2.git
cd ai-agents-v2
```

### 2. Build the knowledge base’s vector index

```bash
# Run Docker Compose with build profile (should take around 5 minutes to finish)
docker compose --profile build run --build --rm build-index
```

### 3. Start all services

You can use either Docker Compose directly or the convenient Makefile commands:

#### Using Makefile (for development)

```bash
# Start all services with development configuration
make up

# Start all services and rebuild containers
make up-build

# Stop all services
make down

# Build containers without starting
make build
```

#### Using Docker Compose directly

```bash
# Start all services with Docker Compose
docker compose up -d --build

# Or use the development configuration with live reload
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d --build
```

### 4. Verify Services

```bash
# Check service status
docker-compose ps

# View logs
docker-compose logs -f backend
docker-compose logs -f frontend
docker-compose logs -f redis

# Or using Makefile
make logs service=backend
make logs service=frontend
make logs service=redis
```

### 5. Access the Application

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Redis**: localhost:6379

### 6. Health Checks

```bash
# Backend health
curl http://localhost:8000/health

# Frontend health
curl http://localhost:3000/health
```

## ☸️ Running on Kubernetes

### 1. Prerequisites

- Kubernetes cluster (local or cloud)
- kubectl configured
- Ingress NGINX controller

### 2. Environment Setup

Create a `secrets.yaml` file in `k8s` the directory:

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: openai-secret
  namespace: default
  labels:
    app: ai-agents-app
type: Opaque
data:
  # Base64 encoded API key - replace with actual API key
  # To encode: echo -n "your-api-key" | base64
  api-key: "your-api-key-in-base-64"
```

### 3. Deploy the Application

The deployment script takes care of all steps required for the deployment.

Note: the script will need to build the vector index on the first run, which should take around 5 minutes.

```bash
# Navigate to k8s directory
cd k8s

# Run the deployment script
chmod +x deploy.sh
./deploy.sh
```

### 4. Verify Deployment

```bash
# Check pod status
kubectl get pods

# Check services
kubectl get services

# Check ingress
kubectl get ingress

# View logs
kubectl logs -l app=backend
kubectl logs -l app=frontend
kubectl logs -l app=redis
```

### 5. Access the Application Frontend

- **Local**: http://localhost/frontend (configured via ingress)

## 🌐 Frontend Access and Testing

### Chat Interface Features

- **Multi-conversation Support**: Create and manage multiple conversations
- **Real-time Chat**: Interactive chat interface with the AI agents
- **Conversation History**: View and continue previous conversations
- **User Management**: Automatic user ID generation and management

### Testing Multiple Conversations

1. **Access the Frontend**: Navigate to http://localhost:3000 (Docker) or http://localhost/frontend (k8s)
2. **Create New Conversations**: Click "Nova Conversa" to start new conversations
3. **Switch Between Conversations**: Use the sidebar to switch between different conversations
4. **Test Different Query Types**:
   - **Math Queries**: "What is 15 \* 3?", "Calculate the square root of 16"
   - **Knowledge Queries**: "What are the transaction fees?", "How does PIX work?"
   - **Mixed Queries**: Test the router's decision-making capabilities

### Example Test Scenarios

#### Math Agent Testing

```
User: "Quanto é 15*3?"
Expected: Router → MathAgent → "45"

User: "Calculate (100/5)+2"
Expected: Router → MathAgent → "22"
```

#### Knowledge Agent Testing

```
User: "Quais as taxas da maquininha?"
Expected: Router → KnowledgeAgent → InfinitePay fee information

User: "Como funciona o pagamento por PIX?"
Expected: Router → KnowledgeAgent → PIX payment process information
```

#### Router Testing

```
User: "Hello, how are you?" (English)
Expected: Router → KnowledgeAgent (general knowledge)

User: "こんにちは" (Japanese)
Expected: Router → UnsupportedLanguage
```

## 📊 Example Logs (JSON Format)

The system uses structured JSON logging.

### Example of the entire flow for Knowledge Agent, from request to response:

```json
{
  "conversation_id": "c44u4w03dx",
  "user_id": "u5iu1zr123",
  "message_preview": "Quais são as taxas da maquininha?",
  "event": "Chat request received",
  "timestamp": "2025-10-13T10:51:51.872528+00:00",
  "agent": "System",
  "level": "info",
  "logger": "app.api.v1.chat"
}
{
  "conversation_id": "c44u4w03dx",
  "user_id": "u5iu1zr123",
  "query_preview": "Quais são as taxas da maquininha?",
  "event": "Routing query",
  "timestamp": "2025-10-13T10:51:51.873861+00:00",
  "agent": "RouterAgent",
  "level": "info",
  "logger": "app.agents.router_agent"
}
{
  "conversation_id": "c44u4w03dx",
  "user_id": "u5iu1zr123",
  "decision": "KnowledgeAgent",
  "query_preview": "Quais são as taxas da maquininha?",
  "event": "Agent decision made",
  "timestamp": "2025-10-13T10:51:53.167338+00:00",
  "agent": "RouterAgent",
  "level": "info",
  "logger": "app.agents.router_agent"
}
{
  "conversation_id": "c44u4w03dx",
  "user_id": "u5iu1zr123",
  "processed_content": "KnowledgeAgent",
  "execution_time": "1.294s",
  "agent_name": "RouterAgent",
  "query_preview": "Quais são as taxas da maquininha?",
  "event": "Agent processing completed",
  "timestamp": "2025-10-13T10:51:53.167651+00:00",
  "agent": "System",
  "level": "info",
  "logger": "app.services.chat_dispatcher"
}
{
  "query": "Quais são as taxas da maquininha?",
  "query_preview": "Quais são as taxas da maquininha?",
  "event": "Starting knowledge base query",
  "timestamp": "2025-10-13T10:51:53.168194+00:00",
  "agent": "KnowledgeAgent",
  "level": "info",
  "logger": "app.agents.knowledge_agent.main"
}
{
  "query": "Quais são as taxas da maquininha?",
  "answer_preview": "As taxas da maquininha variam conforme o plano de recebimento e o produto utilizado. Elas são calcul",
  "sources": [
    {
      "url": "https://ajuda.infinitepay.io/pt-BR/articles/6038283-como-usar-a-calculadora-de-taxas-na-maquininha-smart",
      "source": "infinitepay_help_center",
      "score": 0.49031518308296335
    },
    {
      "url": "https://ajuda.infinitepay.io/pt-BR/articles/3567351-quais-modelos-de-maquinas-de-cartao-posso-comprar",
      "source": "infinitepay_help_center",
      "score": 0.4612320147210406
    },
    {
      "url": "https://ajuda.infinitepay.io/pt-BR/articles/3359956-quais-sao-as-taxas-da-infinitepay",
      "source": "infinitepay_help_center",
      "score": 0.4611695581021876
    },
    {
      "url": "https://ajuda.infinitepay.io/pt-BR/articles/9455289-como-obter-taxas-ainda-mais-baixas",
      "source": "infinitepay_help_center",
      "score": 0.46057754780834537
    },
    {
      "url": "https://ajuda.infinitepay.io/pt-BR/articles/4844397-o-que-vem-com-a-maquininha-smart",
      "source": "infinitepay_help_center",
      "score": 0.45545958181531576
    }
  ],
  "event": "Knowledge base query completed",
  "timestamp": "2025-10-13T10:51:57.123465+00:00",
  "agent": "KnowledgeAgent",
  "level": "info",
  "logger": "app.agents.knowledge_agent.main"
}
{
  "conversation_id": "c44u4w03dx",
  "user_id": "u5iu1zr123",
  "processed_content": "As taxas da maquininha variam conforme o plano de recebimento e o produto utilizado. Elas são calculadas com base nas formas de pagamento aceitas e nos valores aplicados às transações. Além disso, as taxas podem ser reduzidas de acordo com o faturamento mensal do cliente na InfinitePay.",
  "execution_time": "3.956s",
  "agent_name": "KnowledgeAgent",
  "query_preview": "Quais são as taxas da maquininha?",
  "event": "Agent processing completed",
  "timestamp": "2025-10-13T10:51:57.124394+00:00",
  "agent": "System",
  "level": "info",
  "logger": "app.services.chat_dispatcher"
}
{
  "conversation_id": "c44u4w03dx",
  "user_id": "u5iu1zr123",
  "processed_content": "As taxas da maquininha variam conforme o plano de recebimento e o produto utilizado. Elas são calculadas com base nas formas de pagamento aceitas e nos valores aplicados às transações. Além disso, as taxas podem ser reduzidas de acordo com o faturamento mensal do cliente na InfinitePay.",
  "execution_time": "0.000s",
  "agent_name": "RouterAgent",
  "query_preview": "Quais são as taxas da maquininha?",
  "event": "Agent processing completed",
  "timestamp": "2025-10-13T10:51:57.124934+00:00",
  "agent": "System",
  "level": "info",
  "logger": "app.services.chat_dispatcher"
}
{
  "conversation_id": "c44u4w03dx",
  "user_id": "u5iu1zr123",
  "router_decision": "KnowledgeAgent",
  "execution_time": 5.253060340881348,
  "response_preview": "As taxas da maquininha variam conforme o plano de recebimento e o produto utilizado. Elas são calcul",
  "workflow_history": [
    {
      "agent": "RouterAgent",
      "action": "_route_query",
      "result": "KnowledgeAgent"
    },
    {
      "agent": "KnowledgeAgent",
      "action": "_process_knowledge",
      "result": "As taxas da maquininha variam conforme o plano de recebimento e o produto utilizado. Elas são calculadas com base nas formas de pagamento aceitas e nos valores aplicados às transações. Além disso, as taxas podem ser reduzidas de acordo com o faturamento mensal do cliente na InfinitePay."
    },
    {
      "agent": "RouterAgent",
      "action": "_convert_response",
      "result": "As taxas da maquininha variam conforme o plano de recebimento e o produto utilizado. Elas são calculadas com base nas formas de pagamento aceitas e nos valores aplicados às transações. Além disso, as taxas podem ser reduzidas de acordo com o faturamento mensal do cliente na InfinitePay."
    }
  ],
  "event": "Chat request completed",
  "timestamp": "2025-10-13T10:51:57.125403+00:00",
  "agent": "System",
  "level": "info",
  "logger": "app.api.v1.chat"
}
{
  "event": "Added message to conversation c44u4w03dx",
  "timestamp": "2025-10-13T10:51:57.134170+00:00",
  "agent": "System",
  "level": "info",
  "logger": "app.services.redis_service"
}
{
  "conversation_id": "c44u4w03dx",
  "user_id": "u5iu1zr123",
  "event": "Conversation saved to Redis",
  "timestamp": "2025-10-13T10:51:57.134448+00:00",
  "agent": "System",
  "level": "info",
  "logger": "app.api.v1.chat"
}
```

### Example of the entire flow for Math Agent, from request to response:

```json
{
  "conversation_id": "c44u4w03dx",
  "user_id": "u5iu1zr123",
  "message_preview": "Quanto é 3*4?",
  "event": "Chat request received",
  "timestamp": "2025-10-13T10:59:54.688572+00:00",
  "agent": "System",
  "level": "info",
  "logger": "app.api.v1.chat"
}
{
  "conversation_id": "c44u4w03dx",
  "user_id": "u5iu1zr123",
  "query_preview": "Quanto é 3*4?",
  "event": "Routing query",
  "timestamp": "2025-10-13T10:59:54.690766+00:00",
  "agent": "RouterAgent",
  "level": "info",
  "logger": "app.agents.router_agent"
}
{
  "conversation_id": "c44u4w03dx",
  "user_id": "u5iu1zr123",
  "decision": "MathAgent",
  "query_preview": "Quanto é 3*4?",
  "event": "Agent decision made",
  "timestamp": "2025-10-13T10:59:55.344539+00:00",
  "agent": "RouterAgent",
  "level": "info",
  "logger": "app.agents.router_agent"
}
{
  "conversation_id": "c44u4w03dx",
  "user_id": "u5iu1zr123",
  "processed_content": "MathAgent",
  "execution_time": "0.655s",
  "agent_name": "RouterAgent",
  "query_preview": "Quanto é 3*4?",
  "event": "Agent processing completed",
  "timestamp": "2025-10-13T10:59:55.344985+00:00",
  "agent": "System",
  "level": "info",
  "logger": "app.services.chat_dispatcher"
}
{
  "query": "Quanto é 3*4?",
  "event": "Starting math evaluation",
  "timestamp": "2025-10-13T10:59:55.345463+00:00",
  "agent": "MathAgent",
  "level": "info",
  "logger": "app.agents.math_agent"
}
{
  "query": "Quanto é 3*4?",
  "result": "12",
  "event": "Math evaluation completed",
  "timestamp": "2025-10-13T10:59:55.743570+00:00",
  "agent": "MathAgent",
  "level": "info",
  "logger": "app.agents.math_agent"
}
{
  "conversation_id": "c44u4w03dx",
  "user_id": "u5iu1zr123",
  "processed_content": "12",
  "execution_time": "0.399s",
  "agent_name": "MathAgent",
  "query_preview": "Quanto é 3*4?",
  "event": "Agent processing completed",
  "timestamp": "2025-10-13T10:59:55.743903+00:00",
  "agent": "System",
  "level": "info",
  "logger": "app.services.chat_dispatcher"
}
{
  "agent_type": "MathAgent",
  "response_preview": "12",
  "query_preview": "Quanto é 3*4?",
  "event": "Starting response conversion",
  "timestamp": "2025-10-13T10:59:55.744162+00:00",
  "agent": "RouterAgent",
  "level": "info",
  "logger": "app.agents.router_agent"
}
{
  "agent_type": "MathAgent",
  "original_response_preview": "12",
  "converted_response_preview": "3 * 4 equals 12.",
  "event": "Response conversion completed",
  "timestamp": "2025-10-13T10:59:56.463257+00:00",
  "agent": "RouterAgent",
  "level": "info",
  "logger": "app.agents.router_agent"
}
{
  "conversation_id": "c44u4w03dx",
  "user_id": "u5iu1zr123",
  "processed_content": "3 * 4 é igual a 12.",
  "execution_time": "0.719s",
  "agent_name": "RouterAgent",
  "query_preview": "Quanto é 3*4?",
  "event": "Agent processing completed",
  "timestamp": "2025-10-13T10:59:56.463552+00:00",
  "agent": "System",
  "level": "info",
  "logger": "app.services.chat_dispatcher"
}
{
  "conversation_id": "c44u4w03dx",
  "user_id": "u5iu1zr123",
  "router_decision": "MathAgent",
  "execution_time": 1.7753267288208008,
  "response_preview": "3 * 4 é igual a 12.",
  "workflow_history": [
    {
      "agent": "RouterAgent",
      "action": "_route_query",
      "result": "MathAgent"
    },
    {
      "agent": "MathAgent",
      "action": "_process_math",
      "result": "12"
    },
    {
      "agent": "RouterAgent",
      "action": "_convert_response",
      "result": "3 * 4 equals 12."
    }
  ],
  "event": "Chat request completed",
  "timestamp": "2025-10-13T10:59:56.463772+00:00",
  "agent": "System",
  "level": "info",
  "logger": "app.api.v1.chat"
}
{
  "event": "Added message to conversation c44u4w03dx",
  "timestamp": "2025-10-13T10:59:56.470590+00:00",
  "agent": "System",
  "level": "info",
  "logger": "app.services.redis_service"
}
{
  "conversation_id": "c44u4w03dx",
  "user_id": "u5iu1zr123",
  "event": "Conversation saved to Redis",
  "timestamp": "2025-10-13T10:59:56.470868+00:00",
  "agent": "System",
  "level": "info",
  "logger": "app.api.v1.chat"
}
```

## 🔒 Security: Sanitization and Prompt Injection Protection

### Input Sanitization

The system implements multiple layers of security:

#### 1. **Bleach-based HTML Sanitization**

```python
# Located in: backend/app/security/sanitization.py
def sanitize_user_input(text: str) -> str:
    allowed_tags = ["b", "i", "u", "em", "strong", "p", "br", "span"]
    allowed_attributes = {"span": ["class"], "p": ["class"]}

    sanitized = bleach.clean(
        text,
        tags=allowed_tags,
        attributes=allowed_attributes,
        strip=True
    )
    return sanitized
```

#### 2. **Prompt Injection Detection**

The Router Agent includes comprehensive prompt injection detection:

```python
# Located in: backend/app/security/constants.py and backend/app/agents/router_agent.py
SUSPICIOUS_PATTERNS = [
    "ignore previous instructions",
    "forget everything",
    "system prompt",
    "you are now",
    "act as",
    "pretend to be",
    "roleplay",
    "jailbreak",
    "developer mode",
    "admin mode",
    "override",
    "bypass",
    "exploit",
    "hack",
    "inject",
    "execute",
    "run command",
    "system call",
    "file://",
    "http://",
    "https://",
    "<script>",
    "javascript:",
    "data:",
    "eval(",
    "exec(",
    "import os",
    "subprocess",
    "shell",
    "terminal",
    "command line",
    "prompt injection",
    "llm injection",
    # Portuguese patterns
    "ignore as instruções anteriores",
    "esqueça tudo",
    "prompt do sistema",
    "você agora é",
    "aja como",
    "finja ser",
    "interprete o papel de",
]
```

#### 3. **Security Response Protocol**

When suspicious content is detected:

- **Logs the attempt** with structured logging
- **Routes to KnowledgeAgent** for safety (instead of rejecting)
- **Preserves conversation flow** while maintaining security
- **Tracks patterns** for security monitoring

#### 4. **Language Restriction**

- **Supported Languages**: English and Portuguese only
- **Unsupported Language Detection**: Routes to `UnsupportedLanguage` response
- **Character Validation**: Blocks non-Latin characters (except mathematical symbols)

#### 5. **Agent-Specific Security**

- **MathAgent**: Only processes mathematical expressions, blocks code execution
- **KnowledgeAgent**: Only accesses indexed documentation, no external data access
- **RouterAgent**: Classification-only, no content generation or system access

## 🧪 Running Tests

### Test Structure

The project includes comprehensive test suites:

- **Unit Tests**: Individual agent functionality
- **Integration Tests**: API endpoint testing

### Running Tests

#### 1. **All Tests**

```bash
# Using Poetry (recommended)
cd backend
poetry run python run_tests.py --type all

# Or activate Poetry shell first
poetry shell
cd backend
python run_tests.py --type all
```

#### 2. **Specific Test Suites**

```bash
# Router Agent tests only
cd backend
poetry run python run_tests.py --type router

# Math Agent tests only
poetry run python run_tests.py --type math

# Chat API tests only
poetry run python run_tests.py --type chat

# Unit tests (Router + Math)
poetry run python run_tests.py --type unit
```

#### 3. **With Coverage Reports**

```bash
# Terminal coverage report
cd backend
poetry run python run_tests.py --type coverage

# HTML coverage report
poetry run python run_tests.py --type coverage-html

# Comprehensive reports (HTML + XML)
poetry run python run_tests.py --type coverage-report
```

#### 4. **Verbose Output**

```bash
# Verbose test output
cd backend
poetry run python run_tests.py --type all --verbose

# Suppress warnings
poetry run python run_tests.py --type all --no-warnings
```

#### 5. **Coverage Thresholds**

```bash
# Set minimum coverage threshold
cd backend
poetry run python run_tests.py --type coverage --coverage-threshold 85

# Fail if coverage below threshold
poetry run python run_tests.py --type coverage --coverage-fail-under 80
```

## 🔧 Development

### Poetry Commands

This project uses Poetry for dependency management. Here are the most common commands:

```bash
# Install Poetry (if not already installed)
curl -sSL https://install.python-poetry.org | python3 -

# Install all dependencies (production, dev)
poetry install --with dev

# Install only production dependencies
poetry install --only=main

# Add a new dependency
poetry add package-name

# Add a development dependency
poetry add --group dev package-name

# Update dependencies
poetry update

# Show dependency tree
poetry show --tree

# Activate virtual environment
poetry shell

# Run commands in virtual environment without activating shell
poetry run command

# Export requirements.txt (if needed for legacy systems)
poetry export -f requirements.txt --output requirements.txt --without-hashes
```

### Backend Development

The backend uses Poetry for dependency management:

```bash
# Install Poetry (if not already installed)
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install --with dev

# Run development server
poetry run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Run tests
poetry run python run_tests.py --type all

# Run linting and formatting
poetry run ruff check .
poetry run ruff format .
poetry run mypy .
```

### Frontend Development

```bash
cd frontend

# Install dependencies
npm install

# Run development server
npm run dev
```

### Redis Development

```bash
# Start Redis locally
docker run -d -p 6379:6379 redis:7.2-alpine

# Connect to Redis CLI
redis-cli -h localhost -p 6379
```

## 📁 Project Structure

```
ai-agents-v2/
├── backend/                # FastAPI backend (Poetry-managed)
│   ├── app/
│   │   ├── agents/         # AI agents (Router, Math, Knowledge)
│   │   ├── api/
│   │   ├── core/           # Core functionality (logging, settings, LLM)
│   │   ├── security/       # Security and sanitization
│   │   └── services/       # External and internal services (Redis, LLM client, Chat Dispatcher)
│   ├── tests/
│   ├── scripts/
│   ├── vector_store/
│   ├── Dockerfile
│   ├── pyproject.toml
│   ├── poetry.lock
│   └── run_tests.py        # Test runner script
├── frontend/               # React frontend
│   ├── src/                # React components and services
│   ├── Dockerfile
│   └── package.json
├── k8s/                    # Kubernetes manifests
│   ├── backend/
│   ├── frontend/
│   ├── redis/
│   └── deploy.sh
├── docker-compose.yml
├── docker-compose.dev.yml  # Development overrides
├── Makefile                # Docker deployment commands
└── README.md
```

## 🚨 Troubleshooting

### Common Issues

#### 1. **Redis Connection Issues**

```bash
# Check Redis status
make logs service=redis
# or
docker-compose logs redis

# Test Redis connection
redis-cli -h localhost -p 6379 ping
```

#### 2. **OpenAI API Issues**

```bash
# Verify API key in environment
echo $OPENAI_API_KEY

# Check API key validity
curl -H "Authorization: Bearer $OPENAI_API_KEY" https://api.openai.com/v1/models
```

#### 3. **Frontend Not Loading**

```bash
# Check frontend container
make logs service=frontend
# or
docker-compose logs frontend

# Verify frontend health
curl http://localhost:3000/health
```

#### 4. **Backend API Issues**

```bash
# Check backend logs
make logs service=backend
# or
docker-compose logs backend

# Test API health
curl http://localhost:8000/health

# Check API documentation
open http://localhost:8000/docs
```

### Kubernetes Issues

#### 1. **Pod Not Starting**

```bash
# Check pod status
kubectl get pods

# Describe pod for details
kubectl describe pod <pod-name>

# Check pod logs
kubectl logs <pod-name>
```

#### 2. **Service Not Accessible**

```bash
# Check service status
kubectl get services

# Check ingress status
kubectl get ingress

# Test service connectivity
kubectl port-forward service/backend-service 8000:8000
```

## 📝 API Documentation

### Chat Endpoint

```http
POST /api/v1/chat
Content-Type: application/json

{
  "conversation_id": "conv_abc123",
  "user_id": "user_xyz789",
  "message": "What is 15 * 3?"
}
```

### Response Format

```json
{
  "user_id": "user_xyz789",
  "conversation_id": "conv_abc123",
  "router_decision": "MathAgent",
  "response": "15 * 3 equals 45.",
  "source_agent_response": "45",
  "workflow_history": [
    {
      "agent": "RouterAgent",
      "action": "route_query",
      "result": "MathAgent"
    },
    {
      "agent": "MathAgent",
      "action": "solve_math",
      "result": "45"
    }
  ]
}
```

### Conversation History

```http
GET /api/v1/chat/history/{conversation_id}
```

### User Conversations

```http
GET /api/v1/chat/user/{user_id}/conversations
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
