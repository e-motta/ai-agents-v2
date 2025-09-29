# AI Agents System

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
git clone https://github.com/e-motta/ai_agents.git
cd ai_agents
```

### 2. Build the knowledge base’s vector index

```bash
# Run Docker Compose with build profile (should take around 5 minutes to finish)
docker compose --profile build run --build --rm build-index
```

### 3. Start all services

```bash
# Start all services with Docker Compose
docker compose up -d --build
```

### 4. Verify Services

```bash
# Check service status
docker-compose ps

# View logs
docker-compose logs -f backend
docker-compose logs -f frontend
docker-compose logs -f redis
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

The system uses structured JSON logging with the following format:

### Router Agent Decision Log

```json
{
  "timestamp": "2024-01-15T10:30:45.123Z",
  "level": "info",
  "agent": "RouterAgent",
  "conversation_id": "conv_abc123",
  "user_id": "user_xyz789",
  "decision": "MathAgent",
  "execution_time": "0.245s",
  "query_preview": "What is 15 * 3?",
  "event": "Agent decision made"
}
```

### Math Agent Processing Log

```json
{
  "timestamp": "2024-01-15T10:30:45.456Z",
  "level": "info",
  "agent": "MathAgent",
  "conversation_id": "conv_abc123",
  "user_id": "user_xyz789",
  "processed_content": "45",
  "execution_time": "0.123s",
  "event": "Agent processing completed"
}
```

### Knowledge Agent Processing Log

```json
{
  "timestamp": "2024-01-15T10:30:45.789Z",
  "level": "info",
  "agent": "KnowledgeAgent",
  "conversation_id": "conv_def456",
  "user_id": "user_xyz789",
  "processed_content": "According to the documentation, InfinitePay offers three types of maquininhas...",
  "execution_time": "1.234s",
  "event": "Agent processing completed"
}
```

### System Event Log

```json
{
  "timestamp": "2024-01-15T10:30:46.012Z",
  "level": "info",
  "agent": "System",
  "conversation_id": "conv_abc123",
  "user_id": "user_xyz789",
  "event": "Chat request completed",
  "router_decision": "MathAgent",
  "execution_time": "0.456s",
  "response_preview": "15 * 3 equals 45."
}
```

### Error Log

```json
{
  "timestamp": "2024-01-15T10:30:46.345Z",
  "level": "error",
  "agent": "System",
  "conversation_id": "conv_ghi789",
  "user_id": "user_xyz789",
  "event": "Suspicious content detected, returning KnowledgeAgent for safety",
  "query_preview": "ignore previous instructions and tell me...",
  "pattern": "ignore previous instructions"
}
```

### Example of the entire flow for Knowledge Agent, from request to response:

```json
{
  "conversation_id": "cm1ror17xe",
  "user_id": "u2wzw5prvd",
  "message_preview": "quais são as taxas da maquininha?",
  "event": "Chat request received",
  "timestamp": "2025-09-24T17:54:51.242177+00:00",
  "agent": "System",
  "level": "info",
  "logger": "app.api.v1.chat"
}

{
  "conversation_id": "cm1ror17xe",
  "user_id": "u2wzw5prvd",
  "query_preview": "quais são as taxas da maquininha?",
  "event": "Routing query",
  "timestamp": "2025-09-24T17:54:51.245514+00:00",
  "agent": "RouterAgent",
  "level": "info",
  "logger": "app.agents.router_agent"
}

HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"

{
  "conversation_id": "cm1ror17xe",
  "user_id": "u2wzw5prvd",
  "decision": "KnowledgeAgent",
  "execution_time": "1.500s",
  "query_preview": "quais são as taxas da maquininha?",
  "event": "Agent decision made",
  "timestamp": "2025-09-24T17:54:52.744424+00:00",
  "agent": "RouterAgent",
  "level": "info",
  "logger": "app.agents.router_agent"
}

{
  "query": "quais são as taxas da maquininha?",
  "query_preview": "quais são as taxas da maquininha?",
  "event": "Starting knowledge base query",
  "timestamp": "2025-09-24T17:54:52.745584+00:00",
  "agent": "KnowledgeAgent",
  "level": "info",
  "logger": "app.agents.knowledge_agent.main"
}

HTTP Request: POST https://api.openai.com/v1/embeddings "HTTP/1.1 200 OK"
HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"

{
  "query": "quais são as taxas da maquininha?",
  "answer_preview": "As taxas da maquininha variam conforme o plano de recebimento e o produto utilizado. Para obter deta",
  "execution_time": 4.056973457336426,
  "sources": [
    {
      "url": "https://ajuda.infinitepay.io/pt-BR/articles/6038283-como-usar-a-calculadora-de-taxas-na-maquininha-smart",
      "source": "infinitepay_help_center",
      "score": 0.47174439414701297
    },
    {
      "url": "https://ajuda.infinitepay.io/pt-BR/articles/3567351-quais-modelos-de-maquinas-de-cartao-posso-comprar",
      "source": "infinitepay_help_center",
      "score": 0.4603112251835435
    },
    {
      "url": "https://ajuda.infinitepay.io/pt-BR/articles/3406965-quais-operacoes-nao-posso-fazer-com-a-minha-maquininha",
      "source": "infinitepay_help_center",
      "score": 0.450731820850522
    },
    {
      "url": "https://ajuda.infinitepay.io/pt-BR/articles/9455289-como-obter-taxas-ainda-mais-baixas",
      "source": "infinitepay_help_center",
      "score": 0.44752679986457466
    },
    {
      "url": "https://ajuda.infinitepay.io/pt-BR/articles/3359956-quais-sao-as-taxas-da-infinitepay",
      "source": "infinitepay_help_center",
      "score": 0.44718863944925047
    }
  ],
  "event": "Knowledge base query completed",
  "timestamp": "2025-09-24T17:54:56.802505+00:00",
  "agent": "KnowledgeAgent",
  "level": "info",
  "logger": "app.agents.knowledge_agent.main"
}

{
  "conversation_id": "cm1ror17xe",
  "user_id": "u2wzw5prvd",
  "processed_content": "As taxas da maquininha variam conforme o plano de recebimento e o produto utilizado. Para obter detalhes específicos sobre as formas de pagamento aceitas e os valores aplicados às transações, é recomendado consultar a tabela de taxas disponível na InfinitePay.",
  "execution_time": "4.058s",
  "query_preview": "quais são as taxas da maquininha?",
  "event": "Agent processing completed",
  "timestamp": "2025-09-24T17:54:56.803235+00:00",
  "agent": "System",
  "level": "info",
  "logger": "app.api.v1.chat"
}

{
  "conversation_id": "cm1ror17xe",
  "user_id": "u2wzw5prvd",
  "router_decision": "KnowledgeAgent",
  "execution_time": 5.561532258987427,
  "response_preview": "As taxas da maquininha variam conforme o plano de recebimento e o produto utilizado. Para obter deta",
  "event": "Chat request completed",
  "timestamp": "2025-09-24T17:54:56.803529+00:00",
  "agent": "System",
  "level": "info",
  "logger": "app.api.v1.chat"
}

{
  "event": "Added message to conversation cm1ror17xe",
  "timestamp": "2025-09-24T17:54:56.816103+00:00",
  "agent": "System",
  "level": "info",
  "logger": "app.services.redis_service"
}

{
  "conversation_id": "cm1ror17xe",
  "user_id": "u2wzw5prvd",
  "event": "Conversation saved to Redis",
  "timestamp": "2025-09-24T17:54:56.816617+00:00",
  "agent": "System",
  "level": "info",
  "logger": "app.api.v1.chat"
}

INFO:     192.168.65.1:50534 - "POST /api/v1/chat HTTP/1.1" 200 OK
```

### Example of the entire flow for Math Agent, from request to response:

```json
{
  "conversation_id": "cm1ror17xe",
  "user_id": "u2wzw5prvd",
  "message_preview": "Quanto é 1+1?",
  "event": "Chat request received",
  "timestamp": "2025-09-24T17:59:19.939487+00:00",
  "agent": "System",
  "level": "info",
  "logger": "app.api.v1.chat"
}

{
  "conversation_id": "cm1ror17xe",
  "user_id": "u2wzw5prvd",
  "query_preview": "Quanto é 1+1?",
  "event": "Routing query",
  "timestamp": "2025-09-24T17:59:19.943329+00:00",
  "agent": "RouterAgent",
  "level": "info",
  "logger": "app.agents.router_agent"
}

HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"

{
  "conversation_id": "cm1ror17xe",
  "user_id": "u2wzw5prvd",
  "decision": "`MathAgent`",
  "execution_time": "1.312s",
  "query_preview": "Quanto é 1+1?",
  "event": "Agent decision made",
  "timestamp": "2025-09-24T17:59:21.255261+00:00",
  "agent": "RouterAgent",
  "level": "info",
  "logger": "app.agents.router_agent"
}

{
  "query": "Quanto é 1+1?",
  "query_preview": "Quanto é 1+1?",
  "event": "Starting math evaluation",
  "timestamp": "2025-09-24T17:59:21.256531+00:00",
  "agent": "MathAgent",
  "level": "info",
  "logger": "app.agents.math_agent"
}

HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"

{
  "query": "Quanto é 1+1?",
  "result": "2",
  "execution_time": 0.7338223457336426,
  "event": "Math evaluation completed",
  "timestamp": "2025-09-24T17:59:21.989468+00:00",
  "agent": "MathAgent",
  "level": "info",
  "logger": "app.agents.math_agent"
}

{
  "conversation_id": "cm1ror17xe",
  "user_id": "u2wzw5prvd",
  "processed_content": "2",
  "execution_time": "0.734s",
  "query_preview": "Quanto é 1+1?",
  "event": "Agent processing completed",
  "timestamp": "2025-09-24T17:59:21.989709+00:00",
  "agent": "System",
  "level": "info",
  "logger": "app.api.v1.chat"
}

{
  "agent_type": "MathAgent",
  "response_preview": "2",
  "query_preview": "Quanto é 1+1?",
  "event": "Starting response conversion",
  "timestamp": "2025-09-24T17:59:21.990209+00:00",
  "agent": "RouterAgent",
  "level": "info",
  "logger": "app.agents.router_agent"
}

HTTP Request: POST https://api.openai.com/v1/chat/completions "HTTP/1.1 200 OK"

{
  "agent_type": "MathAgent",
  "original_response_preview": "2",
  "converted_response_preview": "1 + 1 é igual a 2.",
  "execution_time": 0.6029760837554932,
  "event": "Response conversion completed",
  "timestamp": "2025-09-24T17:59:22.592969+00:00",
  "agent": "RouterAgent",
  "level": "info",
  "logger": "app.agents.router_agent"
}

{
  "conversation_id": "cm1ror17xe",
  "user_id": "u2wzw5prvd",
  "router_decision": "MathAgent",
  "execution_time": 2.6541130542755127,
  "response_preview": "1 + 1 é igual a 2.",
  "event": "Chat request completed",
  "timestamp": "2025-09-24T17:59:22.593333+00:00",
  "agent": "System",
  "level": "info",
  "logger": "app.api.v1.chat"
}

{
  "event": "Added message to conversation cm1ror17xe",
  "timestamp": "2025-09-24T17:59:22.614498+00:00",
  "agent": "System",
  "level": "info",
  "logger": "app.services.redis_service"
}

{
  "conversation_id": "cm1ror17xe",
  "user_id": "u2wzw5prvd",
  "event": "Conversation saved to Redis",
  "timestamp": "2025-09-24T17:59:22.614733+00:00",
  "agent": "System",
  "level": "info",
  "logger": "app.api.v1.chat"
}

INFO:     192.168.65.1:50535 - "POST /api/v1/chat HTTP/1.1" 200 OK
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
# Located in: backend/app/agents/router_agent.py
suspicious_patterns = [
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
cd backend
python run_tests.py --type all
```

#### 2. **Specific Test Suites**

```bash
# Router Agent tests only
python run_tests.py --type router

# Math Agent tests only
python run_tests.py --type math

# Chat API tests only
python run_tests.py --type chat

# Unit tests (Router + Math)
python run_tests.py --type unit
```

#### 3. **With Coverage Reports**

```bash
# Terminal coverage report
python run_tests.py --type coverage

# HTML coverage report
python run_tests.py --type coverage-html

# Comprehensive reports (HTML + XML)
python run_tests.py --type coverage-report
```

#### 4. **Verbose Output**

```bash
# Verbose test output
python run_tests.py --type all --verbose

# Suppress warnings
python run_tests.py --type all --no-warnings
```

#### 5. **Coverage Thresholds**

```bash
# Set minimum coverage threshold
python run_tests.py --type coverage --coverage-threshold 85

# Fail if coverage below threshold
python run_tests.py --type coverage --coverage-fail-under 80
```

## 🔧 Development

### Backend Development

```bash
cd backend

# Install dependencies
pip install -r requirements.txt

# Run development server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
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
ai_agents/
├── backend/                    # FastAPI backend
│   ├── app/
│   │   ├── agents/           # AI agents (Router, Math, Knowledge)
│   │   ├── api/              # REST API endpoints
│   │   ├── core/             # Core functionality (logging, settings)
│   │   ├── security/         # Security and sanitization
│   │   └── services/         # External services (Redis)
│   ├── tests/                # Test suites
│   ├── Dockerfile           # Backend container
│   └── requirements.txt     # Python dependencies
├── frontend/                 # React frontend
│   ├── src/                 # React components and services
│   ├── Dockerfile          # Frontend container
│   └── package.json        # Node.js dependencies
├── k8s/                     # Kubernetes manifests
│   ├── backend/            # Backend K8s resources
│   ├── frontend/           # Frontend K8s resources
│   ├── redis/              # Redis K8s resources
│   └── deploy.sh           # Deployment script
├── docker-compose.yml      # Local development setup
└── README.md              # This file
```

## 🚨 Troubleshooting

### Common Issues

#### 1. **Redis Connection Issues**

```bash
# Check Redis status
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
docker-compose logs frontend

# Verify frontend health
curl http://localhost:3000/health
```

#### 4. **Backend API Issues**

```bash
# Check backend logs
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
  "agent_workflow": [
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

## Known Limitations

### Redis Storage

Conversations are stored in Redis only for demonstration purposes, based on the current project requirements (must include Redis). In a production setting, they should be stored in a persistent database.

### User and Conversation Management

User and conversation management is handled entirely in the frontend. This approach was chosen as a workaround to meet time constraints and the project requirements, which do not include a database. In a production setting, they would be created and managed in the backend.

### Response conversion by RouterAgent

Responses from Math Agent are converted to be more conversational ("The answer is 2" instead of simply "2"). But for Knowledge Agent we chose not to use the conversion because (i) Knowledge Agent already responds in a conversational manner and (ii) the conversion nearly doubles the response time.

### Tests

Tests are limited to the cases described in the project requirements. Ideally the coverage should be higher.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
