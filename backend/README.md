# backend/README.md

# Backend Documentation

This directory contains the FastAPI backend for the AI Agents application, featuring intelligent routing between different specialized agents (Math Agent, Knowledge Agent) based on user queries.

## Setup Instructions

1. **Clone the repository:**

   ```
   git clone <repository-url>
   cd backend
   ```

2. **Create a virtual environment (optional but recommended):**

   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the required dependencies:**

   ```
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**

   Create a `.env` file with the following variables:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   REDIS_HOST=localhost
   REDIS_PORT=6379
   REDIS_PASSWORD=your_redis_password
   ```

5. **Run the FastAPI application:**

   ```
   uvicorn app.main:app --reload
   ```

6. **Access the API:**
   - API Base URL: `http://127.0.0.1:8000`
   - Interactive API Documentation: `http://127.0.0.1:8000/docs`
   - Alternative Documentation: `http://127.0.0.1:8000/redoc`

## API Endpoints

### Health Check
- **`GET /health`**
  - **Description**: Health check endpoint for Kubernetes and monitoring
  - **Response**: `{"status": "healthy"}`

### Chat API

#### Send Chat Message
- **`POST /api/v1/chat`**
  - **Description**: Send a message to the AI agents system. The system will automatically route your query to the appropriate agent (Math Agent for mathematical problems, Knowledge Agent for general questions).
  - **Request Body**:
    ```json
    {
      "message": "What is 2 + 2?",
      "user_id": "user123",
      "conversation_id": "conv456"
    }
    ```
  - **Response**:
    ```json
    {
      "user_id": "user123",
      "conversation_id": "conv456",
      "router_decision": "math_agent",
      "response": "2 + 2 = 4",
      "source_agent_response": "The result is 4",
      "agent_workflow": [
        {
          "agent": "router",
          "action": "analyzed_query",
          "result": "routed_to_math_agent"
        },
        {
          "agent": "math_agent",
          "action": "solved_equation",
          "result": "2 + 2 = 4"
        }
      ]
    }
    ```

#### Get Conversation History
- **`GET /api/v1/chat/history/{conversation_id}`**
  - **Description**: Retrieve the complete conversation history for a specific conversation
  - **Path Parameters**:
    - `conversation_id` (string): The unique identifier for the conversation
  - **Response**:
    ```json
    {
      "conversation_id": "conv456",
      "messages": [
        {
          "role": "user",
          "content": "What is 2 + 2?",
          "timestamp": "2024-01-01T12:00:00Z"
        },
        {
          "role": "assistant",
          "content": "2 + 2 = 4",
          "timestamp": "2024-01-01T12:00:01Z"
        }
      ]
    }
    ```

#### Get User Conversations
- **`GET /api/v1/chat/user/{user_id}/conversations`**
  - **Description**: Retrieve all conversation IDs for a specific user
  - **Path Parameters**:
    - `user_id` (string): The unique identifier for the user
  - **Response**:
    ```json
    {
      "user_id": "user123",
      "conversations": [
        {
          "conversation_id": "conv456",
          "created_at": "2024-01-01T12:00:00Z",
          "last_message_at": "2024-01-01T12:05:00Z"
        }
      ]
    }
    ```

## Data Models

### ChatRequest
```json
{
  "message": "string",        // The user's message/query
  "user_id": "string",        // Unique identifier for the user
  "conversation_id": "string" // Unique identifier for the conversation
}
```

### ChatResponse
```json
{
  "user_id": "string",                    // User identifier
  "conversation_id": "string",            // Conversation identifier
  "router_decision": "string",            // Which agent was selected (math_agent, knowledge_agent)
  "response": "string",                   // The final response to the user
  "source_agent_response": "string",      // Raw response from the selected agent
  "agent_workflow": [                     // Step-by-step workflow execution
    {
      "agent": "string",                  // Agent name (router, math_agent, knowledge_agent)
      "action": "string",                 // Action performed by the agent
      "result": "string"                  // Result of the action
    }
  ]
}
```

## Agent Types

The system includes the following specialized agents:

1. **Router Agent**: Analyzes incoming queries and determines which specialized agent should handle them
2. **Math Agent**: Handles mathematical problems, equations, and calculations
3. **Knowledge Agent**: Answers general questions using a pre-built vector store of documentation

## Features

- **Intelligent Routing**: Automatically determines the best agent for each query
- **Conversation Management**: Maintains conversation history with Redis
- **Vector Store**: Pre-built knowledge base for answering general questions
- **Error Handling**: Comprehensive error handling with detailed error responses
- **Logging**: Structured logging for monitoring and debugging
- **CORS Support**: Configured for frontend integration

## Additional Information

- **FastAPI Documentation**: [https://fastapi.tiangolo.com/](https://fastapi.tiangolo.com/)
- **Interactive API Docs**: Available at `/docs` when running the server
- **Health Monitoring**: Use `/health` endpoint for Kubernetes health checks
