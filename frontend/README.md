# AI Agents Chat Frontend

A React-based chat interface for the AI Agents FastAPI backend, built with Vite, Tailwind CSS, and modern React patterns.

## Features

- 💬 **Simple Chat Interface**: Clean and intuitive chat UI
- 🔄 **Multiple Conversations**: Support for multiple conversation threads
- 📱 **Responsive Design**: Works on desktop and mobile devices
- ⚡ **Optimistic Updates**: Instant message display with pending states
- 🔄 **Loading States**: Skeleton loaders and spinners for better UX
- ❌ **Error Handling**: Toast notifications for API errors
- 💾 **Local Storage**: Persistent user ID across browser sessions
- 🎨 **Modern UI**: Built with Tailwind CSS for a beautiful interface

## Tech Stack

- **React**: 18.3.1
- **Vite**: 7.1.6 (Build tool)
- **Tailwind CSS**: 4.1.13 (Styling)
- **JavaScript**: ES6+ (No TypeScript)

## Setup Instructions

### Prerequisites

- Node.js (v16 or higher)
- npm or yarn
- FastAPI backend running on `http://localhost:8000`

### Installation

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm run dev
   ```

4. Open your browser and navigate to `http://localhost:3000`

### Environment Configuration

The API base URL is configured in `.env`:
```
VITE_API_BASE_URL=http://localhost:8000
```

To change the backend URL, update this file or set the environment variable.

## Usage

### Starting a New Conversation

1. Click the "Nova Conversa" button in the sidebar
2. Type your message and press Enter or click "Enviar"
3. The system will automatically create a new conversation ID

### Managing Conversations

- **View Conversations**: All your conversations appear in the left sidebar
- **Switch Conversations**: Click on any conversation to view its history
- **Clear All Data**: Click the "🗑️ Limpar Todas as Conversas" button to reset your user ID and start fresh

### Features in Detail

#### User ID Management
- User IDs are automatically generated and stored in browser localStorage
- Each user ID persists across browser sessions
- Use the "Clear All Conversations" button to generate a new user ID

#### Message Display
- **User Messages**: Appear on the right side in blue
- **Agent Responses**: Appear on the left side in gray
- **Agent Information**: Shows which agent (RouterAgent, KnowledgeAgent, etc.) handled the response
- **Timestamps**: Displayed for all messages
- **Pending States**: Shows "Enviando..." while waiting for API response

#### Error Handling
- Network errors show toast notifications
- API errors display specific error messages
- Loading states prevent multiple simultaneous requests

## API Integration

The frontend integrates with the following FastAPI endpoints:

- `POST /api/v1/chat` - Send messages
- `GET /api/v1/chat/user/{user_id}/conversations` - Get user conversations
- `GET /api/v1/chat/history/{conversation_id}` - Get conversation history

## Development

### Project Structure

```
src/
├── components/          # React components
│   ├── ChatInterface.jsx
│   ├── ConversationList.jsx
│   ├── Message.jsx
│   ├── MessageInput.jsx
│   ├── MessageList.jsx
│   └── ErrorNotification.jsx
├── services/           # API service layer
│   └── api.js
├── utils/              # Utility functions
│   └── storage.js
├── App.jsx             # Main app component
├── main.jsx           # Entry point
└── index.css          # Global styles
```

### Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build

### Customization

#### Styling
- Modify `tailwind.config.js` to customize Tailwind settings
- Update `src/index.css` for global styles
- Component styles use Tailwind utility classes

#### API Configuration
- Update `src/services/api.js` to modify API behavior
- Change `VITE_API_BASE_URL` in `.env` for different environments

## Troubleshooting

### Common Issues

1. **API Connection Errors**
   - Ensure the FastAPI backend is running on `http://localhost:8000`
   - Check the `.env` file has the correct `VITE_API_BASE_URL`

2. **Build Errors**
   - Clear node_modules and reinstall: `rm -rf node_modules && npm install`
   - Check Node.js version compatibility

3. **Styling Issues**
   - Ensure Tailwind CSS is properly configured
   - Check that `@import "tailwindcss";` is in `src/index.css`

### Browser Compatibility

- Chrome (latest)
- Firefox (latest)
- Safari (latest)
- Edge (latest)

## Contributing

1. Follow the existing code structure
2. Use functional components with hooks
3. Maintain consistent error handling patterns
4. Test with the FastAPI backend
5. Ensure responsive design works on mobile devices