import React, { useState, useEffect } from "react";
import { chatApi } from "./services/api";
import {
  getUserId,
  clearUserId,
  generateUserId,
  generateConversationId,
} from "./utils/storage";
import ConversationList from "./components/ConversationList";
import ChatInterface from "./components/ChatInterface";
import ErrorNotification from "./components/ErrorNotification";

function App() {
  const [userId, setUserId] = useState(null);
  const [conversations, setConversations] = useState([]);
  const [currentConversationId, setCurrentConversationId] = useState(null);
  const [isLoadingConversations, setIsLoadingConversations] = useState(true);
  const [error, setError] = useState(null);

  // Initialize user ID on app load
  useEffect(() => {
    const initializeUser = async () => {
      const id = getUserId();
      setUserId(id);
      await loadUserConversations(id);
    };

    initializeUser();
  }, []);

  const loadUserConversations = async (userIdToLoad) => {
    if (!userIdToLoad) return;

    setIsLoadingConversations(true);
    setError(null);

    try {
      const data = await chatApi.getUserConversations(userIdToLoad);
      setConversations(data.conversation_ids || []);
    } catch (err) {
      console.error("Error loading conversations:", err);
      setError("Erro ao carregar conversas");
    } finally {
      setIsLoadingConversations(false);
    }
  };

  const handleSelectConversation = (conversationId) => {
    setCurrentConversationId(conversationId);
  };

  const handleConversationChange = (newConversationId) => {
    setCurrentConversationId(newConversationId);
    // Add the new conversation to the list if it's not already there
    if (!conversations.includes(newConversationId)) {
      setConversations((prev) => [...prev, newConversationId]);
    }
  };

  const handleCreateNewConversation = () => {
    const newConversationId = generateConversationId();
    handleConversationChange(newConversationId);
  };

  const handleClearAllConversations = () => {
    clearUserId();
    const newUserId = generateUserId();
    setUserId(newUserId);
    setConversations([]);
    setCurrentConversationId(null);
    setIsLoadingConversations(true);
    loadUserConversations(newUserId);
  };

  const dismissError = () => {
    setError(null);
  };

  if (!userId) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="h-screen flex">
        <ConversationList
          conversations={conversations}
          currentConversationId={currentConversationId}
          onSelectConversation={handleSelectConversation}
          onCreateNewConversation={handleCreateNewConversation}
          isLoading={isLoadingConversations}
        />

        {currentConversationId ? (
          <ChatInterface
            conversationId={currentConversationId}
            userId={userId}
            onConversationChange={handleConversationChange}
          />
        ) : (
          <div className="flex-1 flex items-center justify-center bg-gray-50">
            <div className="text-center text-gray-500">
              <div className="text-6xl mb-4">💬</div>
              <h2 className="text-2xl font-semibold mb-2">
                Bem-vindo ao Chat!
              </h2>
              <p className="text-lg mb-4">
                Selecione uma conversa existente ou crie uma nova para começar.
              </p>
              <p className="text-sm text-gray-400">
                Use o botão "Nova Conversa" na barra lateral para começar.
              </p>
            </div>
          </div>
        )}
      </div>

      {/* Clear conversations button */}
      <div className="fixed bottom-4 left-4">
        <button
          onClick={handleClearAllConversations}
          className="bg-red-500 hover:bg-red-600 text-white px-4 py-2 rounded-lg text-sm font-medium transition-colors shadow-lg"
        >
          🗑️ Limpar Todas as Conversas
        </button>
      </div>

      <ErrorNotification error={error} onDismiss={dismissError} />
    </div>
  );
}

export default App;
