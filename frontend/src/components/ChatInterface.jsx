import React, { useState, useEffect } from "react";
import { chatApi, ApiError } from "../services/api";
import { generateConversationId } from "../utils/storage";
import MessageList from "./MessageList";
import MessageInput from "./MessageInput";
import ErrorNotification from "./ErrorNotification";

const ChatInterface = ({ conversationId, userId, onConversationChange }) => {
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isLoadingHistory, setIsLoadingHistory] = useState(true);
  const [error, setError] = useState(null);

  // Load conversation history when conversationId changes
  useEffect(() => {
    if (conversationId) {
      loadConversationHistory();
    } else {
      setMessages([]);
      setIsLoadingHistory(false);
    }
  }, [conversationId]);

  const loadConversationHistory = async () => {
    if (!conversationId) return;

    setIsLoadingHistory(true);
    setError(null);

    try {
      const history = await chatApi.getConversationHistory(conversationId);

      // Transform the history into our message format
      const formattedMessages = history.history.map((item) => ({
        user_message: item.user_message,
        agent_response: item.agent_response,
        timestamp: item.timestamp,
        router_decision: "LLM", // Default agent name
      }));

      setMessages(formattedMessages);
    } catch (err) {
      console.error("Error loading conversation history:", err);
      setError(
        err instanceof ApiError
          ? err.message
          : "Erro ao carregar histórico da conversa"
      );
    } finally {
      setIsLoadingHistory(false);
    }
  };

  const handleSendMessage = async (messageText) => {
    if (!conversationId) {
      // Create new conversation
      const newConversationId = generateConversationId();
      onConversationChange(newConversationId);
      // The useEffect will handle loading the new conversation
      return;
    }

    // Add user message optimistically
    const userMessage = {
      user_message: messageText,
      timestamp: new Date().toISOString(),
      isPending: false,
    };

    // Add pending response message
    const pendingResponse = {
      agent_response: "",
      timestamp: new Date().toISOString(),
      isPending: true,
    };

    setMessages((prev) => [...prev, userMessage, pendingResponse]);
    setIsLoading(true);
    setError(null);

    try {
      const response = await chatApi.sendMessage(
        messageText,
        userId,
        conversationId
      );

      // Update the pending message with the actual response
      setMessages((prev) =>
        prev.map((msg) =>
          msg.isPending
            ? {
                ...msg,
                agent_response: response.response,
                router_decision: response.router_decision,
                isPending: false,
              }
            : msg
        )
      );
    } catch (err) {
      console.error("Error sending message:", err);
      setError(
        err instanceof ApiError ? err.message : "Erro ao enviar mensagem"
      );

      // Remove the pending message on error
      setMessages((prev) => prev.filter((msg) => !msg.isPending));
    } finally {
      setIsLoading(false);
    }
  };

  const dismissError = () => {
    setError(null);
  };

  return (
    <div className="flex-1 flex flex-col h-full">
      <div className="bg-white border-b border-gray-200 p-4">
        <h1 className="text-xl font-semibold text-gray-800">
          {conversationId ? `Conversa ${conversationId}` : "Nova Conversa"}
        </h1>
        <p className="text-sm text-gray-500">Usuário: {userId}</p>
      </div>

      <MessageList messages={messages} isLoading={isLoadingHistory} />

      <MessageInput onSendMessage={handleSendMessage} isLoading={isLoading} />

      <ErrorNotification error={error} onDismiss={dismissError} />
    </div>
  );
};

export default ChatInterface;
