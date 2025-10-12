import React, { useRef, useEffect } from "react";
import Message from "./Message";

const MessageList = ({ messages, isLoading }) => {
  const messagesEndRef = useRef(null);
  const messagesContainerRef = useRef(null);

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages]);

  if (isLoading) {
    return (
      <div className="flex-1 p-4">
        <div className="space-y-4">
          {[...Array(3)].map((_, i) => (
            <div key={i} className="animate-pulse">
              <div className="flex justify-start">
                <div className="bg-gray-200 h-16 w-3/4 rounded-lg"></div>
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  if (!messages || messages.length === 0) {
    return (
      <div className="flex-1 flex items-center justify-center p-4">
        <div className="text-center text-gray-500">
          <div className="text-4xl mb-4">💬</div>
          <p>Nenhuma mensagem ainda.</p>
          <p className="text-sm">Envie uma mensagem para começar a conversa!</p>
        </div>
      </div>
    );
  }

  return (
    <div ref={messagesContainerRef} className="flex-1 overflow-y-auto p-4 space-y-4">
      {messages.map((msg, index) => (
        <Message
          key={index}
          message={msg.user_message || msg.agent_response}
          isUser={!!msg.user_message}
          timestamp={msg.timestamp}
          agent={msg.router_decision}
          isPending={msg.isPending}
        />
      ))}
      <div ref={messagesEndRef} />
    </div>
  );
};

export default MessageList;
