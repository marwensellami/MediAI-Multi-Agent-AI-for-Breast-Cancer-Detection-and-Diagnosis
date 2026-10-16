// src/components/chat/ChatInterface.tsx
import React, { useRef, useEffect } from 'react';
import { MessageSquare } from 'lucide-react';
import { ChatMessage } from './ChatMessage';
import { ChatInput } from './ChatInput';
import { useChat } from '@/hooks/useChat';
import { useChatStore } from '@/store/chatStore';
import { cn } from '@/lib/utils';

interface ChatInterfaceProps {
  taskId: string;
  className?: string;
  isOpen?: boolean;
  onToggle?: () => void;
}

export const ChatInterface: React.FC<ChatInterfaceProps> = ({
  taskId,
  className,
  isOpen = true,
  onToggle,
}) => {
  const { messages, isLoading, sendMessage } = useChat(taskId);
  const { setCurrentTaskId } = useChatStore(); // ✅ Utiliser setCurrentTaskId
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // ✅ Réinitialiser le chat quand le taskId change
  useEffect(() => {
    setCurrentTaskId(taskId);
  }, [taskId, setCurrentTaskId]);

  // Scroll automatique
  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages]);

  if (!isOpen) {
    return (
      <button
        onClick={onToggle}
        className="fixed bottom-6 right-6 z-50 flex h-14 w-14 items-center justify-center rounded-full bg-primary-600 text-white shadow-lg hover:bg-primary-700 transition-colors"
        aria-label="Ouvrir le chat"
      >
        <MessageSquare className="h-6 w-6" />
      </button>
    );
  }

  return (
    <div
      className={cn(
        'fixed bottom-6 right-6 z-50 flex h-[500px] w-[400px] flex-col rounded-xl bg-white shadow-2xl dark:bg-gray-800',
        className
      )}
    >
      {/* En-tête */}
      <div className="flex items-center justify-between border-b border-gray-200 px-4 py-3 dark:border-gray-700">
        <div className="flex items-center gap-2">
          <MessageSquare className="h-5 w-5 text-primary-600" />
          <h3 className="font-medium text-gray-900 dark:text-gray-100">Assistant IA</h3>
        </div>
        <button
          onClick={onToggle}
          className="rounded-full p-1 hover:bg-gray-100 dark:hover:bg-gray-700"
          aria-label="Fermer le chat"
        >
          <svg className="h-5 w-5 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 ? (
          <div className="flex h-full flex-col items-center justify-center text-center text-gray-500 dark:text-gray-400">
            <MessageSquare className="mb-2 h-12 w-12 opacity-20" />
            <p className="text-sm">Posez une question sur ce rapport médical.</p>
          </div>
        ) : (
          messages.map((msg, idx) => (
            <ChatMessage key={idx} message={msg} />
          ))
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="border-t border-gray-200 p-3 dark:border-gray-700">
        <ChatInput onSend={sendMessage} isLoading={isLoading} />
      </div>
    </div>
  );
};