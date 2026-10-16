// src/components/chat/ChatMessage.tsx
import React from 'react';
import { cn } from '@/lib/utils';
import { ChatMessage as ChatMessageType } from '@/types/chat.types';
import { Bot, User } from 'lucide-react';

interface ChatMessageProps {
  message: ChatMessageType;
  className?: string;
}

export const ChatMessage: React.FC<ChatMessageProps> = ({ message, className }) => {
  const isUser = message.role === 'user';

  return (
    <div
      className={cn(
        'flex gap-3',
        isUser ? 'flex-row-reverse' : 'flex-row',
        className
      )}
    >
      <div
        className={cn(
          'flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-full',
          isUser ? 'bg-primary-600 text-white' : 'bg-gray-200 text-gray-700 dark:bg-gray-700 dark:text-gray-300'
        )}
      >
        {isUser ? <User className="h-4 w-4" /> : <Bot className="h-4 w-4" />}
      </div>
      <div
        className={cn(
          'max-w-[85%] rounded-2xl px-4 py-2 text-sm',
          isUser
            ? 'bg-primary-600 text-white'
            : 'bg-gray-100 text-gray-900 dark:bg-gray-800 dark:text-gray-100'
        )}
      >
        <div className="whitespace-pre-wrap break-words">{message.content}</div>
        {message.timestamp && (
          <div className="mt-1 text-xs opacity-50">
            {new Date(message.timestamp).toLocaleTimeString()}
          </div>
        )}
      </div>
    </div>
  );
};