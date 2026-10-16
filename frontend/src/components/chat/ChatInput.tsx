// src/components/chat/ChatInput.tsx
import React, { useState, useRef, useEffect } from 'react';
import { Send } from 'lucide-react';
import { Button } from '@/components/common';

interface ChatInputProps {
  onSend: (message: string) => void;
  isLoading?: boolean;
  placeholder?: string;
  className?: string;
}

export const ChatInput: React.FC<ChatInputProps> = ({
  onSend,
  isLoading = false,
  placeholder = 'Posez une question sur le rapport...',
  className,
}) => {
  const [value, setValue] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (value.trim() && !isLoading) {
      onSend(value.trim());
      setValue('');
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  // Auto-resize du textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`;
    }
  }, [value]);

  return (
    <form onSubmit={handleSubmit} className={className}>
      <div className="flex items-end gap-2 rounded-xl border border-gray-200 bg-white p-2 dark:border-gray-700 dark:bg-gray-800">
        <textarea
          ref={textareaRef}
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          rows={1}
          className="max-h-40 flex-1 resize-none border-0 bg-transparent px-2 py-1 text-gray-900 placeholder:text-gray-400 focus:outline-none dark:text-gray-100 dark:placeholder:text-gray-500"
          disabled={isLoading}
        />
        <Button
          type="submit"
          size="sm"
          disabled={!value.trim() || isLoading}
          className="flex-shrink-0"
        >
          {isLoading ? (
            <div className="h-4 w-4 animate-spin rounded-full border-2 border-white border-t-transparent" />
          ) : (
            <Send className="h-4 w-4" />
          )}
        </Button>
      </div>
    </form>
  );
};