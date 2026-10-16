// src/hooks/useChat.ts
import { useState, useCallback, useEffect } from 'react';
import toast from 'react-hot-toast';
import { sendChatMessage } from '@/api/endpoints/chat';
import { useChatStore } from '@/store/chatStore';
import { ChatMessage } from '@/types/chat.types';

export function useChat(taskId: string | null) {
  const [isLoading, setIsLoading] = useState(false);
  const { 
     
    addMessage, 
    setTyping, 
    setCurrentTaskId,
    getMessages,
    clearMessages 
  } = useChatStore();

  // ✅ Récupérer les messages pour ce taskId
  const currentMessages = taskId ? getMessages(taskId) : [];

  // ✅ Mettre à jour le taskId courant
  useEffect(() => {
    if (taskId) {
      setCurrentTaskId(taskId);
    }
  }, [taskId, setCurrentTaskId]);

  const sendMessage = useCallback(
    async (question: string) => {
      if (!taskId) {
        toast.error('Aucune tâche associée.');
        return;
      }

      if (!question.trim()) {
        return;
      }

      const userMessage: ChatMessage = {
        role: 'user',
        content: question.trim(),
        timestamp: Date.now(),
      };
      addMessage(taskId, userMessage);

      const history = currentMessages.map((msg) => ({
        role: msg.role,
        content: msg.content,
      }));

      setIsLoading(true);
      setTyping(true);

      try {
        const response = await sendChatMessage(taskId, question, history);

        const assistantMessage: ChatMessage = {
          role: 'assistant',
          content: response.answer,
          timestamp: Date.now(),
        };
        addMessage(taskId, assistantMessage);
      } catch (error) {
        const errorMessage = error instanceof Error ? error.message : 'Erreur inconnue';
        toast.error(`Erreur du chatbot : ${errorMessage}`);

        const errorAssistantMessage: ChatMessage = {
          role: 'assistant',
          content: `Désolé, une erreur s'est produite : ${errorMessage}`,
          timestamp: Date.now(),
        };
        addMessage(taskId, errorAssistantMessage);
      } finally {
        setIsLoading(false);
        setTyping(false);
      }
    },
    [taskId, currentMessages, addMessage, setTyping]
  );

  const clearChat = useCallback(() => {
    if (taskId) {
      clearMessages(taskId);
    }
  }, [taskId, clearMessages]);

  return {
    messages: currentMessages,
    isLoading,
    sendMessage,
    clearChat,
  };
}