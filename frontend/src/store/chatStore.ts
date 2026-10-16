// src/store/chatStore.ts
import { create } from 'zustand';
import { ChatMessage } from '@/types/chat.types';

interface ChatState {
  messages: Record<string, ChatMessage[]>;  // ✅ Stockage par taskId
  isTyping: boolean;
  currentTaskId: string | null;

  setCurrentTaskId: (taskId: string) => void;
  getMessages: (taskId: string) => ChatMessage[];
  addMessage: (taskId: string, message: ChatMessage) => void;
  setTyping: (isTyping: boolean) => void;
  clearMessages: (taskId: string) => void;
  clearAllMessages: () => void;
}

export const useChatStore = create<ChatState>((set, get) => ({
  messages: {},
  isTyping: false,
  currentTaskId: null,

  setCurrentTaskId: (taskId) => set({ currentTaskId: taskId }),

  getMessages: (taskId) => {
    return get().messages[taskId] || [];
  },

  addMessage: (taskId, message) =>
    set((state) => ({
      messages: {
        ...state.messages,
        [taskId]: [...(state.messages[taskId] || []), message],
      },
    })),

  setTyping: (isTyping) => set({ isTyping }),

  clearMessages: (taskId) =>
    set((state) => {
      const { [taskId]: _, ...rest } = state.messages;
      return { messages: rest };
    }),

  clearAllMessages: () => set({ messages: {} }),
}));