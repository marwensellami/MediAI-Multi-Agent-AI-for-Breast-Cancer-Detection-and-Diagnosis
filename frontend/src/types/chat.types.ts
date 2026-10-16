// src/types/chat.types.ts

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  timestamp?: number;
}

export interface ChatRequest {
  task_id: string;
  question: string;
  history?: ChatMessage[];
}

export interface ChatResponse {
  answer: string;
  task_id: string;
}