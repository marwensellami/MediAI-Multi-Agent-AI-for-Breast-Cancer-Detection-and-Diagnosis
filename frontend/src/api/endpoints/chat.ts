// src/api/endpoints/chat.ts
import apiClient from '@/api/client';
import { ChatRequest, ChatResponse } from '@/types/chat.types';

export async function sendChatMessage(
  taskId: string,
  question: string,
  history?: ChatRequest['history']
): Promise<ChatResponse> {
  const response = await apiClient.post<ChatResponse>(
    '/chat/', // ✅ Avec le slash final
    {
      task_id: taskId,
      question,
      history: history || [],
    }
  );
  return response.data;
}