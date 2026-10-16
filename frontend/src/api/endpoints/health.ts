// src/api/endpoints/health.ts
import apiClient from '@/api/client';
import { HealthResponse } from '@/types/api.types';

export async function checkHealth(): Promise<HealthResponse> {
  const response = await apiClient.get<HealthResponse>('/health');
  return response.data;
}