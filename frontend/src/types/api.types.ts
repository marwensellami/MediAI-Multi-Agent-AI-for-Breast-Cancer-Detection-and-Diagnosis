// src/types/api.types.ts

export interface ApiResponse<T> {
  data: T;
  message?: string;
  status: number;
}

export interface ErrorResponse {
  detail: string;
  status?: number;
}

export interface HealthResponse {
  status: 'ok' | 'unavailable';
}