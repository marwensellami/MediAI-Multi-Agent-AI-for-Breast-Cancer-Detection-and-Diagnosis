// src/store/diagnosisStore.ts
import { create } from 'zustand';
import { DiagnosisProgress, DiagnosisResult } from '@/types/diagnosis.types';

interface DiagnosisState {
  // État
  taskId: string | null;
  status: 'idle' | 'uploading' | 'processing' | 'completed' | 'error';
  progress: number;
  steps: { step: string; percent: number; message: string }[];
  result: DiagnosisResult | null;
  error: string | null;

  // Actions
  setTaskId: (taskId: string) => void;
  setUploading: () => void;
  updateProgress: (data: DiagnosisProgress) => void;
  setResult: (result: DiagnosisResult) => void;
  setError: (error: string) => void;
  reset: () => void;
}

export const useDiagnosisStore = create<DiagnosisState>((set) => ({
  // État initial
  taskId: null,
  status: 'idle',
  progress: 0,
  steps: [],
  result: null,
  error: null,

  setTaskId: (taskId) => set({ taskId, status: 'processing' }),

  setUploading: () => set({ status: 'uploading', progress: 0, error: null }),

  updateProgress: (data) =>
    set((state) => ({
      status: data.status === 'COMPLETED' ? 'completed' : 'processing',
      progress: data.percent,
      steps: data.steps || state.steps,
      error: data.error || null,
    })),

  setResult: (result) =>
    set({
      result,
      status: 'completed',
      progress: 100,
    }),

  setError: (error) =>
    set({
      error,
      status: 'error',
    }),

  reset: () =>
    set({
      taskId: null,
      status: 'idle',
      progress: 0,
      steps: [],
      result: null,
      error: null,
    }),
}));