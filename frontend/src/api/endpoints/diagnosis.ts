// src/api/endpoints/diagnosis.ts
import apiClient from '@/api/client';
import {
  CreateDiagnosisResponse,
  DiagnosisProgress,
  DiagnosisResult,
} from '@/types/diagnosis.types';

export interface CreateDiagnosisPayload {
  patient_id?: string;   // optionnel
  age?: number;
  history?: string;
  risk_factors?: string;
  mammo_file?: File;
  us_file?: File;
  mri_files?: File[];
}

export async function createDiagnosis(
  payload: CreateDiagnosisPayload
): Promise<CreateDiagnosisResponse> {
  const formData = new FormData();
  
  if (payload.patient_id) {
    formData.append('patient_id', payload.patient_id);
  }
  
  // ✅ N'ajouter age que s'il est défini
  if (payload.age !== undefined && payload.age !== null) {
    formData.append('age', String(payload.age));
  }
  
  if (payload.history) formData.append('history', payload.history);
  if (payload.risk_factors) formData.append('risk_factors', payload.risk_factors);

  if (payload.mammo_file) {
    formData.append('mammo_file', payload.mammo_file);
  }
  if (payload.us_file) {
    formData.append('us_file', payload.us_file);
  }
  if (payload.mri_files) {
    payload.mri_files.forEach((file: File) => {
      formData.append('mri_files', file);
    });
  }

  const response = await apiClient.post<CreateDiagnosisResponse>(
    '/diagnosis',
    formData,
    {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    }
  );
  return response.data;
}

/**
 * Récupère la progression d'un diagnostic
 */
export async function getDiagnosisStatus(taskId: string): Promise<DiagnosisProgress> {
  const response = await apiClient.get<DiagnosisProgress>(
    `/diagnosis/${taskId}/status`
  );
  return response.data;
}

/**
 * Récupère le résultat final d'un diagnostic
 */
export async function getDiagnosisResult(taskId: string): Promise<DiagnosisResult> {
  const response = await apiClient.get<DiagnosisResult>(
    `/diagnosis/${taskId}/result`
  );
  return response.data;
}

export async function getDiagnosisReport(diagnosisId: number): Promise<DiagnosisResult> {
  const response = await apiClient.get<DiagnosisResult>(`/diagnosis/${diagnosisId}/report`);
  return response.data;
}

/**
 * Vérifie si une tâche est terminée (utile pour le polling)
 */
export function isTaskComplete(status: string): boolean {
  return status === 'COMPLETED' || status === 'FAILED';
}