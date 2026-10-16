// frontend/src/hooks/useDiagnosis.ts
import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import toast from 'react-hot-toast';
import { createDiagnosis, CreateDiagnosisPayload } from '@/api/endpoints/diagnosis';
import { useDiagnosisStore } from '@/store/diagnosisStore';

export function useDiagnosis() {
  const navigate = useNavigate();
  const [isLoading, setIsLoading] = useState(false);
  const { setTaskId, setUploading, setError } = useDiagnosisStore();

  const submit = async (payload: CreateDiagnosisPayload) => {
    setIsLoading(true);
    setUploading();

    try {
      const response = await createDiagnosis(payload);
      const taskId = response.task_id;
      setTaskId(taskId);
      navigate(`/progress/${taskId}`);
      toast.success('Diagnostic lancé avec succès !');
      return taskId;
    } catch (err: any) {
      // ✅ Extraire le message d'erreur pour éviter d'afficher un objet
      let errorMessage = 'Erreur inconnue';
      if (err.response?.data?.detail) {
        if (typeof err.response.data.detail === 'string') {
          errorMessage = err.response.data.detail;
        } else if (Array.isArray(err.response.data.detail)) {
          errorMessage = err.response.data.detail.map((d: any) => d.msg).join(', ');
        } else if (typeof err.response.data.detail === 'object') {
          errorMessage = JSON.stringify(err.response.data.detail);
        }
      } else if (err.message) {
        errorMessage = err.message;
      }
      
      setError(errorMessage);
      toast.error(`Erreur : ${errorMessage}`);
      return null;
    } finally {
      setIsLoading(false);
    }
  };

  return { submit, isLoading };
}