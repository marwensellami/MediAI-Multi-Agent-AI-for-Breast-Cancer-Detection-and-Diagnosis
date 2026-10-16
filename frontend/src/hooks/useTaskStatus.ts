// src/hooks/useTaskStatus.ts
import { useEffect, useState, useRef } from 'react';
import { getDiagnosisStatus, getDiagnosisResult, isTaskComplete } from '@/api/endpoints/diagnosis';
import { useDiagnosisStore } from '@/store/diagnosisStore';

export function useTaskStatus(taskId: string | null, pollingInterval: number = 3000) {
  const [isPolling, setIsPolling] = useState(false);
  const { updateProgress, setResult, setError } = useDiagnosisStore();
  const intervalRef = useRef<number | null>(null); // ✅ correction du type

  useEffect(() => {
    if (!taskId) {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
      setIsPolling(false);
      return;
    }

    const fetchStatus = async () => {
      try {
        const status = await getDiagnosisStatus(taskId);
        updateProgress(status);

        // Vérifier si la tâche est terminée
        if (isTaskComplete(status.status)) {
          // Récupérer le résultat final
          const result = await getDiagnosisResult(taskId);
          setResult(result);
          setIsPolling(false);
          if (intervalRef.current) {
            clearInterval(intervalRef.current);
            intervalRef.current = null;
          }
        }
      } catch (error) {
        // Ne pas afficher d'erreur pour les 404 (WebSocket ou tâche en cours)
        console.warn('Erreur de polling:', error);
      }
    };

    // Premier appel immédiat
    fetchStatus();
    setIsPolling(true);

    // Mettre en place l'intervalle
    intervalRef.current = setInterval(fetchStatus, pollingInterval) as number;

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
      setIsPolling(false);
    };
  }, [taskId, pollingInterval, updateProgress, setResult, setError]);

  return { isPolling };
}