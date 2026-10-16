// src/hooks/useWebSocket.ts
import { useEffect, useState, useCallback, useRef } from 'react';
import { wsClient } from '@/api/websocket';
import { useDiagnosisStore } from '@/store/diagnosisStore';
import { DiagnosisProgress } from '@/types/diagnosis.types';

export function useWebSocket(taskId: string | null) {
  const [isConnected, setIsConnected] = useState(false);
  const { updateProgress, setError } = useDiagnosisStore();
  const reconnectTimeoutRef = useRef<number | null>(null);
  const mountedRef = useRef(true);

  const handleMessage = useCallback(
    (data: DiagnosisProgress) => {
      if (!mountedRef.current) return;
      updateProgress(data);
      if (data.status === 'FAILED' && data.error) {
        setError(data.error);
      }
    },
    [updateProgress, setError]
  );

  const handleClose = useCallback(() => {
    if (mountedRef.current) {
      setIsConnected(false);
    }
  }, []);

  const handleError = useCallback(() => {
    if (mountedRef.current) {
      setIsConnected(false);
    }
  }, []);

  useEffect(() => {
    mountedRef.current = true;

    if (!taskId) {
      wsClient.disconnect();
      setIsConnected(false);
      return () => {
        mountedRef.current = false;
      };
    }

    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    // Nettoyer les anciens handlers
    // Note: on ne peut pas facilement supprimer des handlers individuels dans l'implémentation actuelle
    // On va simplement s'assurer que la connexion est établie

    const connect = async () => {
      if (!mountedRef.current) return;
      
      try {
        // Ajouter les handlers
        wsClient.onMessage(handleMessage);
        wsClient.onClose(handleClose);
        wsClient.onError(handleError);

        await wsClient.connect(taskId);
        if (mountedRef.current) {
          setIsConnected(true);
          console.log(`📡 WebSocket connecté pour la tâche ${taskId}`);
        }
      } catch (error) {
        console.error('❌ Erreur de connexion WebSocket:', error);
        if (mountedRef.current) {
          setIsConnected(false);
          // Tentative de reconnexion après un délai (gérée par wsClient)
        }
      }
    };

    connect();

    return () => {
      mountedRef.current = false;
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
        reconnectTimeoutRef.current = null;
      }
      // Ne pas déconnecter ici pour permettre la reconnexion automatique
      // wsClient.disconnect();
    };
  }, [taskId, handleMessage, handleClose, handleError]);

  return { isConnected };
}