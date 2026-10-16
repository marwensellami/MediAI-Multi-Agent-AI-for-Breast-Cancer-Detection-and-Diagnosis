// src/pages/Progress.tsx
import React, { useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Layout } from '@/components/layout/Layout';
import { Card } from '@/components/common';
import { ProgressBar, StepLogs } from '@/components/progress';
import { useDiagnosisStore } from '@/store/diagnosisStore';
import { useWebSocket } from '@/hooks/useWebSocket';
import { useTaskStatus } from '@/hooks/useTaskStatus';
import { Loader2 } from 'lucide-react';
// src/pages/Progress.tsx
import { Button } from '@/components/common'; // ✅ à ajouter

export const Progress: React.FC = () => {
  const { taskId } = useParams<{ taskId: string }>();
  const navigate = useNavigate();
  const { progress, steps, status, error } = useDiagnosisStore();
  const { isConnected } = useWebSocket(taskId || null);
  useTaskStatus(taskId || null, 3000);

  useEffect(() => {
    if (status === 'completed' && taskId) {
      navigate(`/results/${taskId}`);
    }
  }, [status, taskId, navigate]);

  if (status === 'error') {
    return (
      <Layout>
        <div className="flex h-[60vh] items-center justify-center">
          <div className="text-center">
            <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-full bg-red-100 dark:bg-red-900/30">
              <span className="text-2xl">❌</span>
            </div>
            <h2 className="text-xl font-semibold text-gray-900 dark:text-gray-100">
              Une erreur est survenue
            </h2>
            <p className="mt-2 text-sm text-gray-500 dark:text-gray-400">{error}</p>
            <Button className="mt-4" onClick={() => navigate('/upload')}>
              Réessayer
            </Button>
          </div>
        </div>
      </Layout>
    );
  }

  return (
    <Layout>
      <div className="mx-auto max-w-2xl space-y-6 py-6">
        <div className="text-center">
          <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">
            Analyse en cours
          </h1>
          <p className="text-sm text-gray-500 dark:text-gray-400">
            L'IA analyse la mammographie, veuillez patienter
          </p>
        </div>

        <Card>
          <div className="space-y-6">
            {/* État de connexion */}
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <div className={`h-3 w-3 rounded-full ${isConnected ? 'bg-green-500' : 'bg-yellow-500 animate-pulse'}`} />
                <span className="text-sm text-gray-600 dark:text-gray-400">
                  {isConnected ? 'Connecté en temps réel' : 'Reconnexion...'}
                </span>
              </div>
              <span className="text-xs text-gray-400">ID: {taskId?.slice(0, 8)}</span>
            </div>

            {/* Progression */}
            <ProgressBar value={progress} label="Progression" showPercentage />

            {/* Étapes */}
            <div className="mt-4">
              <h3 className="mb-3 text-sm font-medium text-gray-700 dark:text-gray-300">
                Étapes en cours
              </h3>
              <StepLogs steps={steps} />
            </div>

            {/* Loader */}
            {status === 'processing' && (
              <div className="flex justify-center py-4">
                <Loader2 className="h-8 w-8 animate-spin text-primary-500" />
              </div>
            )}
          </div>
        </Card>
      </div>
    </Layout>
  );
};