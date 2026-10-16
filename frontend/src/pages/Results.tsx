// src/pages/Results.tsx
import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Layout } from '@/components/layout/Layout';
import { Button, Spinner } from '@/components/common';
import { ReportViewer } from '@/components/results/ReportViewer';
import { ChatInterface } from '@/components/chat/ChatInterface';
import { getDiagnosisReport, getDiagnosisResult } from '@/api/endpoints/diagnosis';
import { DiagnosisResult } from '@/types/diagnosis.types';
import toast from 'react-hot-toast';
import { FileText, MessageSquare, ArrowLeft } from 'lucide-react';

export const Results: React.FC = () => {
  const { taskId } = useParams<{ taskId: string }>();
  const navigate = useNavigate();
  const [result, setResult] = useState<DiagnosisResult | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [showChat, setShowChat] = useState(false);

  useEffect(() => {
    const fetchResult = async () => {
      if (!taskId) {
        navigate('/upload');
        return;
      }

      setIsLoading(true);
      try {
        const numericId = Number(taskId);
        if (!isNaN(numericId) && Number.isInteger(numericId)) {
          // ✅ ID numérique → appel direct /report
          const data = await getDiagnosisReport(numericId);
          setResult(data);
        } else {
          // ✅ UUID → appel /result pour obtenir le rapport et le diagnosis_id
          const data = await getDiagnosisResult(taskId);
          
          if (data.status === 'COMPLETED' && data.diagnosis_id) {
            // ✅ On a le diagnosis_id → on appelle /report pour avoir les images
            const reportData = await getDiagnosisReport(data.diagnosis_id);
            setResult(reportData);
          } else if (data.report) {
            // ⚠️ Si le diagnosis_id n'est pas encore disponible, on affiche le rapport sans images
            console.warn('⚠️ diagnosis_id non disponible, affichage du rapport sans images');
            setResult(data);
            // On peut réessayer dans quelques secondes
            setTimeout(async () => {
              try {
                const retryData = await getDiagnosisResult(taskId);
                if (retryData.diagnosis_id) {
                  const reportData = await getDiagnosisReport(retryData.diagnosis_id);
                  setResult(reportData);
                  toast.success('Images chargées avec succès');
                }
              } catch (e) {
                console.warn('Impossible de récupérer les images après tentative');
              }
            }, 3000);
          } else {
            setResult(data);
          }
        }
      } catch (error) {
        console.error('Erreur de récupération du résultat:', error);
        toast.error('Erreur lors de la récupération du résultat');
        navigate('/upload');
      } finally {
        setIsLoading(false);
      }
    };

    fetchResult();
  }, [taskId, navigate]);

  if (isLoading) {
    return (
      <Layout>
        <div className="flex h-[60vh] items-center justify-center">
          <div className="text-center">
            <Spinner size="lg" />
            <p className="mt-4 text-gray-500 dark:text-gray-400">Chargement des résultats...</p>
          </div>
        </div>
      </Layout>
    );
  }

  if (!result) {
    return (
      <Layout>
        <div className="flex h-[60vh] items-center justify-center">
          <div className="text-center">
            <p className="text-gray-500 dark:text-gray-400">Aucun résultat disponible</p>
            <Button className="mt-4" onClick={() => navigate('/upload')}>
              Nouvelle analyse
            </Button>
          </div>
        </div>
      </Layout>
    );
  }

  return (
    <Layout>
      <div className="space-y-4 py-0">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <button
              onClick={() => navigate('/dashboard')}
              className="rounded-full p-2 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
            >
              <ArrowLeft className="h-5 w-5 text-gray-500" />
            </button>
            
            <div>
              <p><p><p><h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">Résultats de l'analyse</h1></p></p></p>
              <p className="text-sm text-gray-500 dark:text-gray-400">
                Rapport généré pour le diagnostic #{result.diagnosis_id || 'inconnu'}
              </p>
            </div>
          </div>
          <div className="flex gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => setShowChat(!showChat)}
              className="flex items-center gap-2"
            >
              <MessageSquare className="h-4 w-4" />
              {showChat ? 'Masquer le chat' : 'Assistant IA'}
            </Button>
            <Button size="sm" onClick={() => navigate('/upload')}>
              <FileText className="mr-2 h-4 w-4" />
              Nouvelle analyse
            </Button>
          </div>
        </div>

        <div className="grid gap-4 lg:grid-cols-5">  {/* 5 colonnes */}
  <div className="lg:col-span-4">             {/* Rapport sur 4/5 */}
    <ReportViewer result={result} />
  </div>
{showChat && taskId && (
  <div className="lg:col-span-1">
    <div className="h-[900px] overflow-hidden rounded-xl border border-gray-200 bg-white dark:border-gray-700 dark:bg-gray-800">
      <ChatInterface
        key={taskId}
        taskId={taskId}
        isOpen={true}
        onToggle={() => setShowChat(false)} // ✅ ferme le chat
      />
    </div>
  </div>
)}
</div>
      </div>
    </Layout>
  );
};