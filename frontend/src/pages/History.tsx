// src/pages/History.tsx
import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Layout } from '@/components/layout/Layout';
import { Card, Button, Spinner } from '@/components/common';
import { ArrowLeft } from 'lucide-react';
import apiClient from '@/api/client';
import toast from 'react-hot-toast';

interface DiagnosisSummary {
  id: number;
  patient_id: string;
  patient_name: string;
  date: string;
  birads: string;
  confidence: number;
  status: string;
  file_name: string;
}

export const History: React.FC = () => {
  const navigate = useNavigate();
  const [diagnoses, setDiagnoses] = useState<DiagnosisSummary[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchHistory = async () => {
      try {
        const response = await apiClient.get('/diagnosis');
        setDiagnoses(response.data);
      } catch (error) {
        toast.error('Impossible de charger l\'historique');
        console.error(error);
      } finally {
        setLoading(false);
      }
    };
    fetchHistory();
  }, []);

  const getBiradsBadgeClass = (birads: string) => {
    if (!birads) return 'bg-gray-100 text-gray-800';
    const num = parseInt(birads.match(/\d+/)?.[0] || '0');
    if (num <= 2) return 'birads-2';
    if (num === 3) return 'birads-3';
    if (num === 4) return 'birads-4';
    if (num >= 5) return 'birads-5';
    return 'bg-gray-100 text-gray-800';
  };

  if (loading) {
    return (
      <Layout>
        <div className="flex justify-center items-center h-64">
          <Spinner size="lg" />
        </div>
      </Layout>
    );
  }

  return (
    <Layout>
      <div className="space-y-6 py-6">
        <div className="flex items-center gap-3">
          <button onClick={() => navigate('/dashboard')} className="rounded-full p-2 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors">
            <ArrowLeft className="h-5 w-5 text-gray-500" />
          </button>
          <div>
            <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">Historique des analyses</h1>
            <p className="text-sm text-gray-500 dark:text-gray-400">Toutes les analyses effectuées</p>
          </div>
        </div>

        <Card>
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
              <thead className="bg-gray-50 dark:bg-gray-800/50">
                <tr>
                  <th className="px-4 py-3 text-left text-xs font-medium uppercase tracking-wider text-gray-500 dark:text-gray-400">Patient</th>
                  <th className="px-4 py-3 text-left text-xs font-medium uppercase tracking-wider text-gray-500 dark:text-gray-400">Fichier</th>
                  <th className="px-4 py-3 text-left text-xs font-medium uppercase tracking-wider text-gray-500 dark:text-gray-400">Date</th>
                  <th className="px-4 py-3 text-left text-xs font-medium uppercase tracking-wider text-gray-500 dark:text-gray-400">BI-RADS</th>
                  <th className="px-4 py-3 text-left text-xs font-medium uppercase tracking-wider text-gray-500 dark:text-gray-400">Confiance</th>
                  <th className="px-4 py-3 text-left text-xs font-medium uppercase tracking-wider text-gray-500 dark:text-gray-400">Statut</th>
                  <th className="px-4 py-3 text-right text-xs font-medium uppercase tracking-wider text-gray-500 dark:text-gray-400">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200 bg-white dark:divide-gray-700 dark:bg-gray-800">
                {diagnoses.map((item) => (
                  <tr key={item.id} className="transition-colors hover:bg-gray-50 dark:hover:bg-gray-700/50">
                    <td className="whitespace-nowrap px-4 py-3 text-sm font-medium text-gray-900 dark:text-gray-100">{item.patient_id}</td>
                    <td className="whitespace-nowrap px-4 py-3 text-sm text-gray-500 dark:text-gray-400">
                      {item.file_name || 'N/A'}
                    </td>
                    <td className="whitespace-nowrap px-4 py-3 text-sm text-gray-500 dark:text-gray-400">{new Date(item.date).toLocaleDateString('fr-FR')}</td>
                    <td className="whitespace-nowrap px-4 py-3">
                      <span className={`inline-flex rounded-full px-3 py-1 text-xs font-medium ${getBiradsBadgeClass(item.birads)}`}>
                        {item.birads || 'N/A'}
                      </span>
                    </td>
                    <td className="whitespace-nowrap px-4 py-3 text-sm text-gray-500 dark:text-gray-400">{Math.round(item.confidence * 100)}%</td>
                    <td className="whitespace-nowrap px-4 py-3">
                      <span className="inline-flex items-center gap-2 text-sm">
                        <span className="h-2 w-2 rounded-full bg-green-500" />
                        <span className="text-green-600 dark:text-green-400">{item.status}</span>
                      </span>
                    </td>
                    <td className="whitespace-nowrap px-4 py-3 text-right">
                      <Button variant="ghost" size="sm" onClick={() => navigate(`/results/${item.id}`)} className="text-primary-600 hover:text-primary-700 dark:text-primary-400 dark:hover:text-primary-300">
                        Voir
                      </Button>
                    </td>
                  </tr>
                ))}
                {diagnoses.length === 0 && (
                  <tr>
                    <td colSpan={6} className="px-4 py-6 text-center text-gray-500">Aucun diagnostic enregistré</td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </Card>
      </div>
    </Layout>
  );
};