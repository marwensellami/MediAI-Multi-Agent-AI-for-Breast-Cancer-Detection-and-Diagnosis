// src/pages/Dashboard.tsx
import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Activity, AlertCircle, CheckCircle, Plus } from 'lucide-react';
import { Card, Button } from '@/components/common';
import { Layout } from '@/components/layout/Layout';
import apiClient from '@/api/client';
import toast from 'react-hot-toast';
import { DashboardCharts } from '@/components/dashboard/DashboardCharts';

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

interface BiradsData {
  name: string;
  value: number;
}

export const Dashboard: React.FC = () => {
  const navigate = useNavigate();
  const [diagnoses, setDiagnoses] = useState<DiagnosisSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const [biradsChartData, setBiradsChartData] = useState<BiradsData[]>([]);

  const filteredDiagnoses = diagnoses.filter(d =>
  String(d.patient_id).toLowerCase().includes(searchTerm.toLowerCase()) ||
  String(d.patient_name).toLowerCase().includes(searchTerm.toLowerCase())
);

  const fetchDiagnoses = async () => {
    try {
      const response = await apiClient.get('/diagnosis');
      setDiagnoses(response.data);

      // Récupérer les statistiques pour les graphiques
      try {
        const statsResponse = await apiClient.get('/dashboard/stats');
        const biradsData = statsResponse.data.birads_distribution || {};
        const chartData = Object.entries(biradsData).map(([name, value]) => ({
          name,
          value: value as number,
        }));
        setBiradsChartData(chartData);
      } catch (statsError) {
        console.warn('Impossible de charger les statistiques BI-RADS', statsError);
      }
    } catch (error) {
      toast.error('Impossible de charger les diagnostics');
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDiagnoses();
  }, []);

  // Calcul des statistiques
  const total = diagnoses.length;
  const birads4 = diagnoses.filter(d => d.birads && d.birads.includes('4')).length;
  const birads5 = diagnoses.filter(d => d.birads && d.birads.includes('5')).length;
  const normal = diagnoses.filter(d => d.birads && (d.birads.includes('1') || d.birads.includes('2'))).length;

  const stats = [
    {
      label: 'Total analyses',
      value: total,
      icon: Activity,
      gradient: 'from-blue-50 to-blue-100 dark:from-blue-900/20 dark:to-blue-800/20',
      borderColor: 'border-blue-500',
      iconColor: 'text-blue-500',
    },
    {
      label: 'BI-RADS 4',
      value: birads4,
      icon: AlertCircle,
      gradient: 'from-orange-50 to-orange-100 dark:from-orange-900/20 dark:to-orange-800/20',
      borderColor: 'border-orange-500',
      iconColor: 'text-orange-500',
    },
    {
      label: 'BI-RADS 5',
      value: birads5,
      icon: AlertCircle,
      gradient: 'from-red-50 to-red-100 dark:from-red-900/20 dark:to-red-800/20',
      borderColor: 'border-red-500',
      iconColor: 'text-red-500',
    },
    {
      label: 'Examens normaux',
      value: normal,
      icon: CheckCircle,
      gradient: 'from-green-50 to-green-100 dark:from-green-900/20 dark:to-green-800/20',
      borderColor: 'border-green-500',
      iconColor: 'text-green-500',
    },
  ];

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
          <div className="text-gray-500">Chargement...</div>
        </div>
      </Layout>
    );
  }

  return (
    <Layout>
      <div className="space-y-6 py-6">
        <div className="flex flex-col items-start justify-between gap-4 sm:flex-row sm:items-center">
          <div>
            <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">Tableau de bord</h1>
            <p className="text-sm text-gray-500 dark:text-gray-400">Vue d'ensemble des analyses mammographiques</p>
          </div>
          <Button onClick={() => navigate('/upload')} className="flex items-center gap-2">
            <Plus className="h-4 w-4" />
            Nouvelle analyse
          </Button>
        </div>

        {/* Statistiques */}
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {stats.map((stat) => (
            <div
              key={stat.label}
              className={`rounded-xl border-l-4 ${stat.borderColor} bg-gradient-to-br ${stat.gradient} p-5 shadow-sm transition-all duration-200 hover:shadow-md`}
            >
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-600 dark:text-gray-300">{stat.label}</p>
                  <p className="text-3xl font-bold text-gray-900 dark:text-gray-100">{stat.value}</p>
                </div>
                <div className={`rounded-full bg-white/50 p-3 dark:bg-black/20 ${stat.iconColor}`}>
                  <stat.icon className="h-6 w-6" />
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Graphiques */}
        <DashboardCharts biradsData={biradsChartData} />

        {/* Tableau des diagnostics récents */}
        <Card>
          <div className="flex flex-col items-start justify-between gap-4 sm:flex-row sm:items-center">
            <div>
              <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100">Diagnostics récents</h2>
              <p className="text-sm text-gray-500 dark:text-gray-400">Dernières analyses effectuées</p>
            </div>
            <div className="flex items-center gap-3">
              <input
                type="text"
                placeholder="Rechercher un patient..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="rounded-lg border border-gray-300 px-3 py-1.5 text-sm focus:border-primary-500 focus:outline-none dark:border-gray-600 dark:bg-gray-700 dark:text-white"
              />
              <Button variant="outline" size="sm" onClick={() => navigate('/history')}>
                Voir tout
              </Button>
            </div>
          </div>

          <div className="mt-4 overflow-x-auto rounded-lg border border-gray-200 dark:border-gray-700">
            <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
              <thead className="bg-gray-50 dark:bg-gray-800/50">
                <tr>
                  <th className="px-4 py-3 text-left text-xs font-medium uppercase tracking-wider text-gray-500 dark:text-gray-400">Patient</th>
                  <th className="px-4 py-3 text-left text-xs font-medium uppercase tracking-wider text-gray-500 dark:text-gray-400">Fichier</th>
                  <th className="px-4 py-3 text-left text-xs font-medium uppercase tracking-wider text-gray-500 dark:text-gray-400">Date</th>
                  <th className="px-4 py-3 text-left text-xs font-medium uppercase tracking-wider text-gray-500 dark:text-gray-400">BI-RADS</th>
                  <th className="px-4 py-3 text-left text-xs font-medium uppercase tracking-wider text-gray-500 dark:text-gray-400">Statut</th>
                  <th className="px-4 py-3 text-right text-xs font-medium uppercase tracking-wider text-gray-500 dark:text-gray-400">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200 bg-white dark:divide-gray-700 dark:bg-gray-800">
                {filteredDiagnoses.slice(0, 5).map((item) => (
                  <tr key={item.id} className="transition-colors hover:bg-gray-50 dark:hover:bg-gray-700/50">
                    <td className="whitespace-nowrap px-4 py-3 text-sm font-medium text-gray-900 dark:text-gray-100">
                      {item.patient_id}
                    </td>
                    <td className="whitespace-nowrap px-4 py-3 text-sm text-gray-500 dark:text-gray-400">
                    {item.file_name || 'N/A'}
                    </td>
                    <td className="whitespace-nowrap px-4 py-3 text-sm text-gray-500 dark:text-gray-400">
                      {new Date(item.date).toLocaleDateString('fr-FR')}
                    </td>
                    <td className="whitespace-nowrap px-4 py-3">
                      <span className={`inline-flex rounded-full px-3 py-1 text-xs font-medium ${getBiradsBadgeClass(item.birads)}`}>
                        {item.birads || 'N/A'}
                      </span>
                    </td>
                    <td className="whitespace-nowrap px-4 py-3">
                      <span className="inline-flex items-center gap-2 text-sm">
                        <span className="h-2 w-2 rounded-full bg-green-500" />
                        <span className="text-green-600 dark:text-green-400">{item.status}</span>
                      </span>
                    </td>
                    <td className="whitespace-nowrap px-4 py-3 text-right">
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => navigate(`/results/${item.id}`)}
                        className="text-primary-600 hover:text-primary-700 dark:text-primary-400 dark:hover:text-primary-300"
                      >
                        Voir
                      </Button>
                    </td>
                  </tr>
                ))}
                {filteredDiagnoses.length === 0 && (
                  <tr>
                    <td colSpan={5} className="px-4 py-6 text-center text-gray-500">
                      {searchTerm ? 'Aucun patient ne correspond à votre recherche' : 'Aucun diagnostic enregistré'}
                    </td>
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