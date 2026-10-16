// src/routes.tsx
import { Routes, Route, Navigate } from 'react-router-dom';
import { Dashboard } from './pages/Dashboard';
import { Upload } from './pages/Upload';
import { Progress } from './pages/Progress';
import { Results } from './pages/Results';
import { Settings } from './pages/Settings';

import { History } from '@/pages/History';
function AppRoutes() {
  return (
    <Routes>
      <Route path="/" element={<Navigate to="/dashboard" replace />} />
      <Route path="/dashboard" element={<Dashboard />} />
      <Route path="/upload" element={<Upload />} />
      <Route path="/progress/:taskId" element={<Progress />} />
      <Route path="/results/:taskId" element={<Results />} />
      <Route path="/settings" element={<Settings />} />
      <Route path="/history" element={<History />} />
      <Route path="*" element={<div className="flex h-screen items-center justify-center">
        <div className="text-center">
          <h1 className="text-4xl font-bold text-gray-900 dark:text-gray-100">404</h1>
          <p className="text-gray-500 dark:text-gray-400">Page non trouvée</p>
          <a href="/dashboard" className="mt-4 inline-block text-primary-600 hover:underline">
            Retour au tableau de bord
          </a>
        </div>
      </div>} />
    </Routes>
  );
}

export default AppRoutes;