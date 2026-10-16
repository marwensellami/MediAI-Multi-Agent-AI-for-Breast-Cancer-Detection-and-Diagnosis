// src/components/dashboard/DashboardCharts.tsx
import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, PieChart, Pie, Cell, ResponsiveContainer } from 'recharts';

interface BiradsData {
  name: string;
  value: number;
}

const COLORS = ['#22c55e', '#22c55e', '#eab308', '#f97316', '#ef4444'];

export const DashboardCharts: React.FC<{ biradsData: BiradsData[] }> = ({ biradsData }) => {
  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {/* Histogramme BI-RADS */}
      <div className="bg-white dark:bg-gray-800 p-4 rounded-xl shadow">
        <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-4">Répartition BI-RADS</h3>
        <ResponsiveContainer width="100%" height={250}>
          <BarChart data={biradsData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Bar dataKey="value" fill="#3b82f6" />
          </BarChart>
        </ResponsiveContainer>
      </div>
      
      {/* Camembert BI-RADS */}
      <div className="bg-white dark:bg-gray-800 p-4 rounded-xl shadow">
        <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-4">Proportion BI-RADS</h3>
        <ResponsiveContainer width="100%" height={250}>
          <PieChart>
            <Pie
              data={biradsData}
              cx="50%"
              cy="50%"
              labelLine={false}
              label={({ name, percent }) => `${name} ${(percent ?? 0 * 100).toFixed(0)}%`}
              outerRadius={80}
              fill="#8884d8"
              dataKey="value"
            >
              {biradsData.map((_entry, index) => (
                <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
              ))}
            </Pie>
            <Tooltip />
          </PieChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};