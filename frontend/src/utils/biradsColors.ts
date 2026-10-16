// src/utils/biradsColors.ts

export const BIRADS_COLORS: Record<string, { bg: string; text: string; border: string }> = {
  '1': { bg: 'bg-green-100', text: 'text-green-800', border: 'border-green-300' },
  '2': { bg: 'bg-green-100', text: 'text-green-800', border: 'border-green-300' },
  '3': { bg: 'bg-yellow-100', text: 'text-yellow-800', border: 'border-yellow-300' },
  '4A': { bg: 'bg-orange-100', text: 'text-orange-800', border: 'border-orange-300' },
  '4B': { bg: 'bg-orange-200', text: 'text-orange-900', border: 'border-orange-400' },
  '4C': { bg: 'bg-orange-300', text: 'text-orange-900', border: 'border-orange-500' },
  '5': { bg: 'bg-red-100', text: 'text-red-800', border: 'border-red-300' },
  '6': { bg: 'bg-red-200', text: 'text-red-900', border: 'border-red-400' },
};

export const BIRADS_LABELS: Record<string, string> = {
  '1': 'Négatif',
  '2': 'Bénin',
  '3': 'Probablement bénin',
  '4A': 'Suspicion faible',
  '4B': 'Suspicion modérée',
  '4C': 'Suspicion élevée',
  '5': 'Très suspect',
  '6': 'Malignité confirmée',
};

export function getBiradsInfo(level: string | number): { color: { bg: string; text: string; border: string }; label: string } {
  const key = typeof level === 'number' ? String(level) : level;
  const color = BIRADS_COLORS[key] || BIRADS_COLORS['1'];
  const label = BIRADS_LABELS[key] || 'Non classifié';
  return { color, label };
}