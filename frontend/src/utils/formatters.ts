// src/utils/formatters.ts

/**
 * Formate une date en chaîne lisible (DD/MM/YYYY)
 */
export function formatDate(date: string | Date): string {
  const d = typeof date === 'string' ? new Date(date) : date;
  return d.toLocaleDateString('fr-FR', {
    day: '2-digit',
    month: '2-digit',
    year: 'numeric',
  });
}

/**
 * Formate un nombre en pourcentage avec une décimale
 */
export function formatPercent(value: number): string {
  return `${(value * 100).toFixed(1)} %`;
}

/**
 * Formate un nombre avec 1 décimale pour les mesures
 */
export function formatMeasure(value: number | null | undefined): string {
  if (value === null || value === undefined || isNaN(value)) {
    return 'N/A';
  }
  return value.toFixed(1);
}

/**
 * Tronque un texte à une longueur maximale
 */
export function truncateText(text: string, maxLength: number = 200): string {
  if (text.length <= maxLength) return text;
  return text.substring(0, maxLength) + '...';
}

/**
 * Formate un BI-RADS pour l'affichage (ex: "BI-RADS 4B")
 */
export function formatBirads(birads: string | null | undefined): string {
  if (!birads) return 'Non classifié';
  if (birads.startsWith('BI-RADS')) return birads;
  return `BI-RADS ${birads}`;
}