// src/components/results/LesionCard.tsx
import React from 'react';
import { cn } from '@/lib/utils';
import { Card, Badge } from '@/components/common';
import { BiradsBadge } from './BiradsBadge';
import { formatPercent, formatMeasure } from '@/utils/formatters';

export interface LesionCardProps {
  index: number;
  lesion: {
    bbox: number[];
    size_mm?: number | null;
    malignancy_label?: string;
    malignancy_score?: number;
    area_mm2?: number | null;
    perimeter_mm?: number | null;
    circularity?: number | null;
    birads?: string | null;
  };
  className?: string;
  onClick?: () => void;
  isActive?: boolean;
}

export const LesionCard: React.FC<LesionCardProps> = ({
  index,
  lesion,
  className,
  onClick,
  isActive,
}) => {
  const isMalignant = lesion.malignancy_label?.toLowerCase().includes('malin');
  const score = lesion.malignancy_score ?? 0;
  const size = lesion.size_mm;
  const area = lesion.area_mm2;
  const perimeter = lesion.perimeter_mm;
  const circularity = lesion.circularity;

  return (
    <Card
      className={cn(
        'cursor-pointer transition-all hover:shadow-md',
        isActive && 'ring-2 ring-primary-500',
        className
      )}
      onClick={onClick}
    >
      <div className="flex items-start justify-between">
        <div>
          <h4 className="font-medium text-gray-900 dark:text-gray-100">
            Lésion {index + 1}
          </h4>
          <p className="text-sm text-gray-500 dark:text-gray-400">
            Localisation : {lesion.bbox.join(', ')}
          </p>
        </div>
        <Badge variant={isMalignant ? 'danger' : 'success'}>
          {lesion.malignancy_label || 'Indéterminé'}
        </Badge>
      </div>

      <div className="mt-3 grid grid-cols-2 gap-2 text-sm">
        {size !== null && size !== undefined && (
          <div>
            <span className="text-gray-500 dark:text-gray-400">Taille :</span>
            <span className="ml-1 font-medium">{formatMeasure(size)} mm</span>
          </div>
        )}
        <div>
          <span className="text-gray-500 dark:text-gray-400">Score :</span>
          <span className="ml-1 font-medium">{formatPercent(score)}</span>
        </div>
        {area !== null && area !== undefined && (
          <div>
            <span className="text-gray-500 dark:text-gray-400">Aire :</span>
            <span className="ml-1 font-medium">{formatMeasure(area)} mm²</span>
          </div>
        )}
        {perimeter !== null && perimeter !== undefined && (
          <div>
            <span className="text-gray-500 dark:text-gray-400">Périmètre :</span>
            <span className="ml-1 font-medium">{formatMeasure(perimeter)} mm</span>
          </div>
        )}
        {circularity !== null && circularity !== undefined && (
          <div className="col-span-2">
            <span className="text-gray-500 dark:text-gray-400">Circularité :</span>
            <span className="ml-1 font-medium">{circularity.toFixed(3)}</span>
          </div>
        )}
      </div>

      {lesion.birads && (
        <div className="mt-3">
          <BiradsBadge birads={lesion.birads} size="sm" />
        </div>
      )}
    </Card>
  );
};