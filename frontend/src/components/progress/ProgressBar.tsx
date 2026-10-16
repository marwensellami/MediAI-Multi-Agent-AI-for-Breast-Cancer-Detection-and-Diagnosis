// src/components/progress/ProgressBar.tsx
import React from 'react';
import { cn } from '@/lib/utils';

interface ProgressBarProps {
  value: number;
  max?: number;
  label?: string;
  className?: string;
  showPercentage?: boolean;
}

export const ProgressBar: React.FC<ProgressBarProps> = ({
  value,
  max = 100,
  label,
  className,
  showPercentage = true,
}) => {
  const percentage = Math.min(100, Math.max(0, (value / max) * 100));

  return (
    <div className={cn('space-y-2', className)}>
      {(label || showPercentage) && (
        <div className="flex justify-between text-sm">
          {label && <span className="text-gray-700 dark:text-gray-300">{label}</span>}
          {showPercentage && (
            <span className="font-medium text-gray-900 dark:text-gray-100">
              {Math.round(percentage)}%
            </span>
          )}
        </div>
      )}
      <div className="h-2.5 w-full overflow-hidden rounded-full bg-gray-200 dark:bg-gray-700">
        <div
          className="h-full rounded-full bg-primary-600 transition-all duration-500 ease-out"
          style={{ width: `${percentage}%` }}
          role="progressbar"
          aria-valuenow={percentage}
          aria-valuemin={0}
          aria-valuemax={100}
        />
      </div>
    </div>
  );
};