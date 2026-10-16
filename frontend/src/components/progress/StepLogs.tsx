// src/components/progress/StepLogs.tsx
import React from 'react';
import { CheckCircle, Circle, Loader2 } from 'lucide-react';
import { cn } from '@/lib/utils';

interface StepLog {
  step: string;
  percent: number;
  message: string;
}

interface StepLogsProps {
  steps: StepLog[];
  className?: string;
}

export const StepLogs: React.FC<StepLogsProps> = ({ steps, className }) => {
  if (!steps || steps.length === 0) {
    return (
      <div className={cn('text-sm text-gray-500 dark:text-gray-400', className)}>
        En attente du début de l'analyse...
      </div>
    );
  }

  return (
    <div className={cn('space-y-3', className)}>
      {steps.map((step, index) => {
        const isCompleted = index < steps.length - 1;
        const isActive = index === steps.length - 1;

        return (
          <div key={index} className="flex items-start gap-3">
            {isCompleted ? (
              <CheckCircle className="mt-0.5 h-5 w-5 flex-shrink-0 text-green-500" />
            ) : isActive ? (
              <Loader2 className="mt-0.5 h-5 w-5 flex-shrink-0 animate-spin text-primary-500" />
            ) : (
              <Circle className="mt-0.5 h-5 w-5 flex-shrink-0 text-gray-300 dark:text-gray-600" />
            )}
            <div className="min-w-0 flex-1">
              <p className="text-sm font-medium text-gray-900 dark:text-gray-100">{step.step}</p>
              <p className="text-sm text-gray-500 dark:text-gray-400">{step.message}</p>
            </div>
            <span className="text-sm text-gray-400 dark:text-gray-500">{step.percent}%</span>
          </div>
        );
      })}
    </div>
  );
};