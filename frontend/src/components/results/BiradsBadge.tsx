// src/components/results/BiradsBadge.tsx
import React from 'react';
import { cn } from '@/lib/utils';
import { getBiradsInfo } from '@/utils/biradsColors';

interface BiradsBadgeProps {
  birads: string | number;
  showLabel?: boolean;
  className?: string;
  size?: 'sm' | 'md' | 'lg';
}

export const BiradsBadge: React.FC<BiradsBadgeProps> = ({
  birads,
  showLabel = true,
  className,
  size = 'md',
}) => {
  const key = typeof birads === 'number' ? String(birads) : birads;
  const info = getBiradsInfo(key);

  const sizeClasses = {
    sm: 'px-2 py-0.5 text-xs',
    md: 'px-2.5 py-1 text-sm',
    lg: 'px-3 py-1.5 text-base',
  };

  return (
    <span
      className={cn(
        'inline-flex items-center rounded-full font-semibold',
        info.color.bg,
        info.color.text,
        sizeClasses[size],
        className
      )}
    >
      <span className="mr-1">BI-RADS {birads}</span>
      {showLabel && <span className="opacity-75">{info.label}</span>}
    </span>
  );
};