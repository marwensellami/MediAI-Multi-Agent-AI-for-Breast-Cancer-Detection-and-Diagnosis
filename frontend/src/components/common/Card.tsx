// src/components/common/Card.tsx
import React from 'react';
import { cn } from '@/lib/utils';

export interface CardProps extends React.HTMLAttributes<HTMLDivElement> {
  title?: string;
  subtitle?: string;
  gradient?: string;  // 👈 Nouvelle propriété
}

export const Card = React.forwardRef<HTMLDivElement, CardProps>(
  ({ className, title, subtitle, gradient, children, ...props }, ref) => {
    return (
      <div
        ref={ref}
        className={cn(
          'rounded-xl bg-white p-6 shadow-sm dark:bg-gray-800',
          gradient && `bg-gradient-to-br ${gradient}`,
          className
        )}
        {...props}
      >
        {(title || subtitle) && (
          <div className="mb-4">
            {title && <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">{title}</h3>}
            {subtitle && <p className="text-sm text-gray-500 dark:text-gray-400">{subtitle}</p>}
          </div>
        )}
        {children}
      </div>
    );
  }
);
Card.displayName = 'Card';