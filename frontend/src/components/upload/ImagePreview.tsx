// src/components/upload/ImagePreview.tsx
import React, { useState, useEffect } from 'react';
import { cn } from '@/lib/utils';
import { Spinner } from '@/components/common';

interface ImagePreviewProps {
  file: File | null;
  className?: string;
}

export const ImagePreview: React.FC<ImagePreviewProps> = ({ file, className }) => {
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    if (!file) {
      setImageUrl(null);
      return;
    }

    setIsLoading(true);
    const reader = new FileReader();
    reader.onload = (e) => {
      setImageUrl(e.target?.result as string);
      setIsLoading(false);
    };
    reader.onerror = () => {
      setIsLoading(false);
    };
    reader.readAsDataURL(file);
  }, [file]);

  if (!file) {
    return null;
  }

  return (
    <div className={cn('relative overflow-hidden rounded-lg border border-gray-200 dark:border-gray-700', className)}>
      {isLoading ? (
        <div className="flex h-48 items-center justify-center bg-gray-50 dark:bg-gray-800">
          <Spinner size="lg" />
        </div>
      ) : imageUrl ? (
        <img
          src={imageUrl}
          alt="Aperçu de la mammographie"
          className="h-auto max-h-96 w-full object-contain"
        />
      ) : (
        <div className="flex h-48 items-center justify-center bg-gray-50 dark:bg-gray-800">
          <p className="text-sm text-gray-500 dark:text-gray-400">Aperçu non disponible</p>
        </div>
      )}
    </div>
  );
};