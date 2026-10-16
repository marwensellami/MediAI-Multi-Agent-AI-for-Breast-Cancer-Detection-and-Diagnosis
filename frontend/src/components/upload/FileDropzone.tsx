// src/components/upload/FileDropzone.tsx
import React, { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, File, X } from 'lucide-react';
import { cn } from '@/lib/utils';

interface FileDropzoneProps {
  onFileSelect: (file: File) => void;
  file: File | null;
  onRemoveFile: () => void;
  accept?: Record<string, string[]>;
  maxSize?: number;
  className?: string;
}

export const FileDropzone: React.FC<FileDropzoneProps> = ({
  onFileSelect,
  file,
  onRemoveFile,
  accept = {
    'image/*': ['.dcm', '.png', '.jpg', '.jpeg', '.tif', '.tiff'],
  },
  maxSize = 50 * 1024 * 1024, // 50 MB
  className,
}) => {
  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      if (acceptedFiles.length > 0) {
        onFileSelect(acceptedFiles[0]);
      }
    },
    [onFileSelect]
  );

  const { getRootProps, getInputProps, isDragActive, isDragReject } = useDropzone({
    onDrop,
    accept,
    maxSize,
    multiple: false,
  });

  return (
    <div className={cn('space-y-4', className)}>
      {file ? (
        <div className="flex items-center gap-3 rounded-lg border border-gray-200 bg-gray-50 p-3 dark:border-gray-700 dark:bg-gray-800">
          <File className="h-8 w-8 text-primary-500" />
          <div className="flex-1 min-w-0">
            <p className="truncate text-sm font-medium text-gray-900 dark:text-gray-100">
              {file.name}
            </p>
            <p className="text-xs text-gray-500 dark:text-gray-400">
              {(file.size / 1024 / 1024).toFixed(2)} MB
            </p>
          </div>
          <button
            type="button"
            onClick={onRemoveFile}
            className="rounded-full p-1 hover:bg-gray-200 dark:hover:bg-gray-700"
          >
            <X className="h-5 w-5 text-gray-500 dark:text-gray-400" />
          </button>
        </div>
      ) : (
        <div
          {...getRootProps()}
          className={cn(
            'flex cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed p-8 transition-colors',
            isDragActive && 'border-primary-500 bg-primary-50 dark:bg-primary-900/10',
            isDragReject && 'border-red-500 bg-red-50 dark:bg-red-900/10',
            !isDragActive && !isDragReject && 'border-gray-300 hover:border-primary-400 dark:border-gray-600'
          )}
        >
          <input {...getInputProps()} />
          <Upload className="h-12 w-12 text-gray-400 dark:text-gray-500" />
          <p className="mt-2 text-sm text-gray-600 dark:text-gray-400">
            {isDragActive
              ? 'Déposez le fichier ici...'
              : 'Glissez-déposez votre mammographie ou cliquez pour sélectionner'}
          </p>
          <p className="mt-1 text-xs text-gray-400 dark:text-gray-500">
            Formats acceptés : PNG, JPG, JPEG (max 50 MB)
          </p>
        </div>
      )}
    </div>
  );
};