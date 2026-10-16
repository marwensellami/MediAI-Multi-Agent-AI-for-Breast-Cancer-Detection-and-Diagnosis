// src/components/results/ImageViewer.tsx
import React, { useState, useRef, useEffect } from 'react';
import { ZoomIn, ZoomOut, Move } from 'lucide-react';
import { cn } from '@/lib/utils';
import { Button } from '@/components/common';

interface LesionOverlay {
  bbox: number[];
  color?: string;
  label?: string;
  mask?: string | null; // base64 ou url
}

interface ImageViewerProps {
  imageUrl: string;
  lesions?: LesionOverlay[];
  className?: string;
  onLesionClick?: (index: number) => void;
}

export const ImageViewer: React.FC<ImageViewerProps> = ({
  imageUrl,
  lesions = [],
  className,
  onLesionClick,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const imgRef = useRef<HTMLImageElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [scale, setScale] = useState(1);
  const [position, setPosition] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });

  const minScale = 0.5;
  const maxScale = 3;

  const handleZoomIn = () => {
    setScale(Math.min(maxScale, scale + 0.25));
  };

  const handleZoomOut = () => {
    setScale(Math.max(minScale, scale - 0.25));
  };

  const handleReset = () => {
    setScale(1);
    setPosition({ x: 0, y: 0 });
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    setIsDragging(true);
    setDragStart({ x: e.clientX - position.x, y: e.clientY - position.y });
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (isDragging) {
      setPosition({
        x: e.clientX - dragStart.x,
        y: e.clientY - dragStart.y,
      });
    }
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  // Dessiner les overlays sur le canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    const img = imgRef.current;
    if (!canvas || !img || !img.complete) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Dessiner les lésions
    lesions.forEach((lesion, index) => {
      const [x1, y1, x2, y2] = lesion.bbox;
      const color = lesion.color || '#ef4444';

      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

      // Étiquette
      ctx.fillStyle = color;
      ctx.font = '16px sans-serif';
      const label = lesion.label || `Lésion ${index + 1}`;
      ctx.fillText(label, x1, y1 - 8);

      // Si un masque est fourni, on le dessine en overlay
      if (lesion.mask) {
        // Pour l'instant, on pourrait charger le masque via une image
        // Mais on va simplement dessiner un remplissage semi-transparent
        ctx.fillStyle = 'rgba(239, 68, 68, 0.2)';
        ctx.fillRect(x1, y1, x2 - x1, y2 - y1);
      }
    });

    // Gérer les clics sur les lésions
    canvas.onclick = (e) => {
      const rect = canvas.getBoundingClientRect();
      const scaleX = canvas.width / rect.width;
      const scaleY = canvas.height / rect.height;
      const x = (e.clientX - rect.left) * scaleX;
      const y = (e.clientY - rect.top) * scaleY;

      for (let i = 0; i < lesions.length; i++) {
        const [x1, y1, x2, y2] = lesions[i].bbox;
        if (x >= x1 && x <= x2 && y >= y1 && y <= y2) {
          onLesionClick?.(i);
          break;
        }
      }
    };
  }, [lesions, onLesionClick]);

  return (
    <div className={cn('relative overflow-hidden rounded-lg bg-gray-100 dark:bg-gray-900', className)}>
      <div
        ref={containerRef}
        className="relative h-full w-full overflow-hidden"
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        style={{ cursor: isDragging ? 'grabbing' : 'grab' }}
      >
        <div
          style={{
            transform: `translate(${position.x}px, ${position.y}px) scale(${scale})`,
            transformOrigin: 'center center',
            transition: isDragging ? 'none' : 'transform 0.1s ease-out',
          }}
        >
          <img
            ref={imgRef}
            src={imageUrl}
            alt="Mammographie"
            className="h-auto w-full"
            draggable={false}
            onLoad={() => {
              // Réinitialiser le canvas après le chargement
              const canvas = canvasRef.current;
              if (canvas) {
                canvas.width = imgRef.current?.naturalWidth || 0;
                canvas.height = imgRef.current?.naturalHeight || 0;
              }
            }}
          />
          <canvas
            ref={canvasRef}
            className="absolute inset-0 h-full w-full"
            style={{ pointerEvents: 'auto' }}
          />
        </div>
      </div>

      {/* Contrôles */}
      <div className="absolute bottom-4 right-4 flex gap-2">
        <Button
          variant="secondary"
          size="sm"
          onClick={handleZoomOut}
          className="rounded-full p-2"
          aria-label="Dézoomer"
        >
          <ZoomOut className="h-4 w-4" />
        </Button>
        <Button
          variant="secondary"
          size="sm"
          onClick={handleReset}
          className="rounded-full p-2"
          aria-label="Réinitialiser"
        >
          <Move className="h-4 w-4" />
        </Button>
        <Button
          variant="secondary"
          size="sm"
          onClick={handleZoomIn}
          className="rounded-full p-2"
          aria-label="Zoomer"
        >
          <ZoomIn className="h-4 w-4" />
        </Button>
      </div>

      <div className="absolute bottom-4 left-4 text-sm text-white/70 drop-shadow-lg">
        {Math.round(scale * 100)}%
      </div>
    </div>
  );
};