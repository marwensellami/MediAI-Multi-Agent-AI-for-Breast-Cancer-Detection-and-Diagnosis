// src/components/results/ReportViewer.tsx
import React, { useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import { Download, Copy } from 'lucide-react';
import { Button } from '@/components/common';

import { downloadPDF } from '@/utils/downloadPDF';
import toast from 'react-hot-toast';
import { DiagnosisResult } from '@/types/diagnosis.types';

interface ReportViewerProps {
  result: DiagnosisResult;
}

export const ReportViewer: React.FC<ReportViewerProps> = ({ result }) => {
  const reportRef = useRef<HTMLDivElement>(null);

  // LOG POUR DIAGNOSTIC
  useEffect(() => {
    console.log('📸 ReportViewer - result reçu:', {
      hasOriginal: !!result.original_image,
      originalLength: result.original_image?.length || 0,
      hasAnnotated: !!result.annotated_image,
      annotatedLength: result.annotated_image?.length || 0,
      preview: result.original_image?.substring(0, 50) || 'N/A',
    });
  }, [result]);

  const handleCopyReport = async () => {
    try {
      await navigator.clipboard.writeText(result.report);
      toast.success('Rapport copié dans le presse-papiers');
    } catch (error) {
      toast.error('Erreur lors de la copie');
    }
  };

  const handleExportPDF = async () => {
    try {
      // ✅ Construire le nom du fichier
      const baseName = result.file_name || 'rapport_medical';
      // Nettoyer le nom (enlever l'extension et les caractères spéciaux)
      const cleanName = baseName.replace(/\.[^.]+$/, '').replace(/[^a-zA-Z0-9_-]/g, '_');
      const fileName = `rapport_medical_${cleanName}`;
      await downloadPDF('report-content', fileName);
      toast.success('PDF exporté avec succès');
    } catch (error) {
      toast.error('Erreur lors de l\'export PDF');
    }
  };

  // Fonction pour construire une URL d'image valide
  const getImageSrc = (base64: string | undefined) => {
    if (!base64) return '';
    // Si déjà un data URL complet, le retourner tel quel
    if (base64.startsWith('data:image')) {
      return base64;
    }
    // Sinon, ajouter le préfixe PNG
    return `data:image/png;base64,${base64}`;
  };

  const detections = result.detections || [];
  const hasLesions = detections.length > 0;

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-bold text-gray-900 dark:text-gray-100">Rapport médical</h2>
        <div className="flex gap-2">
          <Button variant="outline" size="sm" onClick={handleCopyReport}>
            <Copy className="mr-2 h-4 w-4" />
            Copier
          </Button>
          <Button size="sm" onClick={handleExportPDF}>
            <Download className="mr-2 h-4 w-4" />
            Exporter PDF
          </Button>
        </div>
      </div>

      <div
        id="report-content"
        ref={reportRef}
        className="max-h-[600px] overflow-y-auto rounded-lg border border-gray-200 bg-white p-6 dark:border-gray-700 dark:bg-gray-800"
      >
        {/* En-tête */}
        <div className="mb-4 border-b border-gray-200 pb-4 dark:border-gray-700">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
            RAPPORT RADIOLOGIQUE MAMMAIRE
          </h3>
          <div className="mt-2 grid grid-cols-2 gap-2 text-sm text-gray-600 dark:text-gray-400">
            <div><span className="font-medium">Patient :</span> {result.diagnosis_id || 'Inconnu'}</div>
            <div><span className="font-medium">Date :</span> {new Date().toLocaleDateString('fr-FR')}</div>
            <div className="col-span-2"><span className="font-medium">Modalités :</span> MG</div>
            {result.view && (
    <div><span className="font-medium">Vue :</span> {result.view}</div>  // ✅ AJOUT
  )}
          </div>
        </div>

        {/* BI-RADS */}
        <div className="mb-4 flex items-center gap-3">
          
          
          <span className="ml-auto text-sm text-gray-500 dark:text-gray-400">
            Confiance : {Math.round((result.confidence || 0) * 100)}%
          </span>
        </div>

        {/* Images */}
        <div className="mb-6 grid grid-cols-1 gap-4 md:grid-cols-2">
          {/* Original */}
          <div className="rounded-lg border border-gray-200 p-2 dark:border-gray-700">
            <p className="mb-2 text-center text-sm font-medium text-gray-600 dark:text-gray-400">
              Mammographie originale
            </p>
            {result.original_image ? (
              <img
                src={getImageSrc(result.original_image)}
                alt="Mammographie originale"
                className="w-full rounded border border-gray-200 dark:border-gray-700"
                style={{ maxHeight: '400px', objectFit: 'contain' }}
                onError={() => console.error('❌ Erreur chargement image originale')}
                onLoad={() => console.log('✅ Image originale chargée')}
              />
            ) : (
              <div className="flex h-48 items-center justify-center bg-gray-100 dark:bg-gray-700">
                <span className="text-sm text-gray-500 dark:text-gray-400">Image non disponible</span>
              </div>
            )}
          </div>

          {/* Annotée */}
          <div className="rounded-lg border border-gray-200 p-2 dark:border-gray-700">
            <p className="mb-2 text-center text-sm font-medium text-gray-600 dark:text-gray-400">
              {hasLesions ? 'Lésions détectées' : 'Aucune lésion détectée'}
            </p>
            {result.annotated_image ? (
              <img
                src={getImageSrc(result.annotated_image)}
                alt="Lésions détectées"
                className="w-full rounded border border-gray-200 dark:border-gray-700"
                style={{ maxHeight: '400px', objectFit: 'contain' }}
                onError={() => console.error('❌ Erreur chargement image annotée')}
                onLoad={() => console.log('✅ Image annotée chargée')}
              />
            ) : result.original_image ? (
              <img
                src={getImageSrc(result.original_image)}
                alt="Mammographie"
                className="w-full rounded border border-gray-200 dark:border-gray-700"
                style={{ maxHeight: '400px', objectFit: 'contain' }}
              />
            ) : (
              <div className="flex h-48 items-center justify-center bg-gray-100 dark:bg-gray-700">
                <span className="text-sm text-gray-500 dark:text-gray-400">Image non disponible</span>
              </div>
            )}
            {hasLesions && (
              <div className="mt-2 flex flex-wrap gap-2 text-xs">
                <span className="inline-flex items-center gap-1">
                  <span className="inline-block h-3 w-3 rounded-full bg-red-500" /> Malin
                </span>
                <span className="inline-flex items-center gap-1">
                  <span className="inline-block h-3 w-3 rounded-full bg-green-500" /> Bénin
                </span>
              </div>
            )}
          </div>
        </div>

        {/* Rapport textuel */}
        {result.report && (
          <div className="mt-6 border-t border-gray-200 pt-6 dark:border-gray-700">
            <h4 className="mb-3 font-medium text-gray-900 dark:text-gray-100">Détails du rapport :</h4>
            <div className="prose prose-sm max-w-none dark:prose-invert prose-headings:font-bold prose-headings:text-gray-900 prose-headings:dark:text-gray-100 prose-h1:text-2xl prose-h2:text-xl prose-h3:text-lg prose-p:text-gray-700 prose-p:dark:text-gray-300 prose-ul:list-disc prose-li:text-gray-700 prose-li:dark:text-gray-300 prose-strong:text-gray-900 prose-strong:dark:text-gray-100 prose-hr:border-gray-200 dark:prose-hr:border-gray-700">
             <ReactMarkdown
  components={{
    h1: ({ children }) => <h1 className="text-2xl font-bold my-4">{children}</h1>,
    h2: ({ children }) => <h2 className="text-xl font-semibold my-3">{children}</h2>,
    h3: ({ children }) => <h3 className="text-lg font-semibold my-2">{children}</h3>,
    p: ({ children }) => <p className="mb-2 leading-relaxed">{children}</p>,
  }}
>
  {result.report}
</ReactMarkdown>
            </div>
          </div>
        )}

        <div className="mt-4 border-t border-gray-200 pt-4 text-xs text-gray-400 dark:border-gray-700">
          <p>Ce rapport est une aide à la décision et ne remplace pas l'avis d'un radiologue.</p>
        </div>
      </div>
    </div>
  );
};