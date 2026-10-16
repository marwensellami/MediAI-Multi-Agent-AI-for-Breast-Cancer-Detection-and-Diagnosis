// src/pages/Upload.tsx
import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Layout } from '@/components/layout/Layout';
import { Card, Input, Button, Spinner } from '@/components/common';
import { FileDropzone, ImagePreview } from '@/components/upload';
import { useDiagnosis } from '@/hooks/useDiagnosis';
import toast from 'react-hot-toast';
import { Upload as UploadIcon } from 'lucide-react';

export const Upload: React.FC = () => {
  const navigate = useNavigate();
  const { submit, isLoading } = useDiagnosis();

  const [age, setAge] = useState('');
  const [view] = useState('');
  const [history, setHistory] = useState('');
  const [riskFactors, setRiskFactors] = useState('');
  const [file, setFile] = useState<File | null>(null);
  
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    // ✅ L'âge n'est plus obligatoire
    // Si l'âge est saisi, on vérifie qu'il est valide
    if (age.trim() && (isNaN(Number(age)) || Number(age) < 0)) {
      toast.error('Veuillez saisir un âge valide');
      return;
    }

    if (!file) {
      toast.error('Veuillez sélectionner une mammographie');
      return;
    }

    const payload = {
      age: age.trim() ? Number(age) : undefined,
      view: view.trim() || undefined,
      history: history.trim() || undefined,
      risk_factors: riskFactors.trim() || undefined,
      mammo_file: file,
    };

    const taskId = await submit(payload);
    if (taskId) {
      navigate(`/progress/${taskId}`);
    }
  };

  return (
    <Layout>
      <div className="mx-auto max-w-3xl space-y-8 py-0">
        {/* En-tête */}
        <div className="flex items-center gap-4">
          <div className="flex h-12 w-12 items-center justify-center rounded-full bg-primary-100 dark:bg-primary-900/30">
            <UploadIcon className="h-6 w-6 text-primary-600 dark:text-primary-400" />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">
              Nouvelle analyse mammographique
            </h1>
            <p className="text-sm text-gray-500 dark:text-gray-400">
              Remplissez les informations du patient et uploadez la mammographie
            </p>
          </div>
        </div>

        <form onSubmit={handleSubmit} className="space-y-6">
          {/* Patient Info */}
          <Card>
            <div className="space-y-5">
              <div>
                <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300">
                  Informations du patient
                </h3>
                <p className="text-xs text-gray-500 dark:text-gray-400">
                  Renseignez les données cliniques du patient
                </p>
              </div>

              <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                {/* ✅ Âge n'est plus obligatoire (pas de required) */}
                <Input
                  label="Âge"
                  type="text"
                  placeholder="55"
                  value={age}
                  onChange={(e) => {
                    const value = e.target.value.replace(/[^0-9]/g, '');
                    setAge(value);
                  }}
                />

                
              </div>

              <Input
                label="Antécédents"
                placeholder="Analyse mammographique, antécédents familiaux..."
                value={history}
                onChange={(e) => setHistory(e.target.value)}
              />

              <Input
                label="Facteurs de risque"
                placeholder="Facteurs de risque, traitements en cours..."
                value={riskFactors}
                onChange={(e) => setRiskFactors(e.target.value)}
              />
            </div>
          </Card>

          {/* Upload */}
          <Card>
            <div className="space-y-4">
              <div>
                <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300">
                  Upload de la mammographie
                </h3>
                <p className="text-xs text-gray-500 dark:text-gray-400">
                  Sélectionnez un fichier mammographie de format PNG
                </p>
              </div>
              <FileDropzone
                file={file}
                onFileSelect={setFile}
                onRemoveFile={() => setFile(null)}
              />
              {file && <ImagePreview file={file} />}
            </div>
          </Card>

          {/* Boutons */}
          <div className="flex flex-col gap-3 sm:flex-row sm:gap-4">
            <Button
              type="submit"
              size="lg"
              disabled={isLoading || !file}
              className="flex-1"
            >
              {isLoading ? (
                <>
                  <Spinner size="sm" className="mr-2" />
                  Analyse en cours...
                </>
              ) : (
                "Lancer l'analyse"
              )}
            </Button>
            <Button
              type="button"
              variant="secondary"
              size="lg"
              onClick={() => navigate('/dashboard')}
            >
              Annuler
            </Button>
          </div>
        </form>
      </div>
    </Layout>
  );
};