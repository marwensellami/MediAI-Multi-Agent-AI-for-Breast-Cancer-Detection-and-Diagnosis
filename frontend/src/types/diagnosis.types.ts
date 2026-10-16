// src/types/diagnosis.types.ts

export interface Lesion {
  bbox: number[]; // [x1, y1, x2, y2]
  confidence: number;
  class_name: string;
  malignancy_label?: string;
  malignancy_score?: number;
  size_mm?: number | null;
  area_mm2?: number | null;
  perimeter_mm?: number | null;
  circularity?: number | null;
  mask?: string | null; // base64 ou chemin vers le masque
  modality?: string;
}

export interface DiagnosisSummary {
  total_lesions: number;
  malignant_lesions: number;
  benign_lesions: number;
  message?: string;
}

export interface DiagnosisMetadata {
  modality: string;
  patient_id?: string;
  study_date?: string;
  view?: string;
  pixel_spacing?: number[] | null;
  original_image_shape?: [number, number];
  model_input_shape?: [number, number];
  patient_position?: string;
  [key: string]: unknown;
}

export interface DiagnosisResult {
  diagnosis_id: number;
  report: string;
  birads: string;
  confidence: number;
  status: string;
  detections?: any[];
  original_image?: string;    // base64
  annotated_image?: string;   // base64
  file_name?: string; 
  view?: string;
  density?: string;
}

export interface DiagnosisProgress {
  task_id: string;
  status: 'PENDING' | 'RUNNING' | 'COMPLETED' | 'FAILED';
  percent: number;
  steps: { step: string; percent: number; message: string }[];
  error?: string | null;
}

export interface CreateDiagnosisResponse {
  task_id: string;
}