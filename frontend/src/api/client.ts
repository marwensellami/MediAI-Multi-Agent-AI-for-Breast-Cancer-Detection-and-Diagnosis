// src/api/client.ts
import axios, { AxiosError, AxiosInstance, AxiosResponse } from 'axios';
import toast from 'react-hot-toast';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8001/api/v1';

export const apiClient: AxiosInstance = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 60000, // 60 secondes
});

// Intercepteur pour les requêtes
apiClient.interceptors.request.use(
  (config) => {
    // Ajouter un token d'authentification si nécessaire
    // const token = localStorage.getItem('token');
    // if (token) {
    //   config.headers.Authorization = `Bearer ${token}`;
    // }
    return config;
  },
  (error) => Promise.reject(error)
);

// Intercepteur pour les réponses
apiClient.interceptors.response.use(
  (response: AxiosResponse) => response,
  (error: AxiosError) => {
    if (error.response) {
      const status = error.response.status;
      const data = error.response.data as { detail?: string };

      if (status === 401) {
        // Non autorisé
        toast.error('Session expirée, veuillez vous reconnecter.');
      } else if (status === 404) {
        // Not found (silencieux, géré par les appels)
        return Promise.reject(error);
      } else if (status >= 500) {
        toast.error('Erreur serveur. Veuillez réessayer plus tard.');
      } else if (data?.detail) {
        toast.error(data.detail);
      }
    } else if (error.code === 'ECONNABORTED') {
      toast.error('La requête a expiré. Veuillez réessayer.');
    } else if (error.message) {
      toast.error(error.message);
    }
    return Promise.reject(error);
  }
);

export default apiClient;