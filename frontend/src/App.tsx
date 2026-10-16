// src/App.tsx
import { BrowserRouter } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';
import AppRoutes from './routes';
import { useUserStore } from './store/userStore';
import { useEffect } from 'react';

function App() {
  const { preferences } = useUserStore();

  // Appliquer le thème au chargement
  useEffect(() => {
    const root = document.documentElement;
    if (preferences.theme === 'dark') {
      root.classList.add('dark');
    } else {
      root.classList.remove('dark');
    }
  }, [preferences.theme]);

  return (
    <BrowserRouter>
      <Toaster
        position="top-right"
        toastOptions={{
          duration: 4000,
          style: {
            background: '#363636',
            color: '#fff',
            padding: '12px 16px',
            borderRadius: '8px',
          },
          success: {
            style: {
              background: '#22c55e',
            },
          },
          error: {
            style: {
              background: '#ef4444',
            },
          },
          loading: {
            style: {
              background: '#3b82f6',
            },
          },
        }}
      />
      <AppRoutes />
    </BrowserRouter>
  );
}

export default App;