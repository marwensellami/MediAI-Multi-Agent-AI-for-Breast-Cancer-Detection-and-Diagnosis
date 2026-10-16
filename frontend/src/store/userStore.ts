// src/store/userStore.ts

import { create } from 'zustand';
import { persist } from 'zustand/middleware';

interface UserPreferences {
  theme: 'light' | 'dark';
  language: 'fr' | 'en';
}

interface UserState {
  preferences: UserPreferences;
  setTheme: (theme: 'light' | 'dark') => void;
  setLanguage: (language: 'fr' | 'en') => void;
  toggleTheme: () => void;
}

export const useUserStore = create<UserState>()(
  persist(
    (set) => ({
      preferences: {
        theme: 'light',
        language: 'fr',
      },
      setTheme: (theme) =>
        set((state) => ({
          preferences: { ...state.preferences, theme },
        })),
      setLanguage: (language) =>
        set((state) => ({
          preferences: { ...state.preferences, language },
        })),
      toggleTheme: () =>
        set((state) => ({
          preferences: {
            ...state.preferences,
            theme: state.preferences.theme === 'light' ? 'dark' : 'light',
          },
        })),
    }),
    {
      name: 'user-preferences',
    }
  )
);