// src/pages/Settings.tsx
import React from 'react';
import { useTranslation } from 'react-i18next';
import { Layout } from '@/components/layout/Layout';
import { Card, Button } from '@/components/common';
import { useUserStore } from '@/store/userStore';
import toast from 'react-hot-toast';

export const Settings: React.FC = () => {
  const { t, i18n } = useTranslation();
  const { preferences, setTheme, setLanguage } = useUserStore();
  const isDark = preferences.theme === 'dark';

  const handleThemeToggle = () => {
    setTheme(isDark ? 'light' : 'dark');
    toast.success(isDark ? 'Mode clair activé' : 'Mode sombre activé');
  };

  const handleLanguageChange = (lang: 'fr' | 'en') => {
    setLanguage(lang);
    i18n.changeLanguage(lang);
    toast.success(`Langue changée en ${lang === 'fr' ? 'Français' : 'English'}`);
  };

  return (
    <Layout>
      <div className="mx-auto max-w-2xl space-y-6">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">
            {t('settings.title')}
          </h1>
          <p className="text-sm text-gray-500 dark:text-gray-400">
            {t('settings.subtitle')}
          </p>
        </div>

        <Card title={t('settings.appearance')}>
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="font-medium text-gray-900 dark:text-gray-100">
                  {t('settings.darkMode')}
                </p>
                <p className="text-sm text-gray-500 dark:text-gray-400">
                  {t('settings.darkModeDesc')}
                </p>
              </div>
              <Button
                variant={isDark ? 'primary' : 'secondary'}
                onClick={handleThemeToggle}
              >
                {isDark ? 'Mode sombre activé' : 'Mode clair activé'}
              </Button>
            </div>
          </div>
        </Card>

        <Card title={t('settings.language')}>
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="font-medium text-gray-900 dark:text-gray-100">
                  {t('settings.language')}
                </p>
                <p className="text-sm text-gray-500 dark:text-gray-400">
                  {t('settings.languageDesc')}
                </p>
              </div>
              <div className="flex gap-2">
                <Button
                  variant={preferences.language === 'fr' ? 'primary' : 'secondary'}
                  size="sm"
                  onClick={() => handleLanguageChange('fr')}
                >
                  Français
                </Button>
                <Button
                  variant={preferences.language === 'en' ? 'primary' : 'secondary'}
                  size="sm"
                  onClick={() => handleLanguageChange('en')}
                >
                  English
                </Button>
              </div>
            </div>
          </div>
        </Card>

        <Card title={t('settings.about')}>
          <div className="space-y-2 text-sm text-gray-600 dark:text-gray-400">
            <p>
              <span className="font-medium">Breast Cancer AI Framework</span> v1.0.0
            </p>
            <p>{t('settings.version')}</p>
            <p className="text-xs text-gray-400 dark:text-gray-500">
              {t('settings.copyright')}
            </p>
          </div>
        </Card>
      </div>
    </Layout>
  );
};