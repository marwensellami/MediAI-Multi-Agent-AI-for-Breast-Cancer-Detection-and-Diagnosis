// src/components/layout/Header.tsx
import React, { useState } from 'react';
import { Menu, Moon, Sun, Search, Bell, User } from 'lucide-react';
import { useUserStore } from '@/store/userStore';
import { Button } from '@/components/common/Button';

interface HeaderProps {
  sidebarOpen: boolean;
  setSidebarOpen: (open: boolean) => void;
}

export const Header: React.FC<HeaderProps> = ({ sidebarOpen, setSidebarOpen }) => {
  const { preferences, toggleTheme } = useUserStore();
  const isDark = preferences.theme === 'dark';
  const [] = useState(false);

  return (
    <header className="sticky top-0 z-30 flex h-16 items-center justify-between border-b border-gray-200 bg-white px-4 dark:border-gray-700 dark:bg-gray-800">
      {/* Partie gauche */}
      <div className="flex items-center gap-3">
        <button
          className="lg:hidden"
          onClick={() => setSidebarOpen(!sidebarOpen)}
        >
          <Menu className="h-6 w-6 text-gray-500 dark:text-gray-400" />
        </button>
        <h1 className="text-lg font-semibold text-gray-900 dark:text-white">
          Tableau de bord
        </h1>
      </div>

      {/* Partie droite */}
      <div className="flex items-center gap-2">
        {/* Barre de recherche */}
        <div className="relative hidden md:block">
          <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-gray-400 dark:text-gray-500" />
          <input
            type="text"
            placeholder="Rechercher..."
            className="h-9 w-48 rounded-lg border border-gray-200 bg-gray-50 pl-9 pr-4 text-sm text-gray-900 placeholder:text-gray-400 focus:border-primary-500 focus:outline-none focus:ring-2 focus:ring-primary-500/20 dark:border-gray-700 dark:bg-gray-800 dark:text-white dark:placeholder:text-gray-500"
          />
        </div>

        {/* Notifications */}
        <button className="relative rounded-full p-2 text-gray-500 hover:bg-gray-100 dark:text-gray-400 dark:hover:bg-gray-700">
          <Bell className="h-5 w-5" />
          <span className="absolute right-1.5 top-1.5 h-2 w-2 rounded-full bg-red-500" />
        </button>

        {/* Thème */}
        <Button
          variant="ghost"
          size="sm"
          onClick={toggleTheme}
          className="rounded-full p-2"
        >
          {isDark ? <Sun className="h-5 w-5" /> : <Moon className="h-5 w-5" />}
        </Button>

        {/* Profil */}
        <div className="flex items-center gap-2 pl-2">
          <div className="h-9 w-9 rounded-full bg-gradient-to-br from-primary-500 to-primary-600 flex items-center justify-center text-sm font-semibold text-white">
            <User className="h-5 w-5" />
          </div>
          <span className="hidden text-sm font-medium text-gray-700 dark:text-gray-300 sm:inline">
            Dr. Marwen
          </span>
        </div>
      </div>
    </header>
  );
};