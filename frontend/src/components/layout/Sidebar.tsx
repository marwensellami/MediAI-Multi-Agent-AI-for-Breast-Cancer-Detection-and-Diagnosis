// src/components/layout/Sidebar.tsx
import React from 'react';
import { NavLink } from 'react-router-dom';
import { LayoutDashboard, Upload, Clock, Settings, X } from 'lucide-react';
import { cn } from '@/lib/utils';

interface SidebarProps {
  open: boolean;
  setOpen: (open: boolean) => void;
}

const navigation = [
  { name: 'Tableau de bord', href: '/dashboard', icon: LayoutDashboard },
  { name: 'Nouvelle analyse', href: '/upload', icon: Upload },
  { name: 'Historique', href: '/history', icon: Clock },
  { name: 'Paramètres', href: '/settings', icon: Settings },
];

export const Sidebar: React.FC<SidebarProps> = ({ open, setOpen }) => {
  return (
    <>
      {/* Overlay mobile */}
      {open && (
        <div
          className="fixed inset-0 z-40 bg-black/50 lg:hidden"
          onClick={() => setOpen(false)}
        />
      )}

      <aside
        className={cn(
          'fixed inset-y-0 left-0 z-50 w-64 transform bg-white shadow-lg transition-transform duration-300 ease-in-out dark:bg-gray-900 lg:relative lg:translate-x-0',
          open ? 'translate-x-0' : '-translate-x-full'
        )}
      >
        {/* Logo */}
        <div className="flex h-16 items-center justify-between border-b border-gray-200 px-4 dark:border-gray-800">
          <div className="flex items-center gap-2">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary-600">
              <span className="text-sm font-bold text-white">AI</span>
            </div>
            <span className="text-lg font-bold text-gray-900 dark:text-white">MediAI</span>
          </div>
          <button className="lg:hidden" onClick={() => setOpen(false)}>
            <X className="h-5 w-5 text-gray-500" />
          </button>
        </div>

        {/* Navigation */}
        <nav className="mt-4 px-3">
          <div className="space-y-1">
            {navigation.map((item) => (
              <NavLink
                key={item.name}
                to={item.href}
                className={({ isActive }) =>
                  cn(
                    'group flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-all duration-200',
                    isActive
                      ? 'bg-primary-50 text-primary-700 dark:bg-primary-900/20 dark:text-primary-400'
                      : 'text-gray-700 hover:bg-gray-100 dark:text-gray-300 dark:hover:bg-gray-800'
                  )
                }
              >
                <item.icon
                  className={cn(
                    'h-5 w-5',
                    'text-gray-400 group-hover:text-gray-600 dark:text-gray-500 dark:group-hover:text-gray-300'
                  )}
                />
                <span>{item.name}</span>
              </NavLink>
            ))}
          </div>
        </nav>

        {/* Footer */}
        <div className="absolute bottom-0 left-0 right-0 border-t border-gray-200 p-4 dark:border-gray-800">
          <div className="flex items-center gap-3">
            <div className="h-8 w-8 rounded-full bg-primary-100 flex items-center justify-center text-primary-700 dark:bg-primary-900/30 dark:text-primary-400">
              <span className="text-sm font-semibold">M</span>
            </div>
            <div className="text-sm">
              <p className="font-medium text-gray-900 dark:text-white">Dr. Marwen</p>
              <p className="text-xs text-gray-500 dark:text-gray-400">Radiologue</p>
            </div>
          </div>
        </div>
      </aside>
    </>
  );
};