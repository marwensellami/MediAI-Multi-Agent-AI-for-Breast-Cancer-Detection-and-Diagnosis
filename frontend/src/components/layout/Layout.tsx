import React, { useState } from 'react';
import { Sidebar } from './Sidebar';
import { Header } from './Header';

interface LayoutProps {
  children: React.ReactNode;
}

export const Layout: React.FC<LayoutProps> = ({ children }) => {
  const [sidebarOpen, setSidebarOpen] = useState(true);

  return (
    <div className="flex min-h-screen bg-gray-50 dark:bg-gray-900">
      <Sidebar open={sidebarOpen} setOpen={setSidebarOpen} />
      <div className="flex flex-1 flex-col">
        <Header sidebarOpen={sidebarOpen} setSidebarOpen={setSidebarOpen} />
        <main className="flex-1 p-4 md:p-6">
          {children}
        </main>
      </div>
    </div>
  );
};