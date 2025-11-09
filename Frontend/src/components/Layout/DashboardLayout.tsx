import React from 'react';

interface DashboardLayoutProps {
  sidebar: React.ReactNode;
  main: React.ReactNode;
}

export const DashboardLayout: React.FC<DashboardLayoutProps> = ({ sidebar, main }) => {
  return (
    <div style={{
      display: 'grid',
      gridTemplateColumns: '1fr 320px',
      gridTemplateRows: '100vh',
      width: '100%',
      height: '100vh',
      overflow: 'hidden',
      background: '#f7fafc'
    }}>
      {/* Left Main Area */}
      <div style={{
        overflow: 'hidden',
        padding: '16px'
      }}>
        {main}
      </div>

      {/* Right Sidebar */}
      <div style={{
        borderLeft: '1px solid #E2E8F0',
        overflow: 'hidden'
      }}>
        {sidebar}
      </div>
    </div>
  );
};
