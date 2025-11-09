import { useState } from 'react';
import './App.css';
import { DashboardLayout } from './components/Layout/DashboardLayout';
import { PlaySidebar } from './components/PlaySidebar/PlaySidebar';
import { MainView } from './components/MainView/MainView';

function App() {
  const [selectedPlayId, setSelectedPlayId] = useState<string | null>(null);

  return (
    <DashboardLayout
      main={<MainView selectedPlayId={selectedPlayId} />}
      sidebar={<PlaySidebar onSelectPlay={setSelectedPlayId} />}
    />
  );
}

export default App;
