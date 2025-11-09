import React, { useEffect, useMemo, useState } from 'react';
import { Text } from '@chakra-ui/react';
import { api } from '../../api/client';

interface PlayerInFrame {
  id: number;
  number: string;
  team: 'home' | 'away';
  role: string;
}

interface PlayerStats {
  id: number;
  team: string;
  distance: number;
  speed: number;
}

interface InfoPanelProps {
  playersInFrame: PlayerInFrame[];
  selectedPlayId: string;
  selectedPlayerId?: number;
}

export const InfoPanel: React.FC<InfoPanelProps> = ({ playersInFrame, selectedPlayId, selectedPlayerId }) => {
  const [currentTab, setCurrentTab] = useState<'events' | 'statistics'>('events');
  const [stats, setStats] = useState<PlayerStats[]>([]);
  const [loadingStats, setLoadingStats] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (currentTab !== 'statistics') return;
    let isMounted = true;

    const fetchStats = async () => {
      setLoadingStats(true);
      setError(null);

      // Try a hypothetical endpoint first
      try {
        const data = await api.getAllStats(selectedPlayId);
        if (isMounted) {
          // If a player is selected and all_stats came back empty, try fetching that one directly
          if ((data?.length ?? 0) === 0 && selectedPlayerId != null) {
            try {
              const d = await api.getPlayerStats(selectedPlayId, selectedPlayerId);
              setStats(d ? [{ id: selectedPlayerId, team: d.team, distance: d.distance ?? 0, speed: d.speed ?? 0 }] : []);
            } catch {
              setStats([]);
            } finally {
              setLoadingStats(false);
            }
            return;
          }
          // If a player is selected but not present in all_stats, try fetching it directly
          if (selectedPlayerId != null && !data.some((s: any) => Number(s.id) === selectedPlayerId)) {
            try {
              const d = await api.getPlayerStats(selectedPlayId, selectedPlayerId);
              setStats(d ? [{ id: selectedPlayerId, team: d.team, distance: d.distance ?? 0, speed: d.speed ?? 0 }] : data);
            } catch {
              setStats(data);
            } finally {
              setLoadingStats(false);
            }
            return;
          }
          setStats(data);
          setLoadingStats(false);
        }
        return;
      } catch {}

      // Fallback: fetch for current players in frame
      try {
        const items: PlayerStats[] = [];
        for (const p of playersInFrame) {
          const d = await api.getPlayerStats(selectedPlayId, p.id);
          items.push({ id: p.id, team: d.team, distance: d.distance ?? 0, speed: d.speed ?? 0 });
        }
        if (isMounted) setStats(items);
      } catch (e) {
        if (isMounted) setError('Failed to load stats');
      } finally {
        if (isMounted) setLoadingStats(false);
      }
    };

    fetchStats();

    return () => { isMounted = false; };
  }, [currentTab, playersInFrame, selectedPlayId, selectedPlayerId]);

  // When a player is selected, jump to Statistics tab
  useEffect(() => {
    if (selectedPlayerId != null) setCurrentTab('statistics');
  }, [selectedPlayerId]);

  const styles = useMemo(() => ({
    container: { height: '100%', display: 'flex', flexDirection: 'column' } as React.CSSProperties,
    tabs: { display: 'flex', gap: 8, borderBottom: '1px solid #CBD5E0', padding: '8px 12px', background: '#F7FAFC' } as React.CSSProperties,
    tab: (active: boolean) => ({
      padding: '6px 10px',
      cursor: 'pointer',
      borderRadius: 6,
      background: active ? '#E2E8F0' : 'transparent',
      color: active ? '#1A202C' : '#2D3748',
      border: active ? '1px solid #A0AEC0' : '1px solid transparent'
    }) as React.CSSProperties,
    table: { width: '100%', borderCollapse: 'separate', borderSpacing: 0, background: '#FFFFFF', border: '1px solid #CBD5E0', borderRadius: 8, overflow: 'hidden' } as React.CSSProperties,
    th: { textAlign: 'left', background: '#E2E8F0', color: '#1A202C', fontWeight: 700, fontSize: 12, padding: 10, borderBottom: '1px solid #CBD5E0' } as React.CSSProperties,
    td: { fontSize: 13, padding: 10, color: '#1A202C', borderBottom: '1px solid #E2E8F0' } as React.CSSProperties,
    body: { overflow: 'auto', flex: 1, background: '#F7FAFC', padding: 12 } as React.CSSProperties,
  }), []);

  return (
    <div style={styles.container}>
      {/* Tabs */}
      <div style={styles.tabs}>
        <button style={styles.tab(currentTab === 'events')} onClick={() => setCurrentTab('events')}>Events</button>
        <button style={styles.tab(currentTab === 'statistics')} onClick={() => setCurrentTab('statistics')}>Statistics</button>
      </div>

      {/* Content */}
      <div style={styles.body}>
        {currentTab === 'events' ? (
          <table style={styles.table}>
            <thead>
              <tr>
                <th style={styles.th}>ID</th>
                <th style={styles.th}>Number</th>
                <th style={styles.th}>Team</th>
                <th style={styles.th}>Role</th>
              </tr>
            </thead>
            <tbody>
              {playersInFrame.map(p => (
                <tr key={p.id} style={selectedPlayerId != null && Number(p.id) === selectedPlayerId ? { background: '#FEF3C7' } : undefined}>
                  <td style={styles.td}>{p.id}</td>
                  <td style={styles.td}>{p.number}</td>
                  <td style={styles.td}>{p.team}</td>
                  <td style={styles.td}>{p.role}</td>
                </tr>
              ))}
              {playersInFrame.length === 0 && (
                <tr>
                  <td style={styles.td} colSpan={4}>No players detected in this frame.</td>
                </tr>
              )}
            </tbody>
          </table>
        ) : (
          <div style={{ padding: 12 }}>
            {loadingStats && <Text>Loading stats...</Text>}
            {error && <Text color="red.500">{error}</Text>}
            {!loadingStats && !error && (
              (() => {
                const list = selectedPlayerId != null ? stats.filter(s => Number(s.id) === selectedPlayerId) : stats;
                return (
              <table style={styles.table}>
                <thead>
                  <tr>
                    <th style={styles.th}>Player ID</th>
                    <th style={styles.th}>Team</th>
                    <th style={styles.th}>Total Distance</th>
                    <th style={styles.th}>Max Speed</th>
                  </tr>
                </thead>
                <tbody>
                  {list.map(s => (
                    <tr key={s.id}>
                      <td style={styles.td}>{s.id}</td>
                      <td style={styles.td}>{s.team}</td>
                      <td style={styles.td}>{s.distance?.toFixed(1)} ft</td>
                      <td style={styles.td}>{s.speed?.toFixed(1)} mph</td>
                    </tr>
                  ))}
                  {list.length === 0 && (
                    <tr>
                      <td style={styles.td} colSpan={4}>{stats.length > 0 && selectedPlayerId != null ? `No stats found for player ${selectedPlayerId}` : 'No stats available.'}</td>
                    </tr>
                  )}
                </tbody>
              </table>
                );
              })()
            )}
          </div>
        )}
      </div>
    </div>
  );
}
;
