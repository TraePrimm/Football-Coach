import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Text } from '@chakra-ui/react';
import { api, type FrameData, type PlayStatus } from '../../api/client';
import { VideoPlayerView } from '../VideoPlayer/VideoPlayerView';
import { PlaybackControls } from '../PlaybackControls/PlaybackControls';
import { InfoPanel } from '../InfoPanel/InfoPanel';

interface MainViewProps {
  selectedPlayId: string | null;
}

export const MainView: React.FC<MainViewProps> = ({ selectedPlayId }) => {
  const [playStatus, setPlayStatus] = useState<PlayStatus['status']>('not_started');
  const [processingProgress, setProcessingProgress] = useState<number>(0);
  const [totalFrames, setTotalFrames] = useState<number>(0);
  const [currentFrame, setCurrentFrame] = useState<number>(0);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [lastFrameData, setLastFrameData] = useState<FrameData | null>(null);
  const [selectedPlayerId, setSelectedPlayerId] = useState<number | null>(null);
  const [isTransitioning, setIsTransitioning] = useState<boolean>(false);
  const pollIntervalRef = useRef<number | null>(null);
  const rafRef = useRef<number | null>(null);
  const readyLockRef = useRef<boolean>(false);

  // Debug: log key state changes
  useEffect(() => {
    // eslint-disable-next-line no-console
    console.log('[MainView] state', { selectedPlayId, playStatus, processingProgress, totalFrames, currentFrame, isPlaying, readyLock: readyLockRef.current });
  }, [selectedPlayId, playStatus, processingProgress, totalFrames, currentFrame, isPlaying]);

  // Debug: log selected player changes
  useEffect(() => {
    // eslint-disable-next-line no-console
    console.log('[MainView] selectedPlayerId', selectedPlayerId);
  }, [selectedPlayerId]);

  // Fetch status on play change
  useEffect(() => {
    if (!selectedPlayId) {
      setPlayStatus('not_started');
      setProcessingProgress(0);
      setTotalFrames(0);
      setCurrentFrame(0);
      setIsPlaying(false);
      setLastFrameData(null);
      readyLockRef.current = false;
      return;
    }

    // Reset state for the newly selected play so previous play UI doesn't persist
    stopPolling();
    stopPlaying();
    setIsTransitioning(true);
    setPlayStatus('not_started');
    setProcessingProgress(0);
    setTotalFrames(0);
    setCurrentFrame(0);
    setLastFrameData(null);
    setSelectedPlayerId(null);
    readyLockRef.current = false;

    let isMounted = true;

    const fetchStatus = async () => {
      try {
        // eslint-disable-next-line no-console
        console.log('[MainView] fetchStatus: requesting', { selectedPlayId });
        const status = await api.getPlayStatus(selectedPlayId);
        if (!isMounted) return;
        const nextStatus = readyLockRef.current ? 'ready' : status.status;
        setPlayStatus(nextStatus);
        // eslint-disable-next-line no-console
        console.log('[MainView] fetchStatus: response', { apiStatus: status.status, progress: status.progress, appliedStatus: nextStatus, readyLock: readyLockRef.current });
        setProcessingProgress(status.progress ?? 0);

        if (status.status === 'ready') {
          readyLockRef.current = true;
          // eslint-disable-next-line no-console
          console.log('[MainView] ready state reached, locking ready and ensuring totalFrames');
          // We need total frames. We'll read from the first frame response or keep previous
          if (totalFrames === 0) {
            try {
              const fd = await api.getFrame(selectedPlayId, 0);
              if (!isMounted) return;
              setTotalFrames(fd.total_frames);
              // eslint-disable-next-line no-console
              console.log('[MainView] fetched first frame to infer totalFrames', { totalFrames: fd.total_frames });
            } catch {}
          }
        } else if (status.status === 'processing' && !readyLockRef.current) {
          // eslint-disable-next-line no-console
          console.log('[MainView] status processing -> startPolling()');
          startPolling();
        }
      } catch (e) {
        if (!isMounted) return;
        if (!readyLockRef.current) {
          setPlayStatus('error');
          // eslint-disable-next-line no-console
          console.log('[MainView] fetchStatus error', e);
        }
      }
    };

    fetchStatus();

    return () => {
      isMounted = false;
      stopPolling();
      stopPlaying();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedPlayId]);

  const startPolling = () => {
    if (pollIntervalRef.current) {
      // eslint-disable-next-line no-console
      console.log('[MainView] startPolling ignored: already polling');
      return;
    }
    // eslint-disable-next-line no-console
    console.log('[MainView] startPolling begin');
    pollIntervalRef.current = window.setInterval(async () => {
      if (!selectedPlayId) return;
      try {
        const status = await api.getPlayStatus(selectedPlayId);
        const nextStatus = readyLockRef.current ? 'ready' : status.status;
        setPlayStatus(nextStatus);
        setProcessingProgress(status.progress ?? 0);
        // eslint-disable-next-line no-console
        console.log('[MainView] polling tick', { apiStatus: status.status, progress: status.progress, appliedStatus: nextStatus, readyLock: readyLockRef.current });
        if (status.status === 'ready') {
          readyLockRef.current = true;
          stopPolling();
          // Ensure total frames is known
          if (totalFrames === 0) {
            try {
              const fd = await api.getFrame(selectedPlayId, 0);
              setTotalFrames(fd.total_frames);
              // eslint-disable-next-line no-console
              console.log('[MainView] polling fetched first frame to infer totalFrames', { totalFrames: fd.total_frames });
            } catch {}
          }
        }
      } catch {
        if (!readyLockRef.current) setPlayStatus('error');
        stopPolling();
      }
    }, 2000);
  };

  const stopPolling = () => {
    if (pollIntervalRef.current) {
      clearInterval(pollIntervalRef.current);
      pollIntervalRef.current = null;
      // eslint-disable-next-line no-console
      console.log('[MainView] stopPolling');
    }
  };

  const togglePlay = useCallback(() => {
    setIsPlaying(p => !p);
  }, []);

  const stopPlaying = () => {
    if (rafRef.current) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
    setIsPlaying(false);
    // eslint-disable-next-line no-console
    console.log('[MainView] stopPlaying');
  };

  // Stable frame data handler to avoid refetch loops in VideoPlayerView
  const handleFrameDataStable = useCallback((fd: FrameData) => {
    setLastFrameData(fd);
    setTotalFrames(prev => (prev === 0 ? fd.total_frames : prev));
    setIsTransitioning(false);
    // eslint-disable-next-line no-console
    console.log('[MainView] onFrameData', { totalFrames: fd.total_frames });
  }, []);

  // Playback loop using rAF; advances ~30fps capped by frames
  useEffect(() => {
    if (!isPlaying) {
      if (rafRef.current) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
      return;
    }

    let last = performance.now();
    const fps = 30;
    const frameDuration = 1000 / fps;

    const loop = (now: number) => {
      if (!isPlaying) return;
      const delta = now - last;
      if (delta >= frameDuration) {
        last = now;
        setCurrentFrame(prev => {
          if (totalFrames <= 0) return prev;
          const next = prev + 1;
          if (next >= totalFrames) {
            stopPlaying();
            return totalFrames - 1;
          }
          return next;
        });
      }
      rafRef.current = requestAnimationFrame(loop);
    };

    rafRef.current = requestAnimationFrame(loop);
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    };
  }, [isPlaying, totalFrames]);

  const handleProcess = async () => {
    if (!selectedPlayId) return;
    try {
      const res = await api.processPlay(selectedPlayId);
      if (res.status === 'processing') {
        setPlayStatus('processing');
        setProcessingProgress(0);
        startPolling();
      }
    } catch {
      setPlayStatus('error');
    }
  };

  const layout = useMemo(() => ({
    container: {
      display: 'grid',
      gridTemplateRows: '1fr auto 240px',
      gridGap: '12px',
      height: '100%',
      overflow: 'hidden'
    } as React.CSSProperties,
    topLeft: {
      background: '#000',
      position: 'relative',
      borderRadius: 8,
      overflow: 'hidden',
      minHeight: 0
    } as React.CSSProperties,
    controls: {
      minHeight: 0
    } as React.CSSProperties,
    bottomLeft: {
      background: '#fff',
      border: '1px solid #E2E8F0',
      borderRadius: 8,
      overflow: 'hidden',
      minHeight: 0
    } as React.CSSProperties,
    statusBox: {
      background: '#fff',
      border: '1px solid #E2E8F0',
      borderRadius: 8,
      padding: 16,
      color: '#1A202C',
    } as React.CSSProperties,
  }), []);

  return (
    <div style={layout.container}>
      <div style={layout.topLeft}>
        <VideoPlayerView
          playId={selectedPlayId ?? ''}
          currentFrame={currentFrame}
          onFrameData={handleFrameDataStable}
          isPlaying={isPlaying}
          selectedPlayerId={selectedPlayerId ?? undefined}
          onSelectPlayer={(id: number) => setSelectedPlayerId(id)}
        />
        {/* Overlay states to reduce flicker and keep layout stable */}
        {(isTransitioning || !selectedPlayId || playStatus !== 'ready') && (
          <div style={{ position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'rgba(255, 255, 255, 1)' }}>
            <div style={{ ...layout.statusBox, background: 'transparent', border: 'none' }}>
              {!selectedPlayId && (
                <Text>Select a play from the right to begin.</Text>
              )}
              {selectedPlayId && playStatus === 'not_started' && (
                <div>
                  <Text style={{ marginBottom: 12 }}>This play is not processed yet.</Text>
                  <button onClick={handleProcess} style={{ background: '#3182ce', color: 'white', border: 'none', padding: '8px 12px', borderRadius: 6, cursor: 'pointer' }}>Process Play</button>
                </div>
              )}
              {selectedPlayId && playStatus === 'processing' && (
                <div>
                  <Text style={{ marginBottom: 8 }}>Processing...</Text>
                  <div style={{ height: 8, background: '#E2E8F0', borderRadius: 4, overflow: 'hidden' }}>
                    <div style={{ width: `${processingProgress}%`, height: '100%', background: '#3182ce' }} />
                  </div>
                  <Text style={{ marginTop: 8, fontSize: 12, color: '#4A5568' }}>{processingProgress}%</Text>
                </div>
              )}
              {selectedPlayId && playStatus === 'error' && (
                <Text color="red.500">An error occurred. Try again.</Text>
              )}
              {isTransitioning && playStatus === 'ready' && (
                <Text>Loading play...</Text>
              )}
            </div>
          </div>
        )}
      </div>

      <div style={layout.controls}>
        <PlaybackControls
          isPlaying={isPlaying}
          currentFrame={currentFrame}
          totalFrames={totalFrames}
          onPlayPause={togglePlay}
          onFrameChange={setCurrentFrame}
          onStepForward={() => setCurrentFrame(f => Math.min((totalFrames - 1), f + 1))}
          onStepBackward={() => setCurrentFrame(f => Math.max(0, f - 1))}
        />
      </div>

      <div style={layout.bottomLeft}>
        <InfoPanel
          playersInFrame={lastFrameData?.players ?? []}
          selectedPlayId={selectedPlayId || ''}
          selectedPlayerId={selectedPlayerId ?? undefined}
        />
      </div>
    </div>
  );
};
