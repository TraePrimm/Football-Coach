import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Box, Image, Spinner, Text, Flex } from '@chakra-ui/react';
import { api } from '../../api/client';
import type { FrameData } from '../../api/client';

interface VideoPlayerViewProps {
  playId: string;
  currentFrame: number;
  onFrameData: (data: FrameData) => void;
  isPlaying: boolean;
  selectedPlayerId?: number;
  onSelectPlayer?: (id: number) => void;
}

export const VideoPlayerView: React.FC<VideoPlayerViewProps> = ({
  playId,
  currentFrame,
  onFrameData,
  isPlaying,
  selectedPlayerId,
  onSelectPlayer,
}) => {
  const [displayFrame, setDisplayFrame] = useState<FrameData | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [path, setPath] = useState<{ frame: number; x: number; y: number }[] | null>(null);
  const topdownBoxRef = useRef<HTMLDivElement | null>(null);
  const overlayBoxRef = useRef<SVGSVGElement | null>(null);
  // The actual rendered image area (letterboxed) within the container
  const [innerBox, setInnerBox] = useState<{ x: number; y: number; w: number; h: number } | null>(null);

  // On play switch, avoid clearing displayFrame to prevent flicker; clear overlays/errors only
  useEffect(() => {
    setPath(null);
    setError(null);
    // keep displayFrame so UI stays visible until new frame arrives
  }, [playId]);

  useEffect(() => {
    let isMounted = true;
    const controller = new AbortController();
    
    const fetchFrame = async () => {
      if (!playId) return;
      
      // Only show a blocking spinner if we don't have a frame yet
      if (!displayFrame) setIsLoading(true);
      setError(null);
      
      try {
        const data = await api.getFrame(playId, currentFrame, controller.signal);

        if (isMounted) {
          // Preload images to avoid visible flicker when swapping frames
          await new Promise<void>((resolve, reject) => {
            let loaded = 0;
            const done = () => { loaded += 1; if (loaded === 2) resolve(); };
            const fail = (e: any) => reject(e);

            const img1: HTMLImageElement = new window.Image();
            img1.onload = done;
            img1.onerror = fail;
            img1.src = `data:image/jpeg;base64,${data.original_image}`;

            const img2: HTMLImageElement = new window.Image();
            img2.onload = done;
            img2.onerror = fail;
            img2.src = `data:image/png;base64,${data.topdown_image}`;
          });

          if (!isMounted) return;
          setDisplayFrame(data);
          onFrameData(data);
        }
      } catch (err) {
        // Ignore abort errors as they are expected during rapid updates
        if ((err as any)?.code === 'ERR_CANCELED' || (err as any)?.name === 'CanceledError' || (err as any)?.name === 'AbortError') {
          // console.debug('Frame request canceled');
        } else {
          console.error('Error fetching frame:', err);
          if (isMounted) {
            // If we already have a frame, keep showing it and avoid an error overlay
            if (!displayFrame) {
              setError(`Error loading frame: ${err instanceof Error ? err.message : 'Unknown error'}`);
            }
          }
        }
      } finally {
        if (isMounted) {
          setIsLoading(false);
        }
      }
    };

    fetchFrame();
    
    return () => {
      isMounted = false;
      controller.abort();
    };
  }, [playId, currentFrame, onFrameData]);

  // Fetch path when a player is selected
  useEffect(() => {
    let cancelled = false;
    const run = async () => {
      if (!playId || !selectedPlayerId) {
        setPath(null);
        return;
      }
      try {
        // eslint-disable-next-line no-console
        console.log('[VideoPlayerView] fetch path for', { playId, selectedPlayerId });
        const res = await api.getPlayerPath(playId, selectedPlayerId);
        if (!cancelled) {
          setPath(res.path || []);
          // eslint-disable-next-line no-console
          console.log('[VideoPlayerView] path received', { count: res.path?.length ?? 0 });
        }
      } catch {
        if (!cancelled) setPath([]);
      }
    };
    run();
    return () => { cancelled = true; };
  }, [playId, selectedPlayerId]);

  // Compute overlay elements for path and selected marker
  const overlay = useMemo(() => {
    if (!displayFrame) return null;
    const w = displayFrame.map_width_px ?? 0;
    const h = displayFrame.map_height_px ?? 0;
    if (!w || !h) return null;
    return { w, h, path };
  }, [displayFrame, path]);

  // Recompute the letterboxed inner box whenever container size or map dims change
  useEffect(() => {
    const updateInner = () => {
      const container = topdownBoxRef.current;
      if (!container || !displayFrame) {
        setInnerBox(null);
        return;
      }
      const rect = container.getBoundingClientRect();
      const cw = rect.width;
      const ch = rect.height;
      const mapW = displayFrame.map_width_px ?? cw;
      const mapH = displayFrame.map_height_px ?? ch;
      if (!mapW || !mapH || !cw || !ch) {
        setInnerBox(null);
        return;
      }
      const r = mapW / mapH;
      const cr = cw / ch;
      let w = cw, h = ch, x = 0, y = 0;
      if (cr > r) {
        // container wider than content -> full height, centered horizontally
        h = ch;
        w = ch * r;
        x = (cw - w) / 2;
        y = 0;
      } else {
        // container taller than content -> full width, centered vertically
        w = cw;
        h = cw / r;
        x = 0;
        y = (ch - h) / 2;
      }
      setInnerBox({ x, y, w, h });
    };

    updateInner();
    window.addEventListener('resize', updateInner);
    return () => window.removeEventListener('resize', updateInner);
  }, [displayFrame]);

  const handleTopdownClick = (e: React.MouseEvent) => {
    if (!displayFrame || !onSelectPlayer) return;
    const container = topdownBoxRef.current;
    if (!container) return;
    const rect = container.getBoundingClientRect();
    const ib = innerBox ?? { x: 0, y: 0, w: rect.width, h: rect.height };
    // Click position relative to the actual rendered image area
    const cx = e.clientX - rect.left - ib.x;
    const cy = e.clientY - rect.top - ib.y;

    // Ignore clicks outside the image letterboxed area
    if (cx < 0 || cy < 0 || cx > ib.w || cy > ib.h) return;

    const mapW = displayFrame.map_width_px ?? ib.w;
    const mapH = displayFrame.map_height_px ?? ib.h;

    // Map to top-down coords using inner box scale
    const scaleX = mapW / ib.w;
    const scaleY = mapH / ib.h;
    const tx = cx * scaleX;
    const ty = cy * scaleY;

    // Find nearest player in current frame
    let bestId: number | null = null;
    let bestDist = Infinity;
    for (const p of displayFrame.players) {
      if (typeof p.x !== 'number' || typeof p.y !== 'number') continue;
      const dx = (p.x as number) - tx;
      const dy = (p.y as number) - ty;
      const d2 = dx * dx + dy * dy;
      if (d2 < bestDist) {
        bestDist = d2;
        // p.id can be a string from backend; coerce to number for consistency
        bestId = Number((p as any).id);
      }
    }
    if (bestId != null) {
      // eslint-disable-next-line no-console
      console.log('[VideoPlayerView] onSelectPlayer', bestId);
      onSelectPlayer(bestId);
    }
  };

  if (isLoading && !displayFrame) {
    return (
      <Flex justify="center" align="center" h="100%" bg="gray.100" borderRadius="md">
        <Spinner size="xl" />
      </Flex>
    );
  }

  if (error && !displayFrame) {
    return (
      <Flex justify="center" align="center" h="100%" bg="red.50" borderRadius="md">
        <Text color="red.500">{error}</Text>
      </Flex>
    );
  }

  if (!displayFrame) {
    return (
      <Flex justify="center" align="center" h="100%" bg="white" borderRadius="md">
        <Text color="gray.500">Select a play to begin</Text>
      </Flex>
    );
  }

  return (
    <Box h="100%" display="flex" flexDirection="column" bg="black" borderRadius="md" overflow="hidden">
      {/* Video Views Container */}
      <Flex flex="1" direction={{ base: 'column', md: 'row' }} h="calc(100% - 60px)">
        {/* Original Video View */}
        <Box flex={1} position="relative" borderRightWidth={{ md: '1px' }} borderBottomWidth={{ base: '1px', md: '0' }}>
          <Box position="absolute" top={2} left={2} bg="blackAlpha.700" color="white" px={2} py={1} borderRadius="md" zIndex={1}>
            <Text fontSize="sm">Original View</Text>
          </Box>
          <Image
            src={`data:image/jpeg;base64,${displayFrame.original_image}`}
            alt={`Frame ${currentFrame} - Original`}
            objectFit="contain"
            w="100%"
            h="100%"
            onError={(e) => {
              console.error('Error loading original image');
              const target = e.target as HTMLImageElement;
              target.alt = 'Failed to load original image';
            }}
          />
        </Box>

        {/* Top-Down View */}
        <Box flex={1} position="relative" ref={topdownBoxRef} onClick={handleTopdownClick} cursor={onSelectPlayer ? 'crosshair' : 'default'}>
          <Box position="absolute" top={2} left={2} bg="blackAlpha.700" color="white" px={2} py={1} borderRadius="md" zIndex={1}>
            <Text fontSize="sm">Top-Down View</Text>
          </Box>
          <Image
            src={`data:image/png;base64,${displayFrame.topdown_image}`}
            alt={`Frame ${currentFrame} - Top Down`}
            objectFit="contain"
            w="100%"
            h="100%"
            onError={(e) => {
              console.error('Error loading topdown image');
              const target = e.target as HTMLImageElement;
              target.alt = 'Failed to load top-down view';
            }}
          />
          {/* SVG overlay for path and selection aligned to the contained image area */}
          {overlay && innerBox && (
            <svg
              ref={overlayBoxRef}
              viewBox={`0 0 ${overlay.w} ${overlay.h}`}
              preserveAspectRatio="none"
              style={{ position: 'absolute', left: innerBox.x, top: innerBox.y, width: innerBox.w, height: innerBox.h, pointerEvents: 'none' }}
            >
              {/* Draw path */}
              {path && path.length > 1 && (
                <polyline
                  points={path.map(p => `${p.x},${p.y}`).join(' ')}
                  fill="none"
                  stroke="#F59E0B"
                  strokeWidth={8}
                  strokeOpacity={0.7}
                />
              )}
              {/* Selected player marker at current frame */}
              {typeof selectedPlayerId === 'number' && displayFrame.players.map(p => (Number((p as any).id) === selectedPlayerId && typeof p.x === 'number' && typeof p.y === 'number') ? (
                <circle key={p.id} cx={p.x} cy={p.y} r={18} fill="none" stroke="#10B981" strokeWidth={6} />
              ) : null)}
            </svg>
          )}
        </Box>
      </Flex>

      {/* Frame Info Bar */}
      <Box bg="gray.800" color="white" px={4} py={2} fontSize="sm">
        <Flex justify="space-between" align="center">
          <Text>Frame: {currentFrame + 1} / {displayFrame.total_frames}</Text>
          <Text>{displayFrame.players.length} players detected</Text>
          <Box>
            {isPlaying ? (
              <Text as="span" color="green.400">Playing</Text>
            ) : (
              <Text as="span" color="yellow.400">Paused</Text>
            )}
          </Box>
        </Flex>
      </Box>
    </Box>
  );
};
