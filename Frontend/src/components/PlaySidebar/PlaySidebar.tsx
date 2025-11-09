import React, { useEffect, useState, useRef } from 'react';
import { Box, VStack, Input, Spinner, Text } from '@chakra-ui/react';
import type { Play } from '../../api/client';
import { api } from '../../api/client';
import { PlayListItem } from './PlayListItem';

export const PlaySidebar: React.FC<{ onSelectPlay: (playId: string) => void }> = ({ onSelectPlay }) => {
  const [plays, setPlays] = useState<Play[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState('');
  const [error, setError] = useState<string | null>(null);
  const prefetchedRef = useRef<Set<string>>(new Set());

  useEffect(() => {
    const fetchPlays = async () => {
      try {
        setLoading(true);
        setError(null);
        const data = await api.getPlays();
        setPlays(data);
      } catch (error) {
        console.error('Error fetching plays:', error);
        setError('Failed to load plays. Please try again later.');
      } finally {
        setLoading(false);
      }
    };

    fetchPlays();
  }, []);

  const filteredPlays = plays.filter(play =>
    play.name.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const prefetchPlay = async (playId: string) => {
    if (prefetchedRef.current.has(playId)) return;
    try {
      const status = await api.getPlayStatus(playId);
      if (status.status !== 'ready') return;
      const fd = await api.getFrame(playId, 0);
      // Preload images into the browser cache
      await new Promise<void>((resolve, reject) => {
        let loaded = 0;
        const done = () => { loaded += 1; if (loaded === 2) resolve(); };
        const fail = (e: any) => reject(e);
        const img1 = new Image();
        img1.onload = done;
        img1.onerror = fail;
        img1.src = `data:image/jpeg;base64,${fd.original_image}`;
        const img2 = new Image();
        img2.onload = done;
        img2.onerror = fail;
        img2.src = `data:image/png;base64,${fd.topdown_image}`;
      });
      prefetchedRef.current.add(playId);
    } catch {}
  };

  if (loading) {
    return (
      <Box p={4}>
        <Spinner />
      </Box>
    );
  }

  return (
    <Box w="100%" h="100%" overflowY="auto" p={4} bg="gray.50" borderLeftWidth="1px" color="#1A202C">
      <VStack spacing={4} align="stretch">
        <Box>
          <Input
            placeholder="Search plays..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            mb={4}
          />
          <Text fontSize="lg" fontWeight="bold" mb={4} color="#1A202C">
            Available Plays ({filteredPlays.length})
          </Text>
          {error && (
            <Text color="#C53030" fontSize="sm">{error}</Text>
          )}
        </Box>

        {filteredPlays.length === 0 ? (
          <Text color="#2D3748">No plays found. {searchTerm && 'Try a different search term.'}</Text>
        ) : (
          <VStack spacing={3} align="stretch">
            {filteredPlays.map((play) => (
              <div key={play.id} onMouseEnter={() => prefetchPlay(play.id)}>
                <PlayListItem play={play} onSelect={onSelectPlay} />
              </div>
            ))}
          </VStack>
        )}
      </VStack>
    </Box>
  );
};
