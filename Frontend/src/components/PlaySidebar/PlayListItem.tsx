import React from 'react';
import { Box, Image, Text, Badge } from '@chakra-ui/react';
import type { Play } from '../../api/client';
import { api } from '../../api/client';

interface PlayListItemProps {
  play: Play;
  onSelect: (playId: string) => void;
}

export const PlayListItem: React.FC<PlayListItemProps> = ({ play, onSelect }) => {
  const bgHover = 'gray.100';
  
  // Format duration (assuming duration is in seconds)
  const formatDuration = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <Box
      p={3}
      borderRadius="md"
      borderWidth="1px"
      _hover={{
        bg: bgHover,
        cursor: 'pointer',
        transform: 'translateY(-2px)',
        boxShadow: 'md',
      }}
      transition="all 0.2s"
      onClick={() => onSelect(play.id)}
    >
      <div style={{ display: 'flex', gap: '16px', alignItems: 'flex-start' }}>
        <Box position="relative" w="100px" h="60px" borderRadius="md" overflow="hidden" flexShrink={0}>
          <Image
            src={api.getPlayThumbnailUrl(play.id)}
            alt={`Thumbnail for ${play.name}`}
            objectFit="cover"
            w="100%"
            h="100%"
          />
        </Box>
        
        <div style={{ display: 'flex', flexDirection: 'column', gap: '4px', flex: 1 }}>
          <Text fontWeight="medium" color="#1A202C" style={{ 
            display: '-webkit-box',
            WebkitLineClamp: 1,
            WebkitBoxOrient: 'vertical',
            overflow: 'hidden',
            textOverflow: 'ellipsis'
          }}>
            {play.name}
          </Text>
          <div style={{ display: 'flex', gap: '8px', fontSize: '0.875rem', color: '#4A5568' }}>
            <Text color="#4A5568">{formatDuration(play.duration)}</Text>
            <Badge colorScheme="blue" size="sm">
              {play.duration > 60 ? 'Long Play' : 'Short Play'}
            </Badge>
          </div>
        </div>
      </div>
    </Box>
  );
};
