import React from 'react';
import { Box, Text } from '@chakra-ui/react';

interface PlaybackControlsProps {
  isPlaying: boolean;
  currentFrame: number;
  totalFrames: number;
  onPlayPause: () => void;
  onFrameChange: (frame: number) => void;
  onStepForward: () => void;
  onStepBackward: () => void;
}

export const PlaybackControls: React.FC<PlaybackControlsProps> = ({
  isPlaying,
  currentFrame,
  totalFrames,
  onPlayPause,
  onFrameChange,
  onStepForward,
  onStepBackward,
}) => {
  const progress = totalFrames > 0 ? (currentFrame / (totalFrames - 1)) * 100 : 0;
  
  return (
    <Box 
      style={{
        padding: '1rem',
        backgroundColor: 'white',
        border: '1px solid #E2E8F0',
        borderRadius: '0.375rem',
        boxShadow: '0 1px 2px 0 rgba(0, 0, 0, 0.05)'
      }}
    >
      <div style={{
        display: 'flex',
        flexDirection: 'column',
        gap: '0.5rem'
      }}>
        {/* Playback Buttons */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          gap: '1rem',
          marginBottom: '0.5rem'
        }}>
          <button 
            onClick={onStepBackward}
            style={{
              background: 'none',
              border: 'none',
              cursor: 'pointer',
              padding: '0.25rem',
              borderRadius: '0.25rem',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              width: '2rem',
              height: '2rem',
              color: '#2D3748' // gray.700 to ensure currentColor is visible
            }}
            title="Step backward"
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <polygon points="19 20 9 12 19 4 19 20" />
              <line x1="5" y1="4" x2="5" y2="20" />
            </svg>
          </button>
          
          <button 
            onClick={onPlayPause}
            style={{
              background: '#3182ce',
              color: 'white',
              border: 'none',
              borderRadius: '50%',
              width: '3rem',
              height: '3rem',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              cursor: 'pointer'
            }}
            title={isPlaying ? 'Pause' : 'Play'}
          >
            {isPlaying ? (
              // Pause icon: solid bars, larger size
              <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor" xmlns="http://www.w3.org/2000/svg">
                <rect x="6" y="4" width="4" height="16" rx="1" />
                <rect x="14" y="4" width="4" height="16" rx="1" />
              </svg>
            ) : (
              // Play icon: solid triangle, larger size
              <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor" xmlns="http://www.w3.org/2000/svg">
                <polygon points="5,3 20,12 5,21" />
              </svg>
            )}
          </button>
          
          <button 
            onClick={onStepForward}
            style={{
              background: 'none',
              border: 'none',
              cursor: 'pointer',
              padding: '0.25rem',
              borderRadius: '0.25rem',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              width: '2rem',
              height: '2rem',
              color: '#2D3748' // gray.700 to ensure currentColor is visible
            }}
            title="Step forward"
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <polygon points="5 4 15 12 5 20 5 4" />
              <line x1="19" y1="4" x2="19" y2="20" />
            </svg>
          </button>
          
          <Text 
            style={{
              fontSize: '0.875rem',
              color: '#4A5568',
              minWidth: '100px',
              textAlign: 'center'
            }}
          >
            {formatTimeFromFrame(currentFrame)} / {formatTimeFromFrame(totalFrames)}
          </Text>
        </div>
        
        {/* Slider */}
        <div style={{
          position: 'relative',
          width: '100%',
          height: '4px',
          backgroundColor: '#E2E8F0',
          borderRadius: '2px',
          margin: '0.5rem 0'
        }}>
          <div 
            style={{
              position: 'absolute',
              left: 0,
              top: 0,
              height: '100%',
              width: `${progress}%`,
              backgroundColor: '#3182ce',
              borderRadius: '2px'
            }}
          />
          <input
            type="range"
            min={0}
            max={totalFrames - 1}
            value={currentFrame}
            onChange={(e) => onFrameChange(Number(e.target.value))}
            style={{
              position: 'absolute',
              width: '100%',
              height: '100%',
              opacity: 0,
              cursor: 'pointer',
              zIndex: 2
            }}
            aria-label="Frame slider"
          />
          <div 
            style={{
              position: 'absolute',
              left: `calc(${progress}% - 8px)`,
              top: '50%',
              transform: 'translateY(-50%)',
              width: '16px',
              height: '16px',
              backgroundColor: '#3182ce',
              borderRadius: '50%',
              boxShadow: '0 1px 2px 0 rgba(0, 0, 0, 0.1)'
            }}
          />
        </div>
        
        {/* Frame Info */}
        <div style={{
          display: 'flex',
          justifyContent: 'space-between',
          marginTop: '0.25rem',
          padding: '0 0.25rem'
        }}>
          <Text style={{ fontSize: '0.75rem', color: '#718096' }}>
            Frame: {currentFrame + 1}
          </Text>
          <Text style={{ fontSize: '0.75rem', color: '#718096' }}>
            Total: {totalFrames}
          </Text>
        </div>
      </div>
    </Box>
  );
};

// Helper function to format frame number into time (MM:SS)
function formatTimeFromFrame(frame: number, fps: number = 30): string {
  const totalSeconds = Math.floor(frame / fps);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
}
