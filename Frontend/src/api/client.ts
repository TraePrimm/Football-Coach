import axios from 'axios';

const API_BASE_URL = 'http://localhost:5000/api';

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export interface Play {
  id: string;
  name: string;
  duration: number;
}

export interface Player {
  id: number;
  number: string;
  team: 'home' | 'away';
  role: string;
  x?: number;
  y?: number;
}

export interface FrameData {
  original_image: string;
  topdown_image: string;
  total_frames: number;
  players: Player[];
  map_width_px?: number;
  map_height_px?: number;
}

export interface PlayStatus {
  status: 'not_started' | 'processing' | 'ready' | 'error';
  progress?: number;
}

export const api = {
  // Play endpoints
  getPlays: () => apiClient.get<Play[]>('/plays').then(res => res.data),
  
  // Play management
  getPlayStatus: (playId: string) => 
    apiClient.get<PlayStatus>(`/play/${playId}/status`).then(res => res.data),
    
  processPlay: (playId: string) => 
    apiClient.post<{ status: string }>(`/play/${playId}/process`).then(res => res.data),
  
  // Frame data
  getFrame: (playId: string, frameNumber: number, signal?: AbortSignal) => 
    apiClient.get<FrameData>(`/play/${playId}/frame/${frameNumber}`, { signal }).then(res => res.data),
    
  // Player stats
  getPlayerStats: (playId: string, playerId: number) => 
    apiClient.get(`/play/${playId}/player/${playerId}`).then(res => res.data),
  getAllStats: (playId: string) =>
    apiClient.get(`/play/${playId}/all_stats`).then(res => res.data),
  getPlayerPath: (playId: string, playerId: number) =>
    apiClient.get<{ path: { frame: number; x: number; y: number }[] }>(`/play/${playId}/player/${playerId}/path`).then(res => res.data),
    
  // Thumbnail
  getPlayThumbnailUrl: (playId: string) => 
    `${API_BASE_URL}/play/${playId}/thumbnail`,
};
