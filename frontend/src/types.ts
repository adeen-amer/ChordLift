export interface LyricLine {
  time: number;
  end_time?: number;
  text: string;
}

export interface ChordEvent {
  time: number;
  end_time: number;
  chord: string;
  confidence: number;
  is_low_confidence: boolean;
  strumming: string;
  is_power?: boolean;
  lyrics?: string;
}

export interface SoloNote {
  time: number;
  end: number;
  pitch: number;
  note: string;
  string: number;
  fret: number;
}

export interface SoloSection {
  start: number;
  end: number;
  type: string;
  confidence: string;
  notes: SoloNote[];
}

export interface SongInfo {
  title: string;
  artist?: string;
  album_art_url?: string | null;
}

export interface KeyInfo {
  root: string;
  mode: 'major' | 'minor' | 'dorian' | 'mixolydian';
  display: string;
}

export interface AnalysisData {
  video_id: string;
  timeline: ChordEvent[];
  solos: SoloSection[];
  song?: SongInfo;
  key?: KeyInfo;
  analyzer_version?: string;
  chord_engine?: string;
  lyrics?: {
    source?: string | null;
    synced?: boolean;
    lyrics_version?: string;
    lines?: LyricLine[];
    plain_text?: string;
  };
}
