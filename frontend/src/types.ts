export interface LyricLine {
  time: number;
  end_time?: number;
  text: string;
}

export type AudioLoadState = 'idle' | 'loading' | 'ready' | 'error';

export interface ChordEvent {
  time: number;
  end_time: number;
  chord: string;
  confidence: number;
  is_low_confidence: boolean;
  confidence_tier?: 'low' | 'medium' | 'high';
  strumming?: string;
  /** True when strumming is an onset-density estimate, not detected pattern. */
  strumming_is_heuristic?: boolean;
  is_power?: boolean;
  lyrics?: string;
  /** Original model chord when display was key-adjusted or user-corrected. */
  model_chord?: string;
  concert_chord?: string;
  display_adjusted?: boolean;
  user_corrected?: boolean;
}

export interface BeatGridInfo {
  tempo_bpm: number;
  beat_times: number[];
  downbeat_times: number[];
  beat_duration: number;
}

export interface CapoInfo {
  capo_fret: number;
  display: string | null;
  transpose_semitones: number;
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
  model_timeline?: ChordEvent[];
  beats?: BeatGridInfo;
  capo?: CapoInfo;
  presentation?: string;
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
