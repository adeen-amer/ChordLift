/** Backend API base URL. Empty string = same origin (Vite proxy or nginx). */
export const API_BASE = (import.meta.env.VITE_API_BASE ?? '').replace(/\/$/, '');

export const UPLOAD_ONLY = import.meta.env.VITE_UPLOAD_ONLY === 'true';

export function apiUrl(path: string): string {
  return `${API_BASE}${path.startsWith('/') ? path : `/${path}`}`;
}
