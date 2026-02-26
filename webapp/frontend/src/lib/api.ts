const FALLBACK_API_PORT = 8000;

function trimTrailingSlash(value: string): string {
  return value.replace(/\/+$/, "");
}

export function getApiBase(): string {
  const fromEnv = process.env.NEXT_PUBLIC_API_BASE_URL?.trim();
  if (fromEnv) {
    return trimTrailingSlash(fromEnv);
  }

  if (typeof window !== "undefined") {
    return `${window.location.protocol}//${window.location.hostname}:${FALLBACK_API_PORT}`;
  }

  return `http://localhost:${FALLBACK_API_PORT}`;
}

export function apiUrl(path: string): string {
  const normalizedPath = path.startsWith("/") ? path : `/${path}`;
  return `${getApiBase()}${normalizedPath}`;
}
