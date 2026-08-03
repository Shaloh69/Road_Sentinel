import { io, Socket } from "socket.io-client";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:3001";

export const ADMIN_TOKEN_KEY = "rs_admin_token";

let adminSocket: Socket | null = null;
let adminSocketToken: string | null = null;

/**
 * Connects to the authenticated `/admin` Socket.IO namespace using the given
 * JWT (obtained from POST /api/auth/login). Reconnecting with a different
 * token tears down the old connection first.
 */
export function getAdminSocket(token: string): Socket {
  if (adminSocket && adminSocketToken === token) {
    return adminSocket;
  }

  if (adminSocket) {
    adminSocket.disconnect();
    adminSocket = null;
  }

  adminSocketToken = token;
  adminSocket = io(`${API}/admin`, {
    transports: ["websocket", "polling"],
    reconnectionAttempts: 10,
    reconnectionDelay: 2000,
    auth: { token },
  });

  return adminSocket;
}

export function destroyAdminSocket() {
  adminSocket?.disconnect();
  adminSocket = null;
  adminSocketToken = null;
}

export function getStoredAdminToken(): string | null {
  if (typeof window === "undefined") return null;

  return sessionStorage.getItem(ADMIN_TOKEN_KEY);
}

export function storeAdminToken(token: string): void {
  sessionStorage.setItem(ADMIN_TOKEN_KEY, token);
}

export function clearStoredAdminToken(): void {
  sessionStorage.removeItem(ADMIN_TOKEN_KEY);
  destroyAdminSocket();
}
