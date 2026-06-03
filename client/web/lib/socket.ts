import { io, Socket } from "socket.io-client";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:3001";

let socket: Socket | null = null;

export function getSocket(): Socket {
  if (!socket) {
    socket = io(API, {
      transports: ["websocket", "polling"],
      reconnectionAttempts: 10,
      reconnectionDelay: 2000,
    });

    socket.on("connect", () => {});
    socket.on("disconnect", () => {});
    socket.on("connect_error", () => {});
  }

  return socket;
}

export function destroySocket() {
  socket?.disconnect();
  socket = null;
}
