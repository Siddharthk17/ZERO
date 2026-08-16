export type EngineRequest = {
  fen: string;
  move_time: number;
};

export type EngineResponse = {
  move: string;
  evaluation: number;
  nodes: number;
  error?: string;
};

type Pending = {
  resolve: (value: EngineResponse) => void;
  reject: (reason?: unknown) => void;
};

export class EngineSocket {
  private socket: WebSocket | null = null;
  private pending: Pending | null = null;
  private connectPromise: Promise<void> | null = null;
  private listeners = new Set<(online: boolean) => void>();
  private online = false;

  connect(): Promise<void> {
    if (this.socket?.readyState === WebSocket.OPEN) return Promise.resolve();
    if (this.socket?.readyState === WebSocket.CONNECTING && this.connectPromise) return this.connectPromise;
    const url = process.env.NEXT_PUBLIC_ZERO_WS_URL ?? "ws://localhost:8765";
    this.socket = new WebSocket(url);
    this.connectPromise = new Promise<void>((resolve, reject) => {
      const socket = this.socket;
      if (!socket) {
        reject(new Error("Engine socket unavailable"));
        return;
      }
      socket.onopen = () => {
        this.setOnline(true);
        this.connectPromise = null;
        resolve();
      };
      socket.onclose = () => {
        this.setOnline(false);
        this.connectPromise = null;
        this.pending?.reject(new Error("Engine offline"));
        this.pending = null;
        reject(new Error("Engine offline"));
      };
      socket.onerror = () => {
        this.setOnline(false);
        this.connectPromise = null;
        this.pending?.reject(new Error("Engine offline"));
        this.pending = null;
        reject(new Error("Engine offline"));
      };
      socket.onmessage = (event) => {
        const pending = this.pending;
        this.pending = null;
        if (!pending) return;
        try {
          const response = JSON.parse(event.data) as EngineResponse;
          if (response.error) {
            pending.reject(new Error(response.error));
          } else {
            pending.resolve(response);
          }
        } catch (error) {
          pending.reject(error);
        }
      };
    });
    return this.connectPromise;
  }

  isOnline() {
    return this.online;
  }

  subscribe(listener: (online: boolean) => void) {
    this.listeners.add(listener);
    listener(this.online);
    return () => {
      this.listeners.delete(listener);
    };
  }

  requestBestMove(payload: EngineRequest) {
    return this.connect().then(
      () =>
        new Promise<EngineResponse>((resolve, reject) => {
          if (!this.socket || this.socket.readyState !== WebSocket.OPEN) {
            reject(new Error("Engine offline"));
            return;
          }
          if (this.pending) {
            reject(new Error("Engine busy"));
            return;
          }
          this.pending = { resolve, reject };
          this.socket.send(JSON.stringify(payload));
        }),
    );
  }

  private setOnline(value: boolean) {
    this.online = value;
    this.listeners.forEach((listener) => listener(value));
  }
}

let sharedEngine: EngineSocket | null = null;

export function getEngineSocket() {
  if (!sharedEngine) sharedEngine = new EngineSocket();
  return sharedEngine;
}
