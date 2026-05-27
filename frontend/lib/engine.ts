export type EngineRequest = {
  fen: string;
  move_time: number;
};

export type EngineResponse = {
  move: string;
  evaluation: number;
  nodes: number;
};

type Pending = {
  resolve: (value: EngineResponse) => void;
  reject: (reason?: unknown) => void;
};

export class EngineSocket {
  private socket: WebSocket | null = null;
  private pending: Pending | null = null;
  private listeners = new Set<(online: boolean) => void>();
  private online = false;

  connect() {
    if (this.socket && this.socket.readyState <= WebSocket.OPEN) return;
    this.socket = new WebSocket("ws://localhost:8765");
    this.socket.onopen = () => this.setOnline(true);
    this.socket.onclose = () => {
      this.setOnline(false);
      this.pending?.reject(new Error("Engine offline"));
      this.pending = null;
    };
    this.socket.onerror = () => {
      this.setOnline(false);
      this.pending?.reject(new Error("Engine offline"));
      this.pending = null;
    };
    this.socket.onmessage = (event) => {
      const pending = this.pending;
      this.pending = null;
      if (!pending) return;
      try {
        pending.resolve(JSON.parse(event.data) as EngineResponse);
      } catch (error) {
        pending.reject(error);
      }
    };
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
    this.connect();
    return new Promise<EngineResponse>((resolve, reject) => {
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
    });
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
