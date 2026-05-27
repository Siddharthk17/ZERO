import type { Chess, Move, Square } from "chess.js";

export type Captured = Record<"w" | "b", string[]>;

const pieceValues: Record<string, number> = {
  p: 1,
  n: 3,
  b: 3,
  r: 5,
  q: 9,
  k: 99
};

export function capturedPieces(history: Move[]): Captured {
  const captured: Captured = { w: [], b: [] };
  history.forEach((move) => {
    if (move.captured) {
      captured[move.color === "w" ? "w" : "b"].push(move.captured);
    }
  });
  captured.w.sort((a, b) => pieceValues[a] - pieceValues[b]);
  captured.b.sort((a, b) => pieceValues[a] - pieceValues[b]);
  return captured;
}

export function legalTargets(game: Chess, square: Square) {
  return game.moves({ square, verbose: true }) as Move[];
}

export function isPromotionMove(game: Chess, source: Square, target: Square) {
  const piece = game.get(source);
  if (!piece || piece.type !== "p") return false;
  const rank = target[1];
  return (piece.color === "w" && rank === "8") || (piece.color === "b" && rank === "1");
}

export function gameResultLabel(game: Chess) {
  if (game.isCheckmate()) return "Checkmate";
  if (game.isStalemate()) return "Stalemate";
  if (game.isDraw()) return "Draw";
  return "Game over";
}

export function formatClock(seconds: number) {
  const safe = Math.max(0, Math.floor(seconds));
  const minutes = Math.floor(safe / 60);
  const rest = safe % 60;
  return `${minutes}:${rest.toString().padStart(2, "0")}`;
}

export function evalToWhitePercent(evaluation: number) {
  const clamped = Math.max(-8, Math.min(8, evaluation));
  return Math.round((1 / (1 + Math.exp(-clamped / 2))) * 100);
}
