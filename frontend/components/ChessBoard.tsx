"use client";

import type { CSSProperties } from "react";
import { Chessboard } from "react-chessboard";
import type { BoardOrientation } from "react-chessboard/dist/chessboard/types";
import type { Move, PieceSymbol, Square } from "chess.js";
import { Chess } from "chess.js";

type Props = {
  game: Chess;
  fen: string;
  orientation: BoardOrientation;
  selectedSquare: Square | null;
  legalMoves: Move[];
  lastMove: Move | null;
  checkSquare: Square | null;
  flashSquare: Square | null;
  disabled?: boolean;
  onPieceDrop: (source: Square, target: Square) => boolean;
  onSquareClick: (square: Square) => void;
};

const files = ["a", "b", "c", "d", "e", "f", "g", "h"];
const pieces = ["wP", "wN", "wB", "wR", "wQ", "wK", "bP", "bN", "bB", "bR", "bQ", "bK"] as const;
const pieceAssetVersion = "cburnett-real-1";

export function ZeroChessBoard({
  game,
  fen,
  orientation,
  selectedSquare,
  legalMoves,
  lastMove,
  checkSquare,
  flashSquare,
  disabled,
  onPieceDrop,
  onSquareClick
}: Props) {
  const styles = squareStyles(game, selectedSquare, legalMoves, lastMove, checkSquare, flashSquare);
  const customPieces = Object.fromEntries(
    pieces.map((piece) => [
      piece,
      ({ squareWidth }: { squareWidth: number }) => (
        <img
          className="piece-img"
          src={`/pieces/cburnett/${piece}.svg?v=${pieceAssetVersion}`}
          alt={piece}
          style={{ width: squareWidth, height: squareWidth }}
        />
      )
    ])
  );

  return (
    <div className={disabled ? "pointer-events-none opacity-95" : ""}>
      <Chessboard
        id="zero-board"
        position={fen}
        boardOrientation={orientation}
        arePiecesDraggable={!disabled}
        onPieceDrop={(source, target) => onPieceDrop(source as Square, target as Square)}
        onSquareClick={(square) => onSquareClick(square as Square)}
        customPieces={customPieces}
        customBoardStyle={{
          borderRadius: "3px",
          boxShadow: "0 16px 42px rgba(0,0,0,0.35)"
        }}
        customLightSquareStyle={{ backgroundColor: "#EEEED2" }}
        customDarkSquareStyle={{ backgroundColor: "#769656" }}
        customSquareStyles={styles}
      />
    </div>
  );
}

function squareStyles(
  game: Chess,
  selected: Square | null,
  legalMoves: Move[],
  lastMove: Move | null,
  checkSquare: Square | null,
  flashSquare: Square | null
) {
  const styles: Record<string, CSSProperties> = {};
  if (lastMove) {
    styles[lastMove.from] = { backgroundColor: "rgba(247, 247, 105, 0.62)" };
    styles[lastMove.to] = { backgroundColor: "rgba(247, 247, 105, 0.62)" };
  }
  if (selected) {
    styles[selected] = { ...(styles[selected] ?? {}), backgroundColor: "rgba(255, 255, 0, 0.45)" };
  }
  legalMoves.forEach((move) => {
    const target = move.to;
    const piece = game.get(target as Square);
    styles[target] = piece
      ? {
          ...(styles[target] ?? {}),
          boxShadow: "inset 0 0 0 6px rgba(30, 30, 30, 0.28)",
          borderRadius: "50%"
        }
      : {
          ...(styles[target] ?? {}),
          backgroundImage: "radial-gradient(circle, rgba(30,30,30,0.28) 18%, transparent 19%)"
        };
  });
  if (checkSquare) {
    styles[checkSquare] = {
      ...(styles[checkSquare] ?? {}),
      background: "radial-gradient(circle, rgba(187,62,62,0.92) 0%, rgba(187,62,62,0.65) 45%, rgba(187,62,62,0.15) 80%)"
    };
  }
  if (flashSquare) {
    styles[flashSquare] = {
      ...(styles[flashSquare] ?? {}),
      backgroundColor: "#bb3e3e"
    };
  }
  return styles;
}

export function kingSquare(game: Chess): Square | null {
  if (!game.isCheck()) return null;
  const turn = game.turn();
  for (const file of files) {
    for (let rank = 1; rank <= 8; rank += 1) {
      const square = `${file}${rank}` as Square;
      const piece = game.get(square);
      if (piece?.type === "k" && piece.color === turn) return square;
    }
  }
  return null;
}

export function promotionColor(game: Chess, source: Square): "w" | "b" {
  return game.get(source)?.color ?? "w";
}
