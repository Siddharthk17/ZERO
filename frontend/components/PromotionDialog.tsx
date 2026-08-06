import type { PieceSymbol } from "chess.js";
import Image from "next/image";

const pieces: PieceSymbol[] = ["q", "r", "b", "n"];
const pieceAssetVersion = "cburnett-real-1";
const labels: Record<PieceSymbol, string> = {
  p: "Pawn",
  n: "Knight",
  b: "Bishop",
  r: "Rook",
  q: "Queen",
  k: "King"
};

export function PromotionDialog({
  color,
  onChoose
}: {
  color: "w" | "b";
  onChoose: (piece: PieceSymbol) => void;
}) {
  return (
    <div className="fixed inset-0 z-40 flex items-center justify-center bg-black/60 px-4">
      <div className="grid grid-cols-4 gap-2 rounded-md border border-zero-border bg-zero-panel p-3 shadow-2xl">
        {pieces.map((piece) => (
          <button
            key={piece}
            className="flex h-20 w-20 items-center justify-center rounded-md bg-zero-panel2 hover:bg-[#403c37]"
            onClick={() => onChoose(piece)}
            aria-label={labels[piece]}
          >
            <Image
              className="piece-img"
              src={`/pieces/cburnett/${color}${piece.toUpperCase()}.svg?v=${pieceAssetVersion}`}
              alt={labels[piece]}
              width={64}
              height={64}
              unoptimized
            />
          </button>
        ))}
      </div>
    </div>
  );
}
