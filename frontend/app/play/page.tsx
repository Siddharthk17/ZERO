"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import type { Move, PieceSymbol, Square } from "chess.js";
import { Chess } from "chess.js";
import { Flag, Handshake, RefreshCw, RotateCcw } from "lucide-react";
import { Button } from "@/components/Button";
import { CapturedPieces } from "@/components/CapturedPieces";
import { ZeroChessBoard, kingSquare, promotionColor } from "@/components/ChessBoard";
import { Clock } from "@/components/Clock";
import { GameEndModal } from "@/components/GameEndModal";
import { MoveHistory } from "@/components/MoveHistory";
import { PromotionDialog } from "@/components/PromotionDialog";
import { StatusBanner } from "@/components/StatusBanner";
import { gameResultLabel, isPromotionMove, legalTargets } from "@/lib/chess";
import { getEngineSocket } from "@/lib/engine";

type PendingPromotion = { from: Square; to: Square; color: "w" | "b" };

export default function PlayPage() {
  const gameRef = useRef(new Chess());
  const engine = useMemo(() => getEngineSocket(), []);
  const [fen, setFen] = useState(gameRef.current.fen());
  const [history, setHistory] = useState<Move[]>([]);
  const [selected, setSelected] = useState<Square | null>(null);
  const [thinking, setThinking] = useState(false);
  const [online, setOnline] = useState(false);
  const [orientation, setOrientation] = useState<"white" | "black">("white");
  const [flashSquare, setFlashSquare] = useState<Square | null>(null);
  const [result, setResult] = useState<string | null>(null);
  const [pendingPromotion, setPendingPromotion] = useState<PendingPromotion | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [whiteClock, setWhiteClock] = useState(600);
  const [blackClock, setBlackClock] = useState(600);
  const game = gameRef.current;
  const legalMoves = selected ? legalTargets(game, selected) : [];
  const lastMove = history.at(-1) ?? null;

  useEffect(() => {
    engine.connect();
    return engine.subscribe(setOnline);
  }, [engine]);

  useEffect(() => {
    if (result) return;
    const timer = window.setInterval(() => {
      if (game.turn() === "w") setWhiteClock((value) => Math.max(0, value - 1));
      else setBlackClock((value) => Math.max(0, value - 1));
    }, 1000);
    return () => window.clearInterval(timer);
  }, [fen, result, game]);

  function syncGame() {
    setFen(game.fen());
    setHistory(game.history({ verbose: true }) as Move[]);
    setSelected(null);
    const ended = game.isGameOver();
    if (ended) setResult(gameResultLabel(game));
  }

  function fail(square: Square) {
    setFlashSquare(square);
    window.setTimeout(() => setFlashSquare(null), 220);
  }

  function maybeEngineMove() {
    if (!online || game.isGameOver()) return;
    setThinking(true);
    engine
      .requestBestMove({ fen: game.fen(), move_time: 1000 })
      .then((response) => {
        if (response.move && response.move !== "0000") {
          game.move({ from: response.move.slice(0, 2) as Square, to: response.move.slice(2, 4) as Square, promotion: response.move[4] as PieceSymbol });
          syncGame();
        }
      })
      .catch(() => setOnline(false))
      .finally(() => setThinking(false));
  }

  function movePiece(from: Square, to: Square, promotion?: PieceSymbol) {
    if (online && (thinking || game.turn() !== "w")) return false;
    const move = game.move({ from, to, promotion: promotion ?? "q" });
    if (!move) {
      fail(to);
      return false;
    }
    syncGame();
    if (game.turn() === "b") maybeEngineMove();
    return true;
  }

  function beginMove(from: Square, to: Square) {
    if (isPromotionMove(game, from, to)) {
      setPendingPromotion({ from, to, color: promotionColor(game, from) });
      return false;
    }
    return movePiece(from, to);
  }

  function onSquareClick(square: Square) {
    if (online && (thinking || game.turn() !== "w")) return;
    if (selected) {
      if (legalMoves.some((move) => move.to === square)) {
        beginMove(selected, square);
        return;
      }
      setSelected(null);
    }
    const piece = game.get(square);
    if (piece && (!online || piece.color === game.turn())) {
      setSelected(square);
    }
  }

  function choosePromotion(piece: PieceSymbol) {
    if (!pendingPromotion) return;
    movePiece(pendingPromotion.from, pendingPromotion.to, piece);
    setPendingPromotion(null);
  }

  function rematch() {
    gameRef.current = new Chess();
    setFen(gameRef.current.fen());
    setHistory([]);
    setSelected(null);
    setResult(null);
    setWhiteClock(600);
    setBlackClock(600);
  }

  return (
    <main className="min-h-screen bg-zero-bg px-3 py-4 text-zinc-100 md:px-8">
      <StatusBanner online={online} />
      <div className="mx-auto grid max-w-7xl gap-4 md:grid-cols-[minmax(340px,720px)_360px]">
        <section className="space-y-3">
          <Clock name="ZERO" seconds={blackClock} active={game.turn() === "b"} />
          <ZeroChessBoard
            game={game}
            fen={fen}
            orientation={orientation}
            selectedSquare={selected}
            legalMoves={legalMoves}
            lastMove={lastMove}
            checkSquare={kingSquare(game)}
            flashSquare={flashSquare}
            disabled={false}
            onPieceDrop={beginMove}
            onSquareClick={onSquareClick}
          />
          <Clock name="You" seconds={whiteClock} active={game.turn() === "w"} />
        </section>
        <aside className="grid gap-3 md:max-h-[calc(100vh-2rem)] md:grid-rows-[auto_auto_1fr_auto]">
          <div className="rounded-md bg-zero-panel px-4 py-3">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-sm font-semibold text-zinc-100">ZERO engine</div>
                <div className="text-xs text-zinc-400">{thinking ? "calculating" : online ? "connected" : "offline"}</div>
              </div>
              <div className={`h-3 w-3 rounded-full ${thinking ? "thinking-dot bg-zero-accent" : online ? "bg-zero-accent" : "bg-zinc-600"}`} />
            </div>
          </div>
          <CapturedPieces history={history} />
          <MoveHistory history={history} />
          <div className="grid grid-cols-2 gap-2 rounded-md bg-zero-panel p-3">
            <Button icon={<RefreshCw size={17} />} onClick={() => setOrientation((value) => (value === "white" ? "black" : "white"))}>
              Flip
            </Button>
            <Button icon={<Handshake size={17} />} onClick={() => { setNotice("ZERO declined draw offer"); window.setTimeout(() => setNotice(null), 1800); }}>
              Draw
            </Button>
            <Button variant="danger" icon={<Flag size={17} />} onClick={() => setResult("Resignation")}>
              Resign
            </Button>
            <Button variant="primary" icon={<RotateCcw size={17} />} onClick={rematch}>
              Rematch
            </Button>
            {notice && <div className="col-span-2 rounded-md bg-[#332f29] px-3 py-2 text-sm text-zinc-300">{notice}</div>}
          </div>
        </aside>
      </div>
      {pendingPromotion && <PromotionDialog color={pendingPromotion.color} onChoose={choosePromotion} />}
      <GameEndModal result={result} onRematch={rematch} />
    </main>
  );
}
