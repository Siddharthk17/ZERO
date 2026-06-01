"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import type { Move, PieceSymbol, Square } from "chess.js";
import { Chess } from "chess.js";
import { RotateCcw, Volume2, VolumeX } from "lucide-react";
import { Button } from "@/components/Button";
import { ZeroChessBoard, kingSquare } from "@/components/ChessBoard";
import { EvaluationBar } from "@/components/EvaluationBar";
import { MoveHistory } from "@/components/MoveHistory";
import { StatusBanner } from "@/components/StatusBanner";
import { evalToWhitePercent, gameResultLabel, capturedPieces, getMaterialDifference } from "@/lib/chess";
import { getEngineSocket } from "@/lib/engine";
import { chessAudio } from "@/lib/audio";

const speeds = [1, 2, 5, 10];
const symbols: Record<"w" | "b", Record<string, string>> = {
  w: { p: "♙", n: "♘", b: "♗", r: "♖", q: "♕" },
  b: { p: "♟", n: "♞", b: "♝", r: "♜", q: "♛" }
};

export default function WatchPage() {
  const gameRef = useRef(new Chess());
  const engine = useMemo(() => getEngineSocket(), []);
  
  // State
  const [fen, setFen] = useState(gameRef.current.fen());
  const [history, setHistory] = useState<Move[]>([]);
  const [online, setOnline] = useState(false);
  const [thinking, setThinking] = useState(false);
  const [speed, setSpeed] = useState(2);
  const [evaluation, setEvaluation] = useState(0);
  const [record, setRecord] = useState({ white: 0, black: 0, draws: 0 });
  const [gameKey, setGameKey] = useState(0);
  const [soundEnabled, setSoundEnabled] = useState(true);

  const game = gameRef.current;
  const lastMove = history.at(-1) ?? null;
  const captured = capturedPieces(history);
  const materialDiff = getMaterialDifference(history);

  // Connect socket
  useEffect(() => {
    engine.connect();
    return engine.subscribe(setOnline);
  }, [engine]);

  // Main self-play driver loop
  useEffect(() => {
    if (!online || thinking) return;
    
    if (game.isGameOver()) {
      const label = gameResultLabel(game);
      setRecord((value) => ({
        white: value.white + (game.isCheckmate() && game.turn() === "b" ? 1 : 0),
        black: value.black + (game.isCheckmate() && game.turn() === "w" ? 1 : 0),
        draws: value.draws + (label !== "Checkmate" ? 1 : 0)
      }));
      if (soundEnabled) chessAudio.playGameOver();
      const reset = window.setTimeout(newGame, 1400 / speed);
      return () => window.clearTimeout(reset);
    }

    const timer = window.setTimeout(() => {
      setThinking(true);
      engine
        .requestBestMove({ fen: game.fen(), move_time: Math.max(100, Math.floor(1000 / speed)) })
        .then((response) => {
          if (response.move && response.move !== "0000") {
            const from = response.move.slice(0, 2) as Square;
            const to = response.move.slice(2, 4) as Square;
            const promo = (response.move[4] as PieceSymbol) || "q";
            
            const move = game.move({ from, to, promotion: promo });
            if (move) {
              if (soundEnabled) {
                if (move.captured) chessAudio.playCapture();
                else chessAudio.playMove();
                if (game.isCheck()) chessAudio.playCheck();
              }
              setEvaluation(response.evaluation);
              setFen(game.fen());
              setHistory(game.history({ verbose: true }) as Move[]);
            }
          }
        })
        .catch(() => setOnline(false))
        .finally(() => setThinking(false));
    }, 550 / speed);
    return () => window.clearTimeout(timer);
  }, [fen, online, thinking, speed, gameKey, engine, soundEnabled, game]);

  function newGame() {
    gameRef.current = new Chess();
    setFen(gameRef.current.fen());
    setHistory([]);
    setEvaluation(0);
    setGameKey((value) => value + 1);
  }

  // Format evaluation score nicely
  const formattedEval = evaluation.toFixed(2);
  const whitePercent = evalToWhitePercent(evaluation);

  return (
    <main className="min-h-screen bg-zero-bg px-3 py-4 text-zinc-100 md:px-8">
      <StatusBanner online={online} />

      {/* Arena scoreboard header */}
      <div className="mx-auto mb-4 flex max-w-7xl items-center justify-between rounded-md bg-zero-panel px-4 py-3 border border-zero-border shadow-md">
        <div className="text-sm font-bold text-zinc-200">
          ZERO White: <span className="text-zero-accent">{record.white}</span>
          <span className="mx-3 text-zinc-600">|</span>
          ZERO Black: <span className="text-zero-accent">{record.black}</span>
          <span className="mx-3 text-zinc-600">|</span>
          Draws: <span className="text-zinc-400">{record.draws}</span>
        </div>
        <div className="flex items-center gap-1.5">
          {speeds.map((value) => (
            <button
              key={value}
              onClick={() => setSpeed(value)}
              className={`rounded px-2.5 py-1.5 text-xs font-bold transition-all ${
                speed === value ? "bg-zero-accent text-white font-extrabold" : "bg-zero-panel2 text-zinc-400 hover:text-white"
              }`}
            >
              {value}x
            </button>
          ))}
        </div>
      </div>

      <div className="mx-auto grid max-w-7xl gap-4 md:grid-cols-[40px_minmax(340px,720px)_360px]">
        {/* EVALUATION BAR (LEFT) */}
        <div className="hidden md:block">
          <EvaluationBar whitePercent={whitePercent} />
        </div>

        {/* CHESS BOARD & PLAYER LABELS */}
        <section className="space-y-2">
          {/* OPPONENT (BLACK) PANEL */}
          <div className="flex items-center justify-between bg-zero-panel px-4 py-2.5 rounded-t-md border-b border-zero-border shadow-sm">
            <div className="flex items-center gap-3">
              <div className="h-8 w-8 rounded bg-zinc-700 flex items-center justify-center font-bold text-xs uppercase tracking-wide text-zinc-300">
                ZB
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <span className="text-sm font-bold text-zinc-200">ZERO Black</span>
                  <span className="text-[10px] font-semibold text-zinc-400 bg-zero-panel2 px-1 rounded">2450</span>
                  {thinking && game.turn() === "b" && (
                    <span className="text-[10px] text-zero-accent font-semibold animate-pulse">thinking...</span>
                  )}
                </div>
                <div className="flex items-center gap-1.5 mt-0.5">
                  <div className="flex text-lg leading-none tracking-tight select-none">
                    {captured.b.map((p, idx) => (
                      <span key={`zb-cap-${p}-${idx}`} className="text-zinc-500">
                        {symbols.b[p]}
                      </span>
                    ))}
                  </div>
                  {materialDiff.blackAdvantage > 0 && (
                    <span className="text-[9px] font-bold text-zero-accent bg-[#354521] px-1 rounded">
                      +{materialDiff.blackAdvantage}
                    </span>
                  )}
                </div>
              </div>
            </div>
            {/* Active turn highlight */}
            <div className={`h-2.5 w-2.5 rounded-full ${game.turn() === "b" && !game.isGameOver() ? "bg-zero-accent animate-ping" : "bg-zinc-700"}`} />
          </div>

          {/* BOARD CONTAINER */}
          <div className="bg-zero-panel p-1 rounded-sm shadow-2xl relative">
            <ZeroChessBoard
              game={game}
              fen={fen}
              orientation="white"
              selectedSquare={null}
              legalMoves={[]}
              lastMove={lastMove}
              checkSquare={kingSquare(game)}
              flashSquare={null}
              disabled
              onPieceDrop={() => false}
              onSquareClick={() => undefined}
            />
          </div>

          {/* PLAYER (WHITE) PANEL */}
          <div className="flex items-center justify-between bg-zero-panel px-4 py-2.5 rounded-b-md border-t border-zero-border shadow-sm">
            <div className="flex items-center gap-3">
              <div className="h-8 w-8 rounded bg-zero-accent flex items-center justify-center font-bold text-xs uppercase tracking-wide text-white">
                ZW
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <span className="text-sm font-bold text-zinc-200">ZERO White</span>
                  <span className="text-[10px] font-semibold text-zinc-400 bg-zero-panel2 px-1 rounded">2450</span>
                  {thinking && game.turn() === "w" && (
                    <span className="text-[10px] text-zero-accent font-semibold animate-pulse">thinking...</span>
                  )}
                </div>
                <div className="flex items-center gap-1.5 mt-0.5">
                  <div className="flex text-lg leading-none tracking-tight select-none">
                    {captured.w.map((p, idx) => (
                      <span key={`zw-cap-${p}-${idx}`} className="text-zinc-300">
                        {symbols.w[p]}
                      </span>
                    ))}
                  </div>
                  {materialDiff.whiteAdvantage > 0 && (
                    <span className="text-[9px] font-bold text-zero-accent bg-[#354521] px-1 rounded">
                      +{materialDiff.whiteAdvantage}
                    </span>
                  )}
                </div>
              </div>
            </div>
            {/* Active turn highlight */}
            <div className={`h-2.5 w-2.5 rounded-full ${game.turn() === "w" && !game.isGameOver() ? "bg-zero-accent animate-ping" : "bg-zinc-700"}`} />
          </div>
        </section>

        {/* SIDEBAR PANEL */}
        <aside className="grid gap-3 md:max-h-[calc(100vh-6rem)] md:grid-rows-[auto_1fr_auto]">
          {/* Live evaluation metrics */}
          <div className="rounded-md bg-zero-panel px-4 py-3 border border-zero-border">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-xs font-bold text-zinc-400 uppercase tracking-widest">Self-Play Eval</div>
                <div className="text-lg font-mono font-bold text-white mt-0.5">
                  {evaluation > 0 ? `+${formattedEval}` : formattedEval}
                </div>
              </div>
              <div className={`h-3.5 w-3.5 rounded-full transition-all ${thinking ? "thinking-dot bg-zero-accent" : "bg-zinc-700"}`} />
            </div>
            
            {/* Responsive mobile eval bar */}
            <div className="mt-4 block md:hidden">
              <div className="h-4 overflow-hidden rounded-sm border border-zero-border bg-zinc-900">
                <div className="h-full bg-zinc-100 transition-all duration-300" style={{ width: `${whitePercent}%` }} />
              </div>
            </div>
          </div>

          {/* Move Log */}
          <MoveHistory history={history} />

          {/* Action buttons */}
          <div className="grid grid-cols-2 gap-2 rounded-md bg-zero-panel p-2 border border-zero-border shadow-sm">
            <button
              onClick={() => setSoundEnabled(!soundEnabled)}
              className="flex items-center justify-center gap-1.5 text-xs font-semibold py-1.5 px-2.5 rounded bg-zero-panel2 text-zinc-400 hover:text-white transition-all hover:bg-zinc-800"
            >
              {soundEnabled ? <Volume2 size={13} /> : <VolumeX size={13} />}
              <span>Sounds: {soundEnabled ? "On" : "Off"}</span>
            </button>
            <button
              onClick={newGame}
              className="flex items-center justify-center gap-1.5 text-xs font-semibold py-1.5 px-2.5 rounded bg-zero-accent text-white hover:bg-[#95c55d] transition-all shadow-sm"
            >
              <RotateCcw size={13} />
              <span>Restart</span>
            </button>
          </div>
        </aside>
      </div>
    </main>
  );
}
