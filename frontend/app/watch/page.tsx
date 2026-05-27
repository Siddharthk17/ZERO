"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import type { Move, PieceSymbol, Square } from "chess.js";
import { Chess } from "chess.js";
import { RotateCcw } from "lucide-react";
import { Button } from "@/components/Button";
import { CapturedPieces } from "@/components/CapturedPieces";
import { ZeroChessBoard, kingSquare } from "@/components/ChessBoard";
import { Clock } from "@/components/Clock";
import { EvaluationBar } from "@/components/EvaluationBar";
import { MoveHistory } from "@/components/MoveHistory";
import { StatusBanner } from "@/components/StatusBanner";
import { evalToWhitePercent, gameResultLabel } from "@/lib/chess";
import { getEngineSocket } from "@/lib/engine";

const speeds = [1, 2, 5, 10];

export default function WatchPage() {
  const gameRef = useRef(new Chess());
  const engine = useMemo(() => getEngineSocket(), []);
  const [fen, setFen] = useState(gameRef.current.fen());
  const [history, setHistory] = useState<Move[]>([]);
  const [online, setOnline] = useState(false);
  const [thinking, setThinking] = useState(false);
  const [speed, setSpeed] = useState(2);
  const [evaluation, setEvaluation] = useState(0);
  const [record, setRecord] = useState({ white: 0, black: 0, draws: 0 });
  const [gameKey, setGameKey] = useState(0);
  const game = gameRef.current;
  const lastMove = history.at(-1) ?? null;

  useEffect(() => {
    engine.connect();
    return engine.subscribe(setOnline);
  }, [engine]);

  useEffect(() => {
    if (!online || thinking) return;
    if (game.isGameOver()) {
      const label = gameResultLabel(game);
      setRecord((value) => ({
        white: value.white + (game.isCheckmate() && game.turn() === "b" ? 1 : 0),
        black: value.black + (game.isCheckmate() && game.turn() === "w" ? 1 : 0),
        draws: value.draws + (label !== "Checkmate" ? 1 : 0)
      }));
      const reset = window.setTimeout(newGame, 1400 / speed);
      return () => window.clearTimeout(reset);
    }

    const timer = window.setTimeout(() => {
      setThinking(true);
      engine
        .requestBestMove({ fen: game.fen(), move_time: Math.max(100, Math.floor(1000 / speed)) })
        .then((response) => {
          if (response.move && response.move !== "0000") {
            game.move({
              from: response.move.slice(0, 2) as Square,
              to: response.move.slice(2, 4) as Square,
              promotion: (response.move[4] as PieceSymbol) || "q"
            });
            setEvaluation(response.evaluation);
            setFen(game.fen());
            setHistory(game.history({ verbose: true }) as Move[]);
          }
        })
        .catch(() => setOnline(false))
        .finally(() => setThinking(false));
    }, 550 / speed);
    return () => window.clearTimeout(timer);
  }, [fen, online, thinking, speed, gameKey, engine, game]);

  function newGame() {
    gameRef.current = new Chess();
    setFen(gameRef.current.fen());
    setHistory([]);
    setEvaluation(0);
    setGameKey((value) => value + 1);
  }

  return (
    <main className="min-h-screen bg-zero-bg px-3 py-4 text-zinc-100 md:px-8">
      <StatusBanner online={online} />
      <div className="mx-auto mb-4 flex max-w-7xl items-center justify-between rounded-md bg-zero-panel px-4 py-3">
        <div className="text-sm font-semibold text-zinc-200">
          ZERO-W {record.white} <span className="mx-2 text-zinc-500">/</span> ZERO-B {record.black}
          <span className="mx-2 text-zinc-500">/</span> Draws {record.draws}
        </div>
        <div className="flex items-center gap-2">
          {speeds.map((value) => (
            <button
              key={value}
              onClick={() => setSpeed(value)}
              className={`rounded-md px-3 py-2 text-sm font-semibold ${speed === value ? "bg-zero-accent text-white" : "bg-zero-panel2 text-zinc-300"}`}
            >
              {value}x
            </button>
          ))}
        </div>
      </div>
      <div className="mx-auto grid max-w-7xl gap-4 md:grid-cols-[40px_minmax(340px,720px)_360px]">
        <div className="hidden md:block">
          <EvaluationBar whitePercent={evalToWhitePercent(evaluation)} />
        </div>
        <section className="space-y-3">
          <Clock name="ZERO Black" seconds={600} active={game.turn() === "b"} />
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
          <Clock name="ZERO White" seconds={600} active={game.turn() === "w"} />
        </section>
        <aside className="grid gap-3 md:max-h-[calc(100vh-6rem)] md:grid-rows-[auto_auto_1fr_auto]">
          <div className="rounded-md bg-zero-panel px-4 py-3">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-sm font-semibold text-zinc-100">Live evaluation</div>
                <div className="text-xs text-zinc-400">{evaluation.toFixed(2)}</div>
              </div>
              <div className={`h-3 w-3 rounded-full ${thinking ? "thinking-dot bg-zero-accent" : online ? "bg-zero-accent" : "bg-zinc-600"}`} />
            </div>
            <div className="mt-4 block md:hidden">
              <div className="h-4 overflow-hidden rounded-sm border border-zero-border bg-zinc-900">
                <div className="h-full bg-zinc-100" style={{ width: `${evalToWhitePercent(evaluation)}%` }} />
              </div>
            </div>
          </div>
          <CapturedPieces history={history} />
          <MoveHistory history={history} />
          <div className="rounded-md bg-zero-panel p-3">
            <Button variant="primary" icon={<RotateCcw size={17} />} onClick={newGame} className="w-full">
              New game
            </Button>
          </div>
        </aside>
      </div>
    </main>
  );
}
