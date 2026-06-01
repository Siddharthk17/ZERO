"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import type { Move, PieceSymbol, Square } from "chess.js";
import { Chess } from "chess.js";
import { Flag, Handshake, RefreshCw, Play, Volume2, VolumeX } from "lucide-react";
import { Button } from "@/components/Button";
import { ZeroChessBoard, kingSquare, promotionColor } from "@/components/ChessBoard";
import { GameEndModal } from "@/components/GameEndModal";
import { MoveHistory } from "@/components/MoveHistory";
import { PromotionDialog } from "@/components/PromotionDialog";
import { StatusBanner } from "@/components/StatusBanner";
import { gameResultLabel, isPromotionMove, legalTargets, capturedPieces, getMaterialDifference } from "@/lib/chess";
import { getEngineSocket } from "@/lib/engine";
import { chessAudio } from "@/lib/audio";

type PendingPromotion = { from: Square; to: Square; color: "w" | "b" };
const symbols: Record<"w" | "b", Record<string, string>> = {
  w: { p: "♙", n: "♘", b: "♗", r: "♖", q: "♕" },
  b: { p: "♟", n: "♞", b: "♝", r: "♜", q: "♛" }
};

export default function PlayPage() {
  const gameRef = useRef(new Chess());
  const engine = useMemo(() => getEngineSocket(), []);
  
  // Game state
  const [fen, setFen] = useState(gameRef.current.fen());
  const [history, setHistory] = useState<Move[]>([]);
  const [selected, setSelected] = useState<Square | null>(null);
  const [thinking, setThinking] = useState(false);
  const [online, setOnline] = useState(false);
  const [flashSquare, setFlashSquare] = useState<Square | null>(null);
  const [result, setResult] = useState<string | null>(null);
  const [pendingPromotion, setPendingPromotion] = useState<PendingPromotion | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  
  // Match settings
  const [gameStarted, setGameStarted] = useState(false);
  const [playerColor, setPlayerColor] = useState<"w" | "b">("w");
  const [timeControl, setTimeControl] = useState<number>(600); // 10 minutes default
  const [whiteClock, setWhiteClock] = useState(600);
  const [blackClock, setBlackClock] = useState(600);
  const [orientation, setOrientation] = useState<"white" | "black">("white");
  const [soundEnabled, setSoundEnabled] = useState(true);
  const [activeTab, setActiveTab] = useState<"setup" | "moves">("setup");

  const game = gameRef.current;
  const legalMoves = selected ? legalTargets(game, selected) : [];
  const lastMove = history.at(-1) ?? null;
  const captured = capturedPieces(history);
  const materialDiff = getMaterialDifference(history);

  // Connect to engine socket
  useEffect(() => {
    engine.connect();
    return engine.subscribe(setOnline);
  }, [engine]);

  // Handle countdown timer
  useEffect(() => {
    if (!gameStarted || result || timeControl === Infinity) return;
    const timer = window.setInterval(() => {
      if (game.turn() === "w") {
        setWhiteClock((value) => {
          if (value <= 1) {
            setResult("Time Out (Black wins)");
            if (soundEnabled) chessAudio.playGameOver();
            return 0;
          }
          return value - 1;
        });
      } else {
        setBlackClock((value) => {
          if (value <= 1) {
            setResult("Time Out (White wins)");
            if (soundEnabled) chessAudio.playGameOver();
            return 0;
          }
          return value - 1;
        });
      }
    }, 1000);
    return () => window.clearInterval(timer);
  }, [fen, result, gameStarted, soundEnabled, game, timeControl]);

  // Synchronize board UI state
  function syncGame() {
    setFen(game.fen());
    setHistory(game.history({ verbose: true }) as Move[]);
    setSelected(null);
    
    if (game.isGameOver()) {
      setResult(gameResultLabel(game));
      if (soundEnabled) chessAudio.playGameOver();
    }
  }

  // Flash invalid move feedback
  function fail(square: Square) {
    setFlashSquare(square);
    window.setTimeout(() => setFlashSquare(null), 220);
  }

  // Trigger engine calculation
  function maybeEngineMove() {
    if (!online || game.isGameOver()) return;
    setThinking(true);
    engine
      .requestBestMove({ fen: game.fen(), move_time: 1000 })
      .then((response) => {
        if (response.move && response.move !== "0000") {
          const from = response.move.slice(0, 2) as Square;
          const to = response.move.slice(2, 4) as Square;
          const promo = response.move[4] as PieceSymbol | undefined;
          
          const move = game.move({ from, to, promotion: promo ?? "q" });
          if (move) {
            if (soundEnabled) {
              if (move.captured) chessAudio.playCapture();
              else chessAudio.playMove();
              if (game.isCheck()) chessAudio.playCheck();
            }
            syncGame();
          }
        }
      })
      .catch(() => setOnline(false))
      .finally(() => setThinking(false));
  }

  // Move validation and routing
  function movePiece(from: Square, to: Square, promotion?: PieceSymbol) {
    if (!gameStarted || game.isGameOver()) return false;
    
    // Safety check: ensure the piece being moved belongs to the player
    const piece = game.get(from);
    if (!piece || piece.color !== playerColor) return false;
    
    if (online && (thinking || game.turn() !== playerColor)) return false;
    
    const move = game.move({ from, to, promotion: promotion ?? "q" });
    if (!move) {
      fail(to);
      return false;
    }
    
    if (soundEnabled) {
      if (move.captured) chessAudio.playCapture();
      else chessAudio.playMove();
      if (game.isCheck()) chessAudio.playCheck();
    }
    
    syncGame();
    
    // Switch turn to engine
    if (!game.isGameOver()) {
      window.setTimeout(maybeEngineMove, 300);
    }
    return true;
  }

  function beginMove(from: Square, to: Square) {
    // Safety check: ensure the piece being moved belongs to the player
    const piece = game.get(from);
    if (!piece || piece.color !== playerColor) return false;

    if (isPromotionMove(game, from, to)) {
      setPendingPromotion({ from, to, color: promotionColor(game, from) });
      return false;
    }
    return movePiece(from, to);
  }

  function onSquareClick(square: Square) {
    if (!gameStarted || game.isGameOver()) return;
    if (online && (thinking || game.turn() !== playerColor)) return;
    
    if (selected) {
      if (legalMoves.some((move) => move.to === square)) {
        beginMove(selected, square);
        return;
      }
      setSelected(null);
    }
    
    const piece = game.get(square);
    if (piece && piece.color === playerColor) {
      setSelected(square);
    }
  }

  function choosePromotion(piece: PieceSymbol) {
    if (!pendingPromotion) return;
    movePiece(pendingPromotion.from, pendingPromotion.to, piece);
    setPendingPromotion(null);
  }

  // Start new matchup
  function startGame(colorSelection: "w" | "b") {
    gameRef.current = new Chess();
    setFen(gameRef.current.fen());
    setHistory([]);
    setSelected(null);
    setResult(null);
    
    setPlayerColor(colorSelection);
    setOrientation(colorSelection === "w" ? "white" : "black");
    
    setWhiteClock(timeControl);
    setBlackClock(timeControl);
    setGameStarted(true);
    setActiveTab("moves");

    // If starting as Black, White engine moves first
    if (colorSelection === "b") {
      setThinking(true);
      window.setTimeout(maybeEngineMove, 500);
    }
  }

  function resign() {
    if (!gameStarted || result) return;
    setResult("Resignation (ZERO wins)");
    if (soundEnabled) chessAudio.playGameOver();
  }

  function rematch() {
    setGameStarted(false);
    setActiveTab("setup");
    setResult(null);
  }

  // Format second clock representation
  function formatClockTime(seconds: number) {
    if (seconds === Infinity) return "∞";
    const min = Math.floor(seconds / 60);
    const sec = seconds % 60;
    return `${min}:${sec.toString().padStart(2, "0")}`;
  }

  // Helper values to structure Opponent / Player panels
  const opponentName = playerColor === "w" ? "ZERO (Engine)" : "You";
  const opponentRating = playerColor === "w" ? "2450" : "1500";
  const opponentClock = playerColor === "w" ? blackClock : whiteClock;
  const opponentCaptured = playerColor === "w" ? captured.b : captured.w;
  const opponentAdvantage = playerColor === "w" ? materialDiff.blackAdvantage : materialDiff.whiteAdvantage;
  const opponentClockActive = gameStarted && !result && (playerColor === "w" ? game.turn() === "b" : game.turn() === "w");

  const playerName = playerColor === "w" ? "You" : "ZERO (Engine)";
  const playerRating = playerColor === "w" ? "1500" : "2450";
  const playerClock = playerColor === "w" ? whiteClock : blackClock;
  const playerCaptured = playerColor === "w" ? captured.w : captured.b;
  const playerAdvantage = playerColor === "w" ? materialDiff.whiteAdvantage : materialDiff.blackAdvantage;
  const playerClockActive = gameStarted && !result && (playerColor === "w" ? game.turn() === "w" : game.turn() === "b");

  return (
    <main className="min-h-screen bg-zero-bg px-3 py-4 text-zinc-100 md:px-8">
      <StatusBanner online={online} />
      
      <div className="mx-auto grid max-w-7xl gap-6 md:grid-cols-[minmax(340px,720px)_360px]">
        {/* LEFT COLUMN: Chess Board & Integrated Player Panels */}
        <section className="flex flex-col gap-2">
          {/* OPPONENT PANEL */}
          <div className="flex items-center justify-between bg-zero-panel px-4 py-2.5 rounded-t-md border-b border-zero-border shadow-sm">
            <div className="flex items-center gap-3">
              <div className="h-8 w-8 rounded bg-zinc-700 flex items-center justify-center font-bold text-xs uppercase tracking-wide text-zinc-300">
                {opponentName.slice(0, 2)}
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <span className="text-sm font-bold text-zinc-200">{opponentName}</span>
                  <span className="text-[10px] font-semibold text-zinc-400 bg-zero-panel2 px-1 rounded">
                    {opponentRating}
                  </span>
                  {thinking && playerColor === "w" && (
                    <span className="text-[10px] text-zero-accent font-semibold animate-pulse">thinking...</span>
                  )}
                </div>
                {/* Captured Pieces by Opponent */}
                <div className="flex items-center gap-1.5 mt-0.5">
                  <div className="flex text-lg leading-none tracking-tight select-none">
                    {opponentCaptured.map((p, idx) => (
                      <span key={`opp-cap-${p}-${idx}`} className={playerColor === "w" ? "text-zinc-500" : "text-zinc-300"}>
                        {symbols[playerColor === "w" ? "b" : "w"][p]}
                      </span>
                    ))}
                  </div>
                  {opponentAdvantage > 0 && (
                    <span className="text-[9px] font-bold text-zero-accent bg-[#354521] px-1 rounded">
                      +{opponentAdvantage}
                    </span>
                  )}
                </div>
              </div>
            </div>
            {/* Clock display */}
            <div className={`font-mono text-lg font-bold px-3 py-1.5 rounded transition-all shadow-inner ${
              opponentClockActive
                ? opponentClock < 30 ? "bg-red-950 text-red-300 animate-pulse border border-red-800" : "bg-white text-zinc-950 font-extrabold"
                : "bg-zinc-900 text-zinc-400"
            }`}>
              {formatClockTime(opponentClock)}
            </div>
          </div>

          {/* MAIN INTERACTIVE CHESS BOARD */}
          <div className="bg-zero-panel p-1 rounded-sm shadow-2xl relative">
            <ZeroChessBoard
              game={game}
              fen={fen}
              orientation={orientation}
              selectedSquare={selected}
              legalMoves={legalMoves}
              lastMove={lastMove}
              checkSquare={kingSquare(game)}
              flashSquare={flashSquare}
              disabled={!gameStarted || result !== null}
              onPieceDrop={beginMove}
              onSquareClick={onSquareClick}
            />
          </div>

          {/* PLAYER PANEL */}
          <div className="flex items-center justify-between bg-zero-panel px-4 py-2.5 rounded-b-md border-t border-zero-border shadow-sm">
            <div className="flex items-center gap-3">
              <div className="h-8 w-8 rounded bg-zero-accent flex items-center justify-center font-bold text-xs uppercase tracking-wide text-white">
                {playerName.slice(0, 2)}
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <span className="text-sm font-bold text-zinc-200">{playerName}</span>
                  <span className="text-[10px] font-semibold text-zinc-400 bg-zero-panel2 px-1 rounded">
                    {playerRating}
                  </span>
                  {thinking && playerColor === "b" && (
                    <span className="text-[10px] text-zero-accent font-semibold animate-pulse">thinking...</span>
                  )}
                </div>
                {/* Captured Pieces by Player */}
                <div className="flex items-center gap-1.5 mt-0.5">
                  <div className="flex text-lg leading-none tracking-tight select-none">
                    {playerCaptured.map((p, idx) => (
                      <span key={`ply-cap-${p}-${idx}`} className={playerColor === "w" ? "text-zinc-300" : "text-zinc-500"}>
                        {symbols[playerColor === "w" ? "w" : "b"][p]}
                      </span>
                    ))}
                  </div>
                  {playerAdvantage > 0 && (
                    <span className="text-[9px] font-bold text-zero-accent bg-[#354521] px-1 rounded">
                      +{playerAdvantage}
                    </span>
                  )}
                </div>
              </div>
            </div>
            {/* Clock display */}
            <div className={`font-mono text-lg font-bold px-3 py-1.5 rounded transition-all shadow-inner ${
              playerClockActive
                ? playerClock < 30 ? "bg-red-950 text-red-300 animate-pulse border border-red-800" : "bg-white text-zinc-950 font-extrabold"
                : "bg-zinc-900 text-zinc-400"
            }`}>
              {formatClockTime(playerClock)}
            </div>
          </div>
        </section>

        {/* RIGHT COLUMN: Sidebar Panel Controls */}
        <aside className="flex flex-col gap-3 max-h-[calc(100vh-2rem)]">
          {/* Header Tab Layout */}
          <div className="grid grid-cols-2 bg-zero-panel p-1 rounded-md border border-zero-border">
            <button
              onClick={() => setActiveTab("setup")}
              className={`py-2 rounded font-semibold text-sm transition-all ${
                activeTab === "setup" ? "bg-zero-panel2 text-white shadow-sm" : "text-zinc-400 hover:text-white"
              }`}
            >
              Play Arena
            </button>
            <button
              onClick={() => setActiveTab("moves")}
              className={`py-2 rounded font-semibold text-sm transition-all ${
                activeTab === "moves" ? "bg-zero-panel2 text-white shadow-sm" : "text-zinc-400 hover:text-white"
              }`}
            >
              Game Log
            </button>
          </div>

          {/* TAB CONTENT: Game Setup */}
          {activeTab === "setup" && (
            <div className="flex flex-col gap-4 bg-zero-panel p-4 rounded-md border border-zero-border flex-1">
              <div>
                <h3 className="text-sm font-bold text-zinc-300 mb-2">1. Choose Match Timer</h3>
                <div className="grid grid-cols-5 gap-1.5">
                  {[
                    { label: "1 min", sec: 60 },
                    { label: "3 min", sec: 180 },
                    { label: "5 min", sec: 300 },
                    { label: "10 min", sec: 600 },
                    { label: "∞", sec: Infinity }
                  ].map((tc) => (
                    <button
                      key={tc.label}
                      disabled={gameStarted}
                      onClick={() => {
                        setTimeControl(tc.sec);
                        setWhiteClock(tc.sec);
                        setBlackClock(tc.sec);
                      }}
                      className={`py-2 text-xs font-bold rounded transition-all border ${
                        timeControl === tc.sec
                          ? "bg-zero-accent/15 border-zero-accent text-white"
                          : "bg-zero-panel2 border-transparent text-zinc-400 hover:text-white"
                      }`}
                    >
                      {tc.label}
                    </button>
                  ))}
                </div>
              </div>

              <div>
                <h3 className="text-sm font-bold text-zinc-300 mb-2">2. Select Side</h3>
                <div className="grid grid-cols-2 gap-2">
                  <button
                    disabled={gameStarted}
                    onClick={() => startGame("w")}
                    className="flex flex-col items-center justify-center p-4 rounded-md border border-zero-border bg-white text-zinc-950 font-bold hover:bg-zinc-100 transition-all shadow-md group disabled:opacity-50"
                  >
                    <span className="text-3xl leading-none mb-1 text-zinc-800">♔</span>
                    <span className="text-xs tracking-wider uppercase">Play as White</span>
                  </button>
                  <button
                    disabled={gameStarted}
                    onClick={() => startGame("b")}
                    className="flex flex-col items-center justify-center p-4 rounded-md border border-zero-border bg-zinc-900 text-zinc-100 font-bold hover:bg-zinc-800 transition-all shadow-md group disabled:opacity-50"
                  >
                    <span className="text-3xl leading-none mb-1 text-zinc-400">♚</span>
                    <span className="text-xs tracking-wider uppercase">Play as Black</span>
                  </button>
                </div>
              </div>

              <div className="flex-1 flex flex-col justify-end pt-4">
                {!gameStarted ? (
                  <Button
                    variant="primary"
                    icon={<Play size={18} />}
                    onClick={() => startGame("w")}
                    className="w-full text-base font-extrabold uppercase py-4 bg-zero-accent hover:bg-[#95c55d] text-white rounded-md shadow-lg"
                  >
                    Quick Play (White)
                  </Button>
                ) : (
                  <div className="text-center py-5 border border-[#3c3935] rounded bg-[#211f1c]">
                    <div className="text-xs text-zinc-500 uppercase tracking-widest mb-1">Match In Progress</div>
                    <div className="text-sm font-bold text-zinc-300">You are playing as {playerColor === "w" ? "White" : "Black"}</div>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* TAB CONTENT: Game Log */}
          {activeTab === "moves" && (
            <div className="flex flex-col gap-3 bg-zero-panel p-3 rounded-md border border-zero-border flex-1 min-h-0">
              <MoveHistory history={history} />
              
              <div className="flex items-center justify-between border-t border-[#3c3935] pt-3">
                <button
                  onClick={() => setSoundEnabled(!soundEnabled)}
                  className="flex items-center gap-1.5 text-xs font-semibold text-zinc-400 hover:text-white"
                >
                  {soundEnabled ? <Volume2 size={15} /> : <VolumeX size={15} />}
                  <span>Sounds: {soundEnabled ? "On" : "Off"}</span>
                </button>
                <div className="text-xs text-zinc-500 font-mono">
                  Moves: {history.length}
                </div>
              </div>
            </div>
          )}

          {/* In-Game Action Buttons Panel */}
          <div className="grid grid-cols-2 gap-2 bg-zero-panel p-3 rounded-md border border-zero-border">
            <Button
              icon={<RefreshCw size={15} />}
              onClick={() => setOrientation((val) => (val === "white" ? "black" : "white"))}
            >
              Flip Board
            </Button>
            <Button
              icon={<Handshake size={15} />}
              disabled={!gameStarted || result !== null}
              onClick={() => {
                setNotice("ZERO declined draw offer");
                window.setTimeout(() => setNotice(null), 1800);
              }}
            >
              Draw Offer
            </Button>
            <Button
              variant={!gameStarted || result !== null ? "secondary" : "danger"}
              icon={<Flag size={15} />}
              disabled={!gameStarted || result !== null}
              onClick={resign}
            >
              Resign
            </Button>
            <Button
              variant="primary"
              disabled={!gameStarted}
              onClick={rematch}
            >
              New Match
            </Button>
            {notice && (
              <div className="col-span-2 text-center text-xs font-semibold text-amber-300 bg-[#352a12] py-2 rounded animate-pulse border border-[#52411b]">
                {notice}
              </div>
            )}
          </div>
        </aside>
      </div>

      {pendingPromotion && (
        <PromotionDialog color={pendingPromotion.color} onChoose={choosePromotion} />
      )}
      <GameEndModal result={result} onRematch={rematch} />
    </main>
  );
}
