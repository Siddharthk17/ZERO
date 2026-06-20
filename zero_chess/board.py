"""High-performance, memory-optimized deterministic chess rules engine."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable

from .constants import (
    BISHOP_DIRS,
    BLACK,
    BK,
    BQ,
    EMPTY,
    KING_DELTAS,
    KNIGHT_DELTAS,
    PROMOTION_TYPES,
    QUEEN_DIRS,
    ROOK_DIRS,
    STARTING_FEN,
    WHITE,
    WK,
    WQ,
    color_of,
    opposite,
    parse_square,
    piece_for,
    piece_type,
    square_color,
    square_name,
)
from .move import (
    CAPTURE,
    DOUBLE_PAWN,
    EN_PASSANT,
    KING_CASTLE,
    PROMOTION,
    QUEEN_CASTLE,
    Move,
)
from .zobrist import CASTLING_KEYS, EP_FILE_KEYS, PIECE_INDEX, PIECE_KEYS, TURN_KEY, mask64
from .targets import game_result_to_values

@dataclass(slots=True)
class _State:
    move: Move
    captured: str
    castling_rights: int
    ep_square: int | None
    halfmove_clock: int
    fullmove_number: int
    zobrist_hash: int

class Board:
    """Complete chess position with highly optimized legal move generation."""

    __slots__ = (
        "squares",
        "turn",
        "castling_rights",
        "ep_square",
        "halfmove_clock",
        "fullmove_number",
        "_stack",
        "zobrist_hash",
        "hash_history",
    )

    def __init__(
        self,
        squares: list[str] | str | None = None,
        turn: int = WHITE,
        castling_rights: int = WK | WQ | BK | BQ,
        ep_square: int | None = None,
        halfmove_clock: int = 0,
        fullmove_number: int = 1,
        history: list[int] | None = None,
    ) -> None:
        if isinstance(squares, str):
            other = self.from_fen(squares)
            self.squares = other.squares
            self.turn = other.turn
            self.castling_rights = other.castling_rights
            self.ep_square = other.ep_square
            self.halfmove_clock = other.halfmove_clock
            self.fullmove_number = other.fullmove_number
            self._stack = []
            self.zobrist_hash = other.zobrist_hash
            self.hash_history = other.hash_history
            return
        self.squares = squares[:] if squares is not None else self._parse_placement(STARTING_FEN.split()[0])
        self.turn = turn
        self.castling_rights = castling_rights
        self.ep_square = ep_square
        self.halfmove_clock = halfmove_clock
        self.fullmove_number = fullmove_number
        self._stack: list[_State] = []
        self.zobrist_hash = self.compute_zobrist()
        self.hash_history = history[:] if history is not None else [self.zobrist_hash]

    @classmethod
    def starting_position(cls) -> "Board":
        return cls.from_fen(STARTING_FEN)

    @classmethod
    def from_fen(cls, fen: str) -> "Board":
        fields = fen.strip().split()
        if len(fields) != 6:
            raise ValueError(f"FEN must have six fields: {fen!r}")
        placement, active, castling, ep, halfmove, fullmove = fields
        board = cls._parse_placement(placement)

        if active not in ("w", "b"):
            raise ValueError(f"invalid active color: {active!r}")
        turn = WHITE if active == "w" else BLACK

        rights = 0
        if castling != "-":
            for ch in castling:
                if ch == "K":
                    rights |= WK
                elif ch == "Q":
                    rights |= WQ
                elif ch == "k":
                    rights |= BK
                elif ch == "q":
                    rights |= BQ
                else:
                    raise ValueError(f"invalid castling right: {ch!r}")

        ep_square = None if ep == "-" else parse_square(ep)
        return cls(board, turn, rights, ep_square, int(halfmove), int(fullmove))

    @staticmethod
    def _parse_placement(placement: str) -> list[str]:
        board = [EMPTY] * 64
        ranks = placement.split("/")
        if len(ranks) != 8:
            raise ValueError(f"FEN placement must contain eight ranks: {placement!r}")
        for fen_rank, text in enumerate(ranks):
            rank = 7 - fen_rank
            file_ = 0
            for ch in text:
                if ch.isdigit():
                    file_ += int(ch)
                elif ch in "PNBRQKpnbrqk":
                    if file_ >= 8:
                        raise ValueError(f"too many squares in FEN rank: {text!r}")
                    board[rank * 8 + file_] = ch
                    file_ += 1
                else:
                    raise ValueError(f"invalid FEN piece: {ch!r}")
            if file_ != 8:
                raise ValueError(f"FEN rank does not contain eight files: {text!r}")
        return board

    def copy(self) -> "Board":
        """Return a deep copy of the board including move stack and hash history."""
        return Board(
            self.squares,
            self.turn,
            self.castling_rights,
            self.ep_square,
            self.halfmove_clock,
            self.fullmove_number,
            self.hash_history,
        )

    def fen(self) -> str:
        """Return the standard FEN string for the current position."""
        ranks: list[str] = []
        for rank in range(7, -1, -1):
            empty = 0
            text = []
            for file_ in range(8):
                piece = self.squares[rank * 8 + file_]
                if piece == EMPTY:
                    empty += 1
                else:
                    if empty:
                        text.append(str(empty))
                        empty = 0
                    text.append(piece)
            if empty:
                text.append(str(empty))
            ranks.append("".join(text))
        active = "w" if self.turn == WHITE else "b"
        castling = self.castling_fen()
        ep = "-" if self.ep_square is None else square_name(self.ep_square)
        return f"{'/'.join(ranks)} {active} {castling} {ep} {self.halfmove_clock} {self.fullmove_number}"

    def castling_fen(self) -> str:
        """Return the castling-rights portion of a FEN string (e.g. ``'KQkq'`` or ``'-'``)."""
        text = ""
        if self.castling_rights & WK:
            text += "K"
        if self.castling_rights & WQ:
            text += "Q"
        if self.castling_rights & BK:
            text += "k"
        if self.castling_rights & BQ:
            text += "q"
        return text or "-"

    def compute_zobrist(self) -> int:
        """Compute the full 64-bit Zobrist hash from scratch for the current position."""
        value = 0
        for sq, piece in enumerate(self.squares):
            if piece != EMPTY:
                value ^= PIECE_KEYS[PIECE_INDEX[piece]][sq]
        if self.turn == BLACK:
            value ^= TURN_KEY
        value ^= CASTLING_KEYS[self.castling_rights]
        if self.ep_square is not None:
            value ^= EP_FILE_KEYS[self.ep_square & 7]
        return mask64(value)

    def piece_bitboards(self) -> dict[str, int]:
        """Return a dict mapping each piece symbol to its occupancy bitboard (64-bit int)."""
        bitboards = {piece: 0 for piece in "PNBRQKpnbrqk"}
        for sq, piece in enumerate(self.squares):
            if piece != EMPTY:
                bitboards[piece] |= 1 << sq
        return bitboards

    def occupancy_bitboards(self) -> tuple[int, int, int]:
        """Return ``(white_occ, black_occ, total_occ)`` as 64-bit bitboard integers."""
        white = black = 0
        for sq, piece in enumerate(self.squares):
            if piece in "PNBRQK":
                white |= 1 << sq
            elif piece in "pnbrqk":
                black |= 1 << sq
        return white, black, white | black

    def king_square(self, color: int) -> int:
        """Return the 0-63 square index of the king for the given color.

        Raises ValueError if the position has no king of that color.
        """
        king = "K" if color == WHITE else "k"
        for sq, piece in enumerate(self.squares):
            if piece == king:
                return sq
        raise ValueError("position has no king")

    def is_check(self, color: int | None = None) -> bool:
        """Return True if the given color's king (default: side to move) is attacked."""
        color = self.turn if color is None else color
        return self.is_square_attacked(self.king_square(color), opposite(color))

    def is_square_attacked(self, sq: int, by_color: int) -> bool:
        """Return True if ``sq`` is attacked by any piece of ``by_color``."""
        file_ = sq & 7
        rank = sq >> 3
        squares = self.squares

        if by_color == WHITE:
            if rank > 0:
                if file_ > 0 and squares[sq - 9] == "P":
                    return True
                if file_ < 7 and squares[sq - 7] == "P":
                    return True
            knight, king, rook_like, bishop_like = "N", "K", {"R", "Q"}, {"B", "Q"}
        else:
            if rank < 7:
                if file_ > 0 and squares[sq + 7] == "p":
                    return True
                if file_ < 7 and squares[sq + 9] == "p":
                    return True
            knight, king, rook_like, bishop_like = "n", "k", {"r", "q"}, {"b", "q"}

        for df, dr in KNIGHT_DELTAS:
            f, r = file_ + df, rank + dr
            if 0 <= f < 8 and 0 <= r < 8 and squares[r * 8 + f] == knight:
                return True

        for df, dr in KING_DELTAS:
            f, r = file_ + df, rank + dr
            if 0 <= f < 8 and 0 <= r < 8 and squares[r * 8 + f] == king:
                return True

        if self._attacked_by_slider(file_, rank, ROOK_DIRS, rook_like):
            return True
        return self._attacked_by_slider(file_, rank, BISHOP_DIRS, bishop_like)

    def _attacked_by_slider(
        self, file_: int, rank: int, directions: Iterable[tuple[int, int]], attackers: set[str]
    ) -> bool:
        squares = self.squares
        for df, dr in directions:
            f, r = file_ + df, rank + dr
            while 0 <= f < 8 and 0 <= r < 8:
                piece = squares[r * 8 + f]
                if piece != EMPTY:
                    if piece in attackers:
                        return True
                    break
                f += df
                r += dr
        return False

    def legal_moves(self) -> list[Move]:
        """Return all fully legal moves (king-safety verified) for the side to move."""
        moves: list[Move] = []
        moving = self.turn
        for move in self.pseudo_legal_moves():
            self.push(move)
            legal = not self.is_check(moving)
            self.pop()
            if legal:
                moves.append(move)
        return moves

    def has_legal_moves(self) -> bool:
        """Return True as soon as a single legal move is found (early-exit optimization).

        This avoids generating the complete move list when the caller only needs to
        know whether the game is over, which is the hot path in MCTS terminal checks.
        """
        moving = self.turn
        for move in self.pseudo_legal_moves():
            self.push(move)
            legal = not self.is_check(moving)
            self.pop()
            if legal:
                return True
        return False

    def pseudo_legal_moves(self) -> list[Move]:
        """Return all pseudo-legal moves (may leave own king in check)."""
        moves: list[Move] = []
        turn = self.turn
        squares = self.squares
        for sq in range(64):
            piece = squares[sq]
            if piece == EMPTY:
                continue
            if (piece.isupper() and turn == BLACK) or (piece.islower() and turn == WHITE):
                continue
            kind = piece.upper()
            if kind == "P":
                self._pawn_moves(sq, moves)
            elif kind == "N":
                self._jump_moves(sq, KNIGHT_DELTAS, moves)
            elif kind == "B":
                self._slide_moves(sq, BISHOP_DIRS, moves)
            elif kind == "R":
                self._slide_moves(sq, ROOK_DIRS, moves)
            elif kind == "Q":
                self._slide_moves(sq, QUEEN_DIRS, moves)
            elif kind == "K":
                self._king_moves(sq, moves)
        return moves

    def _pawn_moves(self, sq: int, moves: list[Move]) -> None:
        color = self.turn
        file_ = sq & 7
        rank = sq >> 3
        step = 1 if color == WHITE else -1
        start_rank = 1 if color == WHITE else 6
        promotion_from_rank = 6 if color == WHITE else 1
        one_rank = rank + step

        if 0 <= one_rank < 8:
            to = one_rank * 8 + file_
            if self.squares[to] == EMPTY:
                if rank == promotion_from_rank:
                    for promo in PROMOTION_TYPES:
                        moves.append(Move(sq, to, promo, PROMOTION))
                else:
                    moves.append(Move(sq, to))
                    two_rank = rank + 2 * step
                    if rank == start_rank and 0 <= two_rank < 8:
                        two = two_rank * 8 + file_
                        if self.squares[two] == EMPTY:
                            moves.append(Move(sq, two, None, DOUBLE_PAWN))

        for df in (-1, 1):
            f, r = file_ + df, rank + step
            if not (0 <= f < 8 and 0 <= r < 8):
                continue
            to = r * 8 + f
            target = self.squares[to]
            is_ep = self.ep_square == to
            if target != EMPTY and color_of(target) == opposite(color):
                if rank == promotion_from_rank:
                    for promo in PROMOTION_TYPES:
                        moves.append(Move(sq, to, promo, CAPTURE | PROMOTION))
                else:
                    moves.append(Move(sq, to, None, CAPTURE))
            elif is_ep:
                captured_sq = to - 8 if color == WHITE else to + 8
                expected = "p" if color == WHITE else "P"
                if 0 <= captured_sq < 64 and self.squares[captured_sq] == expected:
                    moves.append(Move(sq, to, None, CAPTURE | EN_PASSANT))

    def _jump_moves(self, sq: int, deltas: Iterable[tuple[int, int]], moves: list[Move]) -> None:
        file_ = sq & 7
        rank = sq >> 3
        squares = self.squares
        turn = self.turn
        for df, dr in deltas:
            f, r = file_ + df, rank + dr
            if 0 <= f < 8 and 0 <= r < 8:
                to = r * 8 + f
                target = squares[to]
                if target == EMPTY:
                    moves.append(Move(sq, to))
                elif (target.isupper() and turn == BLACK) or (target.islower() and turn == WHITE):
                    moves.append(Move(sq, to, None, CAPTURE))

    def _slide_moves(self, sq: int, directions: Iterable[tuple[int, int]], moves: list[Move]) -> None:
        file_ = sq & 7
        rank = sq >> 3
        squares = self.squares
        turn = self.turn
        for df, dr in directions:
            f, r = file_ + df, rank + dr
            while 0 <= f < 8 and 0 <= r < 8:
                to = r * 8 + f
                target = squares[to]
                if target == EMPTY:
                    moves.append(Move(sq, to))
                else:
                    if (target.isupper() and turn == BLACK) or (target.islower() and turn == WHITE):
                        moves.append(Move(sq, to, None, CAPTURE))
                    break
                f += df
                r += dr

    def _king_moves(self, sq: int, moves: list[Move]) -> None:
        self._jump_moves(sq, KING_DELTAS, moves)
        if self.turn == WHITE and sq == 4:  # e1
            if self._can_castle(WK, ["f1", "g1"], ["e1", "f1", "g1"], "h1"):
                moves.append(Move(sq, 6, None, KING_CASTLE))
            if self._can_castle(WQ, ["d1", "c1", "b1"], ["e1", "d1", "c1"], "a1"):
                moves.append(Move(sq, 2, None, QUEEN_CASTLE))
        elif self.turn == BLACK and sq == 60:  # e8
            if self._can_castle(BK, ["f8", "g8"], ["e8", "f8", "g8"], "h8"):
                moves.append(Move(sq, 62, None, KING_CASTLE))
            if self._can_castle(BQ, ["d8", "c8", "b8"], ["e8", "d8", "c8"], "a8"):
                moves.append(Move(sq, 58, None, QUEEN_CASTLE))

    def _can_castle(self, right: int, empty_squares: list[str], safe_squares: list[str], rook_sq: str) -> bool:
        if not (self.castling_rights & right):
            return False
        rook = "R" if self.turn == WHITE else "r"
        if self.squares[parse_square(rook_sq)] != rook:
            return False
        if any(self.squares[parse_square(name)] != EMPTY for name in empty_squares):
            return False
        enemy = opposite(self.turn)
        return not any(self.is_square_attacked(parse_square(name), enemy) for name in safe_squares)

    def push_uci(self, text: str) -> Move:
        """Parse and play a UCI move string, returning the resolved Move object.

        Raises ValueError if the move is illegal in the current position.
        """
        raw = Move.from_uci(text)
        for move in self.legal_moves():
            if move.from_sq == raw.from_sq and move.to_sq == raw.to_sq and move.promotion == raw.promotion:
                self.push(move)
                return move
        raise ValueError(f"illegal move {text!r} in position {self.fen()}")

    def push(self, move: Move) -> None:
        """Apply a move to the board, updating state and pushing to the undo stack.

        Raises ValueError if the source square is empty.
        """
        piece = self.squares[move.from_sq]
        if piece == EMPTY:
            raise ValueError(f"cannot move from empty square: {move}")
        color = self.turn
        old_castling = self.castling_rights
        old_ep = self.ep_square
        captured_sq = move.to_sq
        captured = self.squares[move.to_sq]
        if move.flags & EN_PASSANT:
            captured_sq = move.to_sq - 8 if color == WHITE else move.to_sq + 8
            captured = self.squares[captured_sq]

        self._stack.append(
            _State(
                move,
                captured,
                self.castling_rights,
                self.ep_square,
                self.halfmove_clock,
                self.fullmove_number,
                self.zobrist_hash,
            )
        )

        new_hash = self.zobrist_hash
        new_hash ^= CASTLING_KEYS[old_castling]
        if old_ep is not None:
            new_hash ^= EP_FILE_KEYS[old_ep & 7]
        new_hash ^= PIECE_KEYS[PIECE_INDEX[piece]][move.from_sq]
        if captured != EMPTY:
            new_hash ^= PIECE_KEYS[PIECE_INDEX[captured]][captured_sq]

        self.squares[move.from_sq] = EMPTY
        if move.flags & EN_PASSANT:
            self.squares[captured_sq] = EMPTY

        placed = piece
        if move.promotion:
            placed = piece_for(color, move.promotion)
        self.squares[move.to_sq] = placed
        new_hash ^= PIECE_KEYS[PIECE_INDEX[placed]][move.to_sq]

        if move.flags & KING_CASTLE:
            if color == WHITE:
                self.squares[7] = EMPTY
                self.squares[5] = "R"
                new_hash ^= PIECE_KEYS[PIECE_INDEX["R"]][7]
                new_hash ^= PIECE_KEYS[PIECE_INDEX["R"]][5]
            else:
                self.squares[63] = EMPTY
                self.squares[61] = "r"
                new_hash ^= PIECE_KEYS[PIECE_INDEX["r"]][63]
                new_hash ^= PIECE_KEYS[PIECE_INDEX["r"]][61]
        elif move.flags & QUEEN_CASTLE:
            if color == WHITE:
                self.squares[0] = EMPTY
                self.squares[3] = "R"
                new_hash ^= PIECE_KEYS[PIECE_INDEX["R"]][0]
                new_hash ^= PIECE_KEYS[PIECE_INDEX["R"]][3]
            else:
                self.squares[56] = EMPTY
                self.squares[59] = "r"
                new_hash ^= PIECE_KEYS[PIECE_INDEX["r"]][56]
                new_hash ^= PIECE_KEYS[PIECE_INDEX["r"]][59]

        self._update_castling_rights(piece, move.from_sq, move.to_sq, captured, captured_sq)

        if move.flags & DOUBLE_PAWN:
            self.ep_square = (move.from_sq + move.to_sq) // 2
        else:
            self.ep_square = None
        new_hash ^= CASTLING_KEYS[self.castling_rights]
        if self.ep_square is not None:
            new_hash ^= EP_FILE_KEYS[self.ep_square & 7]

        if piece_type(piece) == "P" or captured != EMPTY:
            self.halfmove_clock = 0
        else:
            self.halfmove_clock += 1

        if color == BLACK:
            self.fullmove_number += 1
        self.turn = opposite(self.turn)
        new_hash ^= TURN_KEY
        self.zobrist_hash = mask64(new_hash)
        self.hash_history.append(self.zobrist_hash)

    def _update_castling_rights(
        self, piece: str, from_sq: int, to_sq: int, captured: str, captured_sq: int
    ) -> None:
        if piece == "K":
            self.castling_rights &= ~(WK | WQ)
        elif piece == "k":
            self.castling_rights &= ~(BK | BQ)
        elif piece == "R":
            if from_sq == 7:
                self.castling_rights &= ~WK
            elif from_sq == 0:
                self.castling_rights &= ~WQ
        elif piece == "r":
            if from_sq == 63:
                self.castling_rights &= ~BK
            elif from_sq == 56:
                self.castling_rights &= ~BQ

        if captured == "R":
            if captured_sq == 7:
                self.castling_rights &= ~WK
            elif captured_sq == 0:
                self.castling_rights &= ~WQ
        elif captured == "r":
            if captured_sq == 63:
                self.castling_rights &= ~BK
            elif captured_sq == 56:
                self.castling_rights &= ~BQ

    def pop(self) -> Move:
        """Undo the last move, restoring all board state. Raises IndexError if the stack is empty."""
        if not self._stack:
            raise IndexError("pop from empty move stack")
        state = self._stack.pop()
        move = state.move
        self.hash_history.pop()
        self.turn = opposite(self.turn)
        self.castling_rights = state.castling_rights
        self.ep_square = state.ep_square
        self.halfmove_clock = state.halfmove_clock
        self.fullmove_number = state.fullmove_number
        self.zobrist_hash = state.zobrist_hash

        piece = self.squares[move.to_sq]
        original = piece_for(self.turn, "P") if move.promotion else piece
        self.squares[move.from_sq] = original
        self.squares[move.to_sq] = state.captured

        if move.flags & EN_PASSANT:
            self.squares[move.to_sq] = EMPTY
            captured_sq = move.to_sq - 8 if self.turn == WHITE else move.to_sq + 8
            self.squares[captured_sq] = state.captured

        if move.flags & KING_CASTLE:
            if self.turn == WHITE:
                self.squares[7] = "R"
                self.squares[5] = EMPTY
            else:
                self.squares[63] = "r"
                self.squares[61] = EMPTY
        elif move.flags & QUEEN_CASTLE:
            if self.turn == WHITE:
                self.squares[0] = "R"
                self.squares[3] = EMPTY
            else:
                self.squares[56] = "r"
                self.squares[59] = EMPTY

        return move

    def has_insufficient_material(self) -> bool:
        """Return True if the position is a forced draw by insufficient material."""
        pieces = [(sq, piece) for sq, piece in enumerate(self.squares) if piece != EMPTY and piece_type(piece) != "K"]
        if not pieces:
            return True
        if len(pieces) == 1 and piece_type(pieces[0][1]) in {"B", "N"}:
            return True
        if len(pieces) == 2 and all(piece_type(piece) == "B" for _, piece in pieces):
            colors = {color_of(piece) for _, piece in pieces}
            bishop_square_colors = {square_color(sq) for sq, _ in pieces}
            return colors == {WHITE, BLACK} and len(bishop_square_colors) == 1
        return False

    def is_fifty_move_draw(self) -> bool:
        """Return True if the halfmove clock has reached 100 (fifty full moves without pawn/capture)."""
        return self.halfmove_clock >= 100

    def is_threefold_repetition(self) -> bool:
        """Return True if the current position has occurred three or more times in the game."""
        return Counter(self.hash_history)[self.zobrist_hash] >= 3

    def is_checkmate(self) -> bool:
        """Return True if the side to move is in checkmate (in check with no legal moves)."""
        return self.is_check(self.turn) and not self.has_legal_moves()

    def is_stalemate(self) -> bool:
        """Return True if the side to move is stalemated (no legal moves but not in check)."""
        return not self.is_check(self.turn) and not self.has_legal_moves()

    def outcome(self, claim_draws: bool = True) -> str | None:
        """Return the game result string ('1-0', '0-1', '1/2-1/2') or None if ongoing.

        Parameters
        ----------
        claim_draws:
            When True, claim draws by the fifty-move rule, threefold repetition,
            or insufficient material before checking for checkmate/stalemate.

        Returns
        -------
        ``'1-0'`` if White wins, ``'0-1'`` if Black wins, ``'1/2-1/2'`` for a draw,
        or ``None`` if the game is still in progress.
        """
        if self.has_insufficient_material():
            return "1/2-1/2"
        if claim_draws and (self.is_fifty_move_draw() or self.is_threefold_repetition()):
            return "1/2-1/2"
        if self.has_legal_moves():
            return None
        if self.is_check(self.turn):
            return "0-1" if self.turn == WHITE else "1-0"
        return "1/2-1/2"

    def result_values(self) -> tuple[float, float] | None:
        """Return (white_reward, black_reward) supporting the asymmetric draw-as-loss penalty."""
        result = self.outcome()
        if result is None:
            return None
        return game_result_to_values(result)

    def result_value(self, perspective: int) -> float | None:
        """Backward compatibility for zero-sum value evaluation paths."""
        res = self.result_values()
        if res is None:
            return None
        return res[0] if perspective == WHITE else res[1]

    def san(self, move: Move) -> str:
        """Return SAN for a legal move in the current position."""
        piece = self.squares[move.from_sq]
        kind = piece_type(piece)
        if move.flags & KING_CASTLE:
            san = "O-O"
        elif move.flags & QUEEN_CASTLE:
            san = "O-O-O"
        else:
            capture = move.is_capture
            prefix = "" if kind == "P" else kind
            if kind != "P":
                prefix += self._san_disambiguator(move, kind)
            elif capture:
                prefix += chr(ord("a") + (move.from_sq & 7))
            san = prefix + ("x" if capture else "") + square_name(move.to_sq)
            if move.promotion:
                san += f"={move.promotion}"

        self.push(move)
        if self.is_check(self.turn):
            san += "#" if not self.legal_moves() else "+"
        self.pop()
        return san

    def _san_disambiguator(self, move: Move, kind: str) -> str:
        candidates = []
        for other in self.legal_moves():
            if other == move or other.to_sq != move.to_sq:
                continue
            piece = self.squares[other.from_sq]
            if piece != EMPTY and color_of(piece) == self.turn and piece_type(piece) == kind:
                candidates.append(other.from_sq)
        if not candidates:
            return ""
        same_file = any((sq & 7) == (move.from_sq & 7) for sq in candidates)
        same_rank = any((sq >> 3) == (move.from_sq >> 3) for sq in candidates)
        if not same_file:
            return chr(ord("a") + (move.from_sq & 7))
        if not same_rank:
            return str((move.from_sq >> 3) + 1)
        return square_name(move.from_sq)