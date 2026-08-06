//! ZERO-X's exact AlphaZero board and policy representation.
//!
//! Every position is encoded from the perspective of the side to move. Black
//! positions are rotated 180 degrees and piece colours are swapped.

use cozy_chess::{Board, Color, Move, Piece, Square};
use std::fmt;

pub const HISTORY: usize = 8;
pub const PIECE_PLANES_PER_HISTORY: usize = 14;
pub const INPUT_CHANNELS: usize = HISTORY * PIECE_PLANES_PER_HISTORY + 9;
pub const BOARD_SQUARES: usize = 64;
pub const INPUT_SIZE: usize = INPUT_CHANNELS * BOARD_SQUARES;
pub const POLICY_PLANES: usize = 73;
pub const POLICY_SIZE: usize = POLICY_PLANES * BOARD_SQUARES;
pub const POLICY_MASK_WORDS: usize = POLICY_SIZE.div_ceil(64);

pub type EncodedBoard = [f32; INPUT_SIZE];

#[derive(Clone)]
pub struct HistoryPosition {
    pub board: Board,
    pub repetitions: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncodingError {
    InvalidUnderpromotion,
    InvalidMoveGeometry,
    SquareOutOfRange,
}

impl fmt::Display for EncodingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidUnderpromotion => f.write_str("invalid underpromotion geometry"),
            Self::InvalidMoveGeometry => f.write_str("move is neither queen-like nor knight-like"),
            Self::SquareOutOfRange => f.write_str("square is outside the board"),
        }
    }
}

impl std::error::Error for EncodingError {}

#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct PolicyMask(pub [u64; POLICY_MASK_WORDS]);

impl Default for PolicyMask {
    fn default() -> Self {
        Self([0; POLICY_MASK_WORDS])
    }
}

impl PolicyMask {
    #[inline]
    pub fn set(&mut self, index: usize) {
        if index < POLICY_SIZE {
            self.0[index >> 6] |= 1_u64 << (index & 63);
        }
    }

    #[inline]
    pub fn contains(&self, index: usize) -> bool {
        index < POLICY_SIZE && (self.0[index >> 6] & (1_u64 << (index & 63))) != 0
    }

    pub fn write_f32_mask(&self, out: &mut [f32]) {
        if out.len() < POLICY_SIZE {
            return;
        }
        for (index, value) in out[..POLICY_SIZE].iter_mut().enumerate() {
            *value = self.contains(index) as u8 as f32;
        }
    }
}

#[inline]
pub const fn orient_square(square: usize, side_to_move: Color) -> usize {
    match side_to_move {
        Color::White => square,
        Color::Black => 63 - square,
    }
}

#[inline]
fn square_index(square: Square) -> usize {
    square as usize
}

pub fn encode_board_into(
    board: &Board,
    history: &[HistoryPosition],
    current_repetitions: u8,
    out: &mut EncodedBoard,
) {
    out.fill(0.0);
    let perspective = board.side_to_move();

    encode_position_planes(board, current_repetitions, perspective, 0, out);
    for (history_index, entry) in history.iter().take(HISTORY - 1).enumerate() {
        encode_position_planes(
            &entry.board,
            entry.repetitions,
            perspective,
            history_index + 1,
            out,
        );
    }

    let extra = HISTORY * PIECE_PLANES_PER_HISTORY;
    if perspective == Color::White {
        fill_plane(out, extra, 1.0);
    }

    let opponent = !perspective;
    if board.castle_rights(perspective).short.is_some() {
        fill_plane(out, extra + 1, 1.0);
    }
    if board.castle_rights(perspective).long.is_some() {
        fill_plane(out, extra + 2, 1.0);
    }
    if board.castle_rights(opponent).short.is_some() {
        fill_plane(out, extra + 3, 1.0);
    }
    if board.castle_rights(opponent).long.is_some() {
        fill_plane(out, extra + 4, 1.0);
    }

    if let Some(file) = board.en_passant() {
        let oriented_file = if perspective == Color::White {
            file as usize
        } else {
            7 - file as usize
        };
        for rank in 0..8 {
            out[(extra + 5) * BOARD_SQUARES + rank * 8 + oriented_file] = 1.0;
        }
    }
    fill_plane(
        out,
        extra + 6,
        (board.fullmove_number().min(512) as f32) / 512.0,
    );
    fill_plane(
        out,
        extra + 7,
        (board.halfmove_clock().min(100) as f32) / 100.0,
    );
    if !board.checkers().is_empty() {
        fill_plane(out, extra + 8, 1.0);
    }
}

#[inline]
fn fill_plane(out: &mut EncodedBoard, plane: usize, value: f32) {
    out[plane * BOARD_SQUARES..(plane + 1) * BOARD_SQUARES].fill(value);
}

fn encode_position_planes(
    board: &Board,
    repetitions: u8,
    perspective: Color,
    history_index: usize,
    out: &mut EncodedBoard,
) {
    let base = history_index * PIECE_PLANES_PER_HISTORY;
    for piece in [
        Piece::Pawn,
        Piece::Knight,
        Piece::Bishop,
        Piece::Rook,
        Piece::Queen,
        Piece::King,
    ] {
        let piece_offset = piece as usize;
        for colour in [perspective, !perspective] {
            let colour_offset = if colour == perspective { 0 } else { 6 };
            let squares = board.pieces(piece) & board.colors(colour);
            for square in squares {
                let oriented = orient_square(square_index(square), perspective);
                out[(base + colour_offset + piece_offset) * BOARD_SQUARES + oriented] = 1.0;
            }
        }
    }
    if repetitions >= 2 {
        fill_plane(out, base + 12, 1.0);
    }
    if repetitions >= 3 {
        fill_plane(out, base + 13, 1.0);
    }
}

pub fn move_to_policy_index(board: &Board, chess_move: Move) -> Result<usize, EncodingError> {
    let perspective = board.side_to_move();
    let from = orient_square(square_index(chess_move.from), perspective);
    let to = orient_square(square_index(chess_move.to), perspective);
    if from >= 64 || to >= 64 {
        return Err(EncodingError::SquareOutOfRange);
    }
    let from_file = (from & 7) as i8;
    let from_rank = (from >> 3) as i8;
    let df = (to & 7) as i8 - from_file;
    let dr = (to >> 3) as i8 - from_rank;

    if let Some(promotion) = chess_move.promotion {
        if promotion != Piece::Queen {
            if dr != 1 || !(-1..=1).contains(&df) {
                return Err(EncodingError::InvalidUnderpromotion);
            }
            let promotion_offset = match promotion {
                Piece::Knight => 0,
                Piece::Bishop => 1,
                Piece::Rook => 2,
                _ => return Err(EncodingError::InvalidUnderpromotion),
            };
            let plane = 64 + promotion_offset * 3 + (df + 1) as usize;
            return Ok(plane * 64 + from);
        }
    }

    const KNIGHT_DIRECTIONS: [(i8, i8); 8] = [
        (1, 2),
        (2, 1),
        (2, -1),
        (1, -2),
        (-1, -2),
        (-2, -1),
        (-2, 1),
        (-1, 2),
    ];
    if let Some(direction) = KNIGHT_DIRECTIONS
        .iter()
        .position(|&delta| delta == (df, dr))
    {
        return Ok((56 + direction) * 64 + from);
    }

    const QUEEN_DIRECTIONS: [(i8, i8); 8] = [
        (0, 1),
        (1, 1),
        (1, 0),
        (1, -1),
        (0, -1),
        (-1, -1),
        (-1, 0),
        (-1, 1),
    ];
    let (direction, distance) = if df == 0 && dr != 0 {
        ((0, dr.signum()), dr.unsigned_abs() as usize)
    } else if dr == 0 && df != 0 {
        ((df.signum(), 0), df.unsigned_abs() as usize)
    } else if df != 0 && df.unsigned_abs() == dr.unsigned_abs() {
        ((df.signum(), dr.signum()), df.unsigned_abs() as usize)
    } else {
        return Err(EncodingError::InvalidMoveGeometry);
    };
    let direction_index = QUEEN_DIRECTIONS
        .iter()
        .position(|&item| item == direction)
        .ok_or(EncodingError::InvalidMoveGeometry)?;
    if !(1..=7).contains(&distance) {
        return Err(EncodingError::InvalidMoveGeometry);
    }
    Ok((direction_index * 7 + (distance - 1)) * 64 + from)
}

pub fn legal_policy_mask(board: &Board) -> PolicyMask {
    let mut mask = PolicyMask::default();
    board.generate_moves(|moves| {
        for chess_move in moves {
            if let Ok(index) = move_to_policy_index(board, chess_move) {
                mask.set(index);
            }
        }
        false
    });
    mask
}

#[must_use]
pub const fn flip_policy_index_horizontally(index: usize) -> usize {
    if index >= POLICY_SIZE {
        return index;
    }
    let plane = index / 64;
    let square = index & 63;
    let flipped_square = (square & !7) | (7 - (square & 7));
    let flipped_plane = if plane < 56 {
        let direction = plane / 7;
        let distance = plane % 7;
        let reflected_direction = match direction {
            0 => 0,
            1 => 7,
            2 => 6,
            3 => 5,
            4 => 4,
            5 => 3,
            6 => 2,
            _ => 1,
        };
        reflected_direction * 7 + distance
    } else if plane < 64 {
        56 + (7 - (plane - 56))
    } else {
        let promotion = (plane - 64) / 3;
        let direction = (plane - 64) % 3;
        64 + promotion * 3 + (2 - direction)
    };
    flipped_plane * 64 + flipped_square
}

pub fn flip_augmentation(
    input: &EncodedBoard,
    policy: &[f32; POLICY_SIZE],
    flipped_input: &mut EncodedBoard,
    flipped_policy: &mut [f32; POLICY_SIZE],
) {
    for plane in 0..INPUT_CHANNELS {
        for rank in 0..8 {
            let row = plane * 64 + rank * 8;
            for file in 0..8 {
                flipped_input[row + (7 - file)] = input[row + file];
            }
        }
    }
    let extra = HISTORY * PIECE_PLANES_PER_HISTORY;
    for square in 0..BOARD_SQUARES {
        flipped_input.swap(
            (extra + 1) * BOARD_SQUARES + square,
            (extra + 2) * BOARD_SQUARES + square,
        );
        flipped_input.swap(
            (extra + 3) * BOARD_SQUARES + square,
            (extra + 4) * BOARD_SQUARES + square,
        );
    }
    for (index, probability) in policy.iter().copied().enumerate() {
        flipped_policy[flip_policy_index_horizontally(index)] = probability;
    }
}
