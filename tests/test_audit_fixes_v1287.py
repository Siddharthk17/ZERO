import pytest
import sqlite3
import tempfile
import random
import queue
import torch
import multiprocessing
from pathlib import Path

from zero_chess.replay import PrioritizedReplayBuffer, Experience
from zero_chess.mcts import MCTS, Node, UniformEvaluator
from zero_chess.model import ZeroNet, ModelConfig, load_model, save_model
from zero_chess.self_play import CudaTrainingRuntime, generate_parallel_games
from zero_chess.constants import VIRTUAL_LOSS_VALUE, VIRTUAL_LOSS_VISITS

def test_sqlite_replay_buffer_probing_efficiency(tmp_path) -> None:
    # Set up a replay buffer with cold storage sqlite
    db_path = tmp_path / "test_replay.sqlite3"
    # To trigger deletion of old experiences, we must exceed cold_capacity.
    # Let's set cold_capacity=5. Hot capacity is 10.
    buffer = PrioritizedReplayBuffer(hot_capacity=10, cold_path=db_path, cold_capacity=5)
    
    # We add 20 experiences.
    # The first 10 go to hot.
    # The next 10 are evicted to cold. Since cold_capacity is 5,
    # the first 5 evicted will be deleted, creating an ID gap.
    for i in range(20):
        exp = Experience(
            fen=f"8/8/8/8/8/8/8/k6k w - - 0 {i+1}",
            policy={},
            value=0.0,
            td_value=0.0,
            wdl=(0.0, 1.0, 0.0)
        )
        buffer.add(exp)
        
    # Verify we have some experiences in cold storage
    assert buffer._cold_count > 0
    
    # Manually check the min/max in the DB to verify the gap is present
    conn = sqlite3.connect(db_path)
    min_id, max_id = conn.execute("SELECT MIN(id), MAX(id) FROM experiences").fetchone()
    conn.close()
    
    # The first 5 evictions have IDs 1, 2, 3, 4, 5. They should be deleted.
    # So min_id should be 6.
    assert min_id > 1
    
    # Sampling cold should work without looping infinitely
    sampled = buffer._sample_cold_unlocked(3)
    assert len(sampled) == 3

def test_mcts_unvisited_default_parameterization() -> None:
    mcts = MCTS(UniformEvaluator())
    parent = Node(visit_count=10, is_expanded=True)
    unexplored_child = Node(prior_probability=0.5)
    
    # Compute PUCT score with different unvisited defaults
    score_default = mcts._puct_score(parent, unexplored_child, unvisited_default=-1.0)
    score_custom = mcts._puct_score(parent, unexplored_child, unvisited_default=0.0)
    
    # Since unvisited_default=0.0 means the child is more valuable to the opponent (0.0 vs -1.0),
    # it is less valuable to us. Therefore, score_default should be greater than score_custom.
    assert score_default > score_custom

def test_squeeze_excitation_reduction_recovery(tmp_path) -> None:
    # Create config with low channel size to trigger max(channels // reduction, 8) clamping
    config = ModelConfig(
        channels=32,
        blocks=1,
        se_reduction=16
    )
    # Check that channels // se_reduction = 2, which clamps to 8 hidden channels
    model = ZeroNet(config)
    
    # Save the model
    model_path = tmp_path / "model_clamped.pt"
    save_model(model_path, model)
    
    # Load the model and check if the configuration's se_reduction recovers to 16
    loaded = load_model(model_path, device="cpu")
    assert loaded.config.se_reduction == 16

def test_cuda_training_runtime_stop_drains_queues() -> None:
    # Mock resources that look like multiprocessing queues
    class MockQueue:
        def __init__(self):
            self.closed = False
            self.joined = False
            self.items = [1, 2, 3]

        def close(self):
            self.closed = True

        def get_nowait(self):
            if not self.items:
                raise queue.Empty()
            return self.items.pop(0)

        def join_thread(self):
            self.joined = True

    class MockEvent:
        def __init__(self):
            self.is_set_val = False

        def set(self):
            self.is_set_val = True

    stop_event = MockEvent()
    q1 = MockQueue()
    q2 = MockQueue()
    resources = [q1, [q2], stop_event]
    
    runtime = CudaTrainingRuntime(
        stop_event=stop_event,
        processes=[],
        resources=resources,
        generation_value=None,
        games_completed=None
    )
    
    # Call stop
    runtime.stop()
    
    assert stop_event.is_set_val is True
    assert q1.closed is True
    assert q1.joined is True
    assert len(q1.items) == 0
    assert q2.closed is True
    assert q2.joined is True
    assert len(q2.items) == 0

def test_centralization_of_magic_numbers() -> None:
    # Ensure VIRTUAL_LOSS_VALUE and VIRTUAL_LOSS_VISITS are imported correctly and match target values
    assert VIRTUAL_LOSS_VALUE == 3.0
    assert VIRTUAL_LOSS_VISITS == 3
