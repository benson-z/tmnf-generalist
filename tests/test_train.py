"""Tests for the training side: labels, augmentation, the input boundary, the model."""

from __future__ import annotations

import inspect
from dataclasses import replace

import numpy as np
import pytest
import torch

from tmnf_train import actions
from tmnf_train.config import Config, DataConfig, ModelConfig, load
from tmnf_train.data import frames as frames_mod
from tmnf_train.data.dataset import (
    OBSERVATION_FIELDS,
    Labels,
    Observation,
    epoch_plan,
    frame_indices,
    mirror_window,
    to_pair,
)
from tmnf_train.model import Model


def small_cfg(**data) -> Config:
    cfg = Config()
    cfg.data = replace(DataConfig(), window=4, downsample=4, **data)
    cfg.model = replace(ModelConfig(), width=64, layers=2, heads=4, tokens_per_frame=4, channels=[8, 16, 16])
    return cfg


# ------------------------------------------------------------- actions

def test_action_layout_roundtrip():
    for a in range(actions.N_ACTIONS):
        assert actions.encode(*actions.decode(a)) == a
    assert actions.keys(actions.encode(1, 0, actions.STEER_LEFT)) == (True, False, True, False)


def test_majority_labels():
    up = np.array([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]], bool)
    down = np.zeros_like(up)
    left = np.array([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]], bool)
    right = np.array([[0, 0, 0, 0, 0], [1, 1, 1, 1, 1]], bool)  # row 2: L+R overlap counts as neither
    lab = actions.from_ticks(up, down, left, right)
    assert actions.decode(lab[0]) == (1, 0, actions.STEER_LEFT)
    # right-only ticks are 3 and 4 -> only 2 -> no steer
    assert actions.decode(lab[1]) == (0, 0, actions.STEER_NONE)


# ------------------------------------------------------------ mirroring

def test_mirror_table_swaps_left_right_only():
    for a in range(actions.N_ACTIONS):
        g, b, s = actions.decode(a)
        mg, mb, ms = actions.decode(int(actions.MIRROR[a]))
        assert (mg, mb) == (g, b)
        assert ms == {0: 2, 1: 1, 2: 0}[s]
    assert np.array_equal(actions.MIRROR[actions.MIRROR], np.arange(actions.N_ACTIONS))


def test_mirror_window_flips_image_and_swaps_steer_labels():
    t, h, w = 3, 6, 8
    frames = np.zeros((t, h, w, 3), np.uint8)
    frames[:, :, 0] = 255  # a bright column on the left edge
    act = np.array([actions.encode(1, 0, 0), actions.encode(0, 1, 2), -1])  # left, right, unlabelled
    path = np.zeros((t, 2, 3), np.float32)
    path[..., 0] = 1.5  # lateral
    path[..., 1] = 7.0  # forward
    f2, a2, p2 = mirror_window(frames, act, path)
    assert (f2[:, :, -1] == 255).all() and (f2[:, :, 0] == 0).all()
    assert actions.decode(a2[0]) == (1, 0, actions.STEER_RIGHT)
    assert actions.decode(a2[1]) == (0, 1, actions.STEER_LEFT)
    assert a2[2] == -1
    assert (p2[..., 0] == -1.5).all() and (p2[..., 1] == 7.0).all()
    # Mirroring twice is the identity.
    f3, a3, p3 = mirror_window(f2, a2, p2)
    assert np.array_equal(f3, frames) and np.array_equal(a3, act) and np.array_equal(p3, path)


def test_mirror_moves_soft_label_mass_between_left_and_right():
    soft = np.zeros((2, actions.N_ACTIONS), np.float32)
    soft[0, actions.encode(1, 0, actions.STEER_LEFT)] = 0.4
    soft[0, actions.encode(1, 0, actions.STEER_NONE)] = 0.6
    soft[1, actions.encode(0, 1, actions.STEER_RIGHT)] = 1.0
    frames = np.zeros((2, 4, 4, 3), np.uint8)
    act = np.array([actions.encode(1, 0, 1), actions.encode(0, 1, 2)])
    *_, s2 = mirror_window(frames, act, np.zeros((2, 1, 3), np.float32), soft)
    assert s2[0, actions.encode(1, 0, actions.STEER_RIGHT)] == np.float32(0.4)
    assert s2[0, actions.encode(1, 0, actions.STEER_NONE)] == np.float32(0.6)
    assert s2[1, actions.encode(0, 1, actions.STEER_LEFT)] == 1.0
    assert np.allclose(s2.sum(1), soft.sum(1))


def test_per_tick_actions_sum_to_soft_label():
    up = np.array([[1, 1, 1, 1, 0]], bool)
    down = np.zeros_like(up)
    left = np.array([[1, 1, 0, 0, 0]], bool)
    right = np.array([[0, 1, 0, 0, 0]], bool)  # tick 1 has both: no steer
    per = actions.per_tick(up, down, left, right)[0]
    assert [actions.decode(a) for a in per] == [(1, 0, 0), (1, 0, 1), (1, 0, 1), (1, 0, 1), (0, 0, 1)]


# ------------------------------------------------------ input boundary

CAR_STATE = {"position", "velocity", "yaw_pitch_roll", "local_speed", "checkpoints", "action",
             "path", "progress", "gas_f", "steer_f", "keys", "camera_position"}


def test_observation_is_frames_and_speed_only():
    assert Observation._fields == OBSERVATION_FIELDS == ("frames", "speed")
    assert not CAR_STATE & set(Observation._fields)
    # The model's forward takes an Observation and nothing else.
    params = list(inspect.signature(Model.forward).parameters)
    assert params == ["self", "obs"]


def test_loader_split_keeps_labels_out_of_observation():
    b, t, k = 2, 4, 3
    chunk = {
        "frames": torch.zeros(b, t + 1, 8, 8, 3, dtype=torch.uint8), "speed": torch.ones(b, t),
        "action": torch.zeros(b, t, dtype=torch.long), "action_soft": torch.zeros(b, t, 12),
        "path": torch.zeros(b, t, k, 3),
        "path_ok": torch.ones(b, t, k, dtype=torch.bool), "progress": torch.zeros(b, t),
        "progress_ok": torch.ones(b, t, dtype=torch.bool),
        "chunk_soft": torch.zeros(b, t, 2, 12), "chunk_ok": torch.ones(b, t, 2, dtype=torch.bool),
    }
    obs, lab = to_pair(chunk, slice(None))
    assert isinstance(obs, Observation) and isinstance(lab, Labels)
    assert set(obs._asdict()) == {"frames", "speed"}


def test_model_rejects_anything_but_observation_and_ignores_labels():
    cfg = small_cfg()
    torch.manual_seed(0)
    model = Model(cfg, n_horizons=2).eval()
    h, w = cfg.input_hw
    obs = Observation(torch.randint(0, 255, (1, 5, h, w, 3), dtype=torch.uint8), torch.rand(1, 4) * 100)
    with pytest.raises(AssertionError):
        model({"frames": obs.frames, "speed": obs.speed, "position": torch.zeros(1, 4, 3)})
    with torch.no_grad():
        a = model(obs)["logits"]
        b = model(Observation(obs.frames.clone(), obs.speed.clone()))["logits"]
    assert torch.equal(a, b)


def test_action_chunk_labels_are_the_next_steps_soft_actions(monkeypatch):
    from tmnf_train.data import dataset as ds

    n, t, nc = 12, 4, 3
    rng = np.random.default_rng(0)
    ticks = np.zeros((n, 12), dtype=np.uint8)
    for i in range(n):  # 5 ticks spread over random actions
        np.add.at(ticks[i], rng.integers(0, 12, 5), 1)
    valid = np.ones(n, dtype=bool)
    valid[-1] = False
    lab = {"valid": valid, "action": ticks.argmax(1), "tick_actions": ticks, "speed": np.zeros(n, np.float32),
           "path": np.zeros((n, 2, 3), np.float32), "path_ok": np.ones((n, 2), bool),
           "progress": np.zeros(n, np.float32), "progress_ok": np.ones(n, bool)}
    cfg = small_cfg(action_chunk=nc)  # window 4
    h, w = cfg.input_hw
    monkeypatch.setattr(ds.index_mod, "load_run", lambda c, name: lab)
    monkeypatch.setattr(ds.RunFrames, "load", lambda self, name, k: np.zeros((k, h, w, 3), np.uint8))
    manifest = {"path_mean": [0, 0, 0], "path_std": [1, 1, 1], "progress_mean": 0.0, "progress_std": 1.0}
    specs = [ds.WindowSpec("r", 6, False), ds.WindowSpec("r", 6, True)]
    batch = ds.RunWindows(cfg.data, manifest, [("r", n, specs)])[0]
    soft = ticks.astype(np.float32) / 5.0
    for b, mirror in enumerate((False, True)):
        for i in range(t):
            f = 6 + i
            for j in range(nc):
                fut = f + 1 + j
                ok = fut < n and valid[fut]
                assert bool(batch["chunk_ok"][b, i, j]) == ok
                want = soft[fut] if ok else np.zeros(12, np.float32)
                if mirror:
                    want = want[actions.MIRROR]
                assert np.allclose(batch["chunk_soft"][b, i, j].numpy(), want)
    # Step t+1 of frame t is the next frame's own soft label.
    assert torch.allclose(batch["chunk_soft"][:, :-1, 0], batch["action_soft"][:, 1:])


def test_chunk_head_only_when_enabled():
    assert Model(small_cfg(), n_horizons=2).chunk is None
    cfg = small_cfg(action_chunk=5)
    model = Model(cfg, n_horizons=2).eval()
    h, w = cfg.input_hw
    obs = Observation(torch.randint(0, 255, (1, 5, h, w, 3), dtype=torch.uint8), torch.rand(1, 4) * 100)
    with torch.no_grad():
        assert model(obs)["chunk_logits"].shape == (1, 4, 5, 12)


# ---------------------------------------------------------------- model

def test_stride2_stem_runs_before_trunk():
    cfg = Config()  # default 320x240
    enc = Model(cfg, 6).encoder
    seen = []
    enc.stem.register_forward_hook(lambda m, i, o: seen.append(("stem", tuple(i[0].shape[-2:]), tuple(o.shape[-2:]))))
    enc.stages[0].conv.register_forward_hook(lambda m, i, o: seen.append(("trunk", tuple(i[0].shape[-2:]), tuple(o.shape[-2:]))))
    with torch.no_grad():
        enc(torch.zeros(1, 6, 240, 320))
    assert enc.stem.stride == (2, 2)
    assert seen[0] == ("stem", (240, 320), (120, 160))
    assert seen[1] == ("trunk", (120, 160), (120, 160))


def test_param_count_in_range():
    n = sum(p.numel() for p in Model(Config(), 6).parameters())
    assert 20e6 <= n <= 40e6, n


def test_inference_paths_match_full_window():
    cfg = small_cfg()
    torch.manual_seed(0)
    model = Model(cfg, n_horizons=2).eval()
    h, w = cfg.input_hw
    T = cfg.data.window
    n = T + 5  # run past the window so the cache has to slide
    frames = torch.randint(0, 255, (1, n, h, w, 3), dtype=torch.uint8)
    speed = torch.rand(1, n) * 300
    cache = None
    tokens = []
    with torch.no_grad():
        for i in range(n):
            prev = frames[:, max(i - 1, 0)]
            pair = model.preprocess(torch.stack([prev, frames[:, i]], 1))[:, 0]
            out, cache = model.step(pair, speed[:, i], cache)
            start = max(0, i - T + 1)
            idx = [max(start - 1, 0)] + list(range(start, i + 1))
            full = model(Observation(frames[:, idx], speed[:, start : i + 1]))
            tokens.append(model.encode_step(pair, speed[:, i]))
            exact = model.core_last(torch.stack(tokens[-T:], 1))
            # Recompute mode is exact at every step.
            torch.testing.assert_close(exact["logits"][0, 0], full["logits"][0, -1], atol=1e-4, rtol=1e-4)
            # The rolling KV cache is exact until it starts to slide (see Model.step).
            if i < T:
                torch.testing.assert_close(out["logits"][0, 0], full["logits"][0, -1], atol=1e-4, rtol=1e-4)


# ------------------------------------------------------------- data

def test_integer_downsampling_is_exact_box_average():
    src = np.random.default_rng(0).integers(0, 256, (2, 240, 320, 3), dtype=np.uint8)
    half = frames_mod.to_input(src, 2, None)
    assert half.shape == (2, 120, 160, 3)
    blk = src.reshape(2, 120, 2, 160, 2, 3).astype(np.int64).sum((2, 4))
    assert np.array_equal(half, ((blk + 2) // 4).astype(np.uint8))
    assert np.array_equal(frames_mod.to_input(src, 1, None), src)
    cropped = frames_mod.to_input(src, 2, [40, 240])
    assert cropped.shape == (2, 100, 160, 3)
    assert np.array_equal(cropped, half[:, 20:])


def test_frame_indices_prev_and_context():
    d = replace(DataConfig(), window=4, frame_stride=2, context_offsets=[20, 40])
    idx = frame_indices(50, d)
    assert list(idx) == [30, 10, 48, 50, 52, 54, 56]
    assert list(frame_indices(0, replace(d, context_offsets=[])))[:2] == [0, 0]


def test_epoch_plan_is_deterministic_and_covers_runs_once():
    d = replace(DataConfig(), window=16)
    runs = [{"name": f"r{i}", "n": 100 + 37 * i} for i in range(30)]
    a = epoch_plan(runs, d, seed=1, epoch=0)
    b = epoch_plan(runs, d, seed=1, epoch=0)
    c = epoch_plan(runs, d, seed=1, epoch=1)
    flat = lambda p: [(s.run, s.start, s.mirror) for blk in p for s in blk]
    assert flat(a) == flat(b) and flat(a) != flat(c)
    # Non-overlapping windows within each run.
    for r in runs:
        starts = sorted(s for (n, s, _) in flat(a) if n == r["name"])
        assert all(y - x >= 16 for x, y in zip(starts, starts[1:]))
        assert starts[-1] + 16 <= r["n"]


def test_config_rejects_unknown_keys(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("model:\n  widht: 3\n")
    with pytest.raises(ValueError):
        load(p)


@pytest.mark.parametrize("inference,context", [("kv_cache", []), ("recompute", []), ("recompute", [4, 8])])
def test_model_episode_runs_every_inference_path(inference, context):
    from tmnf_train.policy_model import ModelPolicy

    cfg = small_cfg(context_offsets=context)
    torch.manual_seed(0)
    model = Model(cfg, n_horizons=2).eval()
    policy = ModelPolicy(model, cfg, "t", torch.device("cpu"), inference)
    rng = np.random.default_rng(0)
    for temperature in (0.0, 0.5):
        ep = policy.episode(seed=3, temperature=temperature)
        ep2 = policy.episode(seed=3, temperature=temperature)
        for _ in range(12):
            frame = rng.integers(0, 256, (240, 320, 3), dtype=np.uint8)
            a, p = ep.act(frame, 120.0)
            b, _ = ep2.act(frame, 120.0)
            assert 0 <= a < actions.N_ACTIONS and abs(p.sum() - 1) < 1e-5
            assert a == b  # same seed, same inputs, same actions
