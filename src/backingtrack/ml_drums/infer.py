# src/backingtrack/ml_drums/infer.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from ..types import BarGrid, Note, TimeSignature
from .data import VOICE_MAP, N_VOICES
from .model import DrumModelConfig, DrumTransformer


# Inverse map: voice_index -> GM drum pitch
INV_VOICE_MAP: Dict[int, int] = {v: k for k, v in VOICE_MAP.items()}

# Pick which voice index is “hi-hat-ish” to enforce hats if needed
HAT_VOICES = [VOICE_MAP.get(42), VOICE_MAP.get(46), VOICE_MAP.get(51)]  # closed, open, ride
HAT_VOICES = [v for v in HAT_VOICES if v is not None]


@dataclass(frozen=True)
class SampleConfig:
    bars: int = 16
    temperature: float = 1.0      # >1.0 = more random, <1.0 = more conservative
    threshold: float = 0.45       # higher = fewer hits (when stochastic=False)
    stochastic: bool = True       # sample Bernoulli vs deterministic threshold
    seed: Optional[int] = None

    # Practical constraints
    force_hats: bool = True
    max_nonhat_hits_per_step: int = 2  # limit big stacks (kick+snare+tom+crash all at once)

    # Velocity shaping
    vel_floor: int = 18
    vel_ceiling: int = 120
    intensity: float = 0.75       # 0..1 overall loudness multiplier


def load_model(model_path: str | Path, device: Optional[str] = None) -> Tuple[DrumTransformer, DrumModelConfig, torch.device]:
    p = Path(model_path)
    if not p.exists():
        raise FileNotFoundError(f"Model not found: {p}")

    dev = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(str(p), map_location=dev)

    cfg = DrumModelConfig(**ckpt["cfg"])
    model = DrumTransformer(cfg).to(dev)
    model.load_state_dict(ckpt["state"])
    model.eval()
    return model, cfg, dev


def _sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x)


def sample_one_bar(
    model: DrumTransformer,
    cfg: DrumModelConfig,
    device: torch.device,
    scfg: SampleConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      hits: (16, V) in {0,1}
      vels: (16, V) in [0,1]
    """
    rng = np.random.default_rng(scfg.seed)

    V = cfg.n_voices
    assert V == N_VOICES, f"Model voices={V}, but code expects N_VOICES={N_VOICES}. Check VOICE_MAP."

    hits = np.zeros((cfg.steps, V), dtype=np.float32)
    vels = np.zeros((cfg.steps, V), dtype=np.float32)

    for t in range(cfg.steps):
        # Build teacher-forcing style input:
        # x[0] is zeros, x[t] contains previous step's events
        x_hits = np.zeros((cfg.steps, V), dtype=np.float32)
        x_vels = np.zeros((cfg.steps, V), dtype=np.float32)
        if t > 0:
            x_hits[1 : t + 1] = hits[0:t]
            x_vels[1 : t + 1] = vels[0:t]

        x = np.concatenate([x_hits, x_vels], axis=1)  # (16, 2V)
        xt = torch.tensor(x[None, :, :], dtype=torch.float32, device=device)  # (1,16,2V)

        with torch.no_grad():
            hit_logits, vel_pred = model(xt)  # (1,16,V), (1,16,V)

        logits_t = hit_logits[0, t, :] / max(1e-6, float(scfg.temperature))
        probs_t = _sigmoid(logits_t).clamp(0.0, 1.0).cpu().numpy()  # (V,)
        vels_t = vel_pred[0, t, :].clamp(0.0, 1.0).cpu().numpy()     # (V,)

        if scfg.stochastic:
            step_hits = (rng.random(V) < probs_t).astype(np.float32)
        else:
            step_hits = (probs_t >= float(scfg.threshold)).astype(np.float32)

        # ---- simple constraints to keep it sane ----
        # limit huge stacks (ignore hats)
        nonhat = [i for i in range(V) if i not in HAT_VOICES]
        nonhat_on = [i for i in nonhat if step_hits[i] > 0.5]
        if len(nonhat_on) > scfg.max_nonhat_hits_per_step:
            # keep the most probable ones
            nonhat_on.sort(key=lambda i: probs_t[i], reverse=True)
            for i_drop in nonhat_on[scfg.max_nonhat_hits_per_step :]:
                step_hits[i_drop] = 0.0

        # force a hat most steps (helps groove feel)
        if scfg.force_hats and HAT_VOICES:
            if sum(step_hits[i] for i in HAT_VOICES) < 0.5:
                # choose best hat candidate
                best_hat = max(HAT_VOICES, key=lambda i: probs_t[i])
                step_hits[best_hat] = 1.0

        # velocities only where hit exists
        step_vels = vels_t * step_hits

        hits[t, :] = step_hits
        vels[t, :] = step_vels

    return hits, vels


def bars_to_notes(
    hits: np.ndarray,
    vels: np.ndarray,
    grid: BarGrid,
    *,
    start_bar: int = 0,
    bar_count: int = 1,
    scfg: SampleConfig = SampleConfig(),
) -> List[Note]:
    """
    Convert sampled bar grids into Note events using the BarGrid timing.
    Assumes 16 steps per bar.
    """
    out: List[Note] = []
    step_len = float(grid.bar_duration) / 16.0
    note_dur = min(0.12, step_len * 0.90)

    V = hits.shape[2] if hits.ndim == 3 else hits.shape[1]

    for b in range(bar_count):
        bar_start = float(grid.time_at(start_bar + b, 0.0))
        H = hits[b] if hits.ndim == 3 else hits
        VEL = vels[b] if vels.ndim == 3 else vels

        for step in range(16):
            t0 = bar_start + step * step_len
            t1 = t0 + note_dur

            for vidx in range(V):
                if H[step, vidx] <= 0.5:
                    continue

                pitch = INV_VOICE_MAP.get(vidx)
                if pitch is None:
                    continue

                # map 0..1 velocity to MIDI, with intensity shaping + clamps
                raw = float(VEL[step, vidx]) * 127.0 * float(scfg.intensity)
                vel = int(max(scfg.vel_floor, min(scfg.vel_ceiling, raw)))
                out.append(Note(pitch=int(pitch), start=float(t0), end=float(t1), velocity=int(vel)))

    return out


def generate_ml_drums(
    model_path: str | Path,
    grid: BarGrid,
    scfg: SampleConfig = SampleConfig(),
) -> List[Note]:
    """
    Generate scfg.bars bars of drum notes using the trained model.
    Assumes 4/4 because training extraction was 4/4 bars.
    """
    # safety: keep 4/4 for now
    ts: TimeSignature = grid.time_signature
    if not (ts.numerator == 4 and ts.denominator == 4):
        raise ValueError("ML drums currently expects 4/4 (trained on 4/4 bars).")

    model, cfg, dev = load_model(model_path)

    rng = np.random.default_rng(scfg.seed)
    all_hits = []
    all_vels = []

    # for variation, change seed per bar deterministically
    base_seed = scfg.seed if scfg.seed is not None else int(rng.integers(0, 1_000_000))

    for b in range(scfg.bars):
        bar_seed = base_seed + b * 997  # deterministic but varied
        bar_cfg = SampleConfig(**{**scfg.__dict__, "seed": bar_seed})
        h, v = sample_one_bar(model, cfg, dev, bar_cfg)
        all_hits.append(h)
        all_vels.append(v)

    hits = np.stack(all_hits, axis=0)  # (bars,16,V)
    vels = np.stack(all_vels, axis=0)  # (bars,16,V)

    return bars_to_notes(hits, vels, grid, start_bar=0, bar_count=scfg.bars, scfg=scfg)
