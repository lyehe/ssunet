# SPAD Simulation Bug: Correlated RNG in DataLoader Workers

## Summary

Two test scenes from the Single Photon Challenge (`blue-bathroom` and `train-passby`) contain correlated binary frames caused by a bug in the SPAD emulation pipeline. The bug is in `visionsim`'s old `emulate.spad()` function, which passed a numpy RNG object into a multi-worker PyTorch DataLoader via `functools.partial`. Each worker received a pickled copy of the same RNG state, so every batch of 4 frames shared identical random thresholds.

## Root Cause

In `visionsim/cli/emulate.py` (commit `ece502c` and earlier):

```python
rng = np.random.default_rng(int(seed))
loader = DataLoader(
    dataset,
    batch_size=4,
    num_workers=os.cpu_count() or 1,
    collate_fn=functools.partial(_spad_collate, rng=rng, ...),
)
```

When PyTorch spawns/forks DataLoader workers, each gets a **copy** of the `rng` object at the same state. The `collate_fn` runs **inside the worker process**, so all workers generate the same random numbers for their first batch, then the same numbers for their second batch, etc.

Reproduced and confirmed:

```
frames [ 0- 3]: randoms=[0.773956, 0.438878, 0.858598, 0.697368]  pid=171616
frames [ 4- 7]: randoms=[0.773956, 0.438878, 0.858598, 0.697368]  pid=165272  ← identical
frames [ 8-11]: randoms=[0.773956, 0.438878, 0.858598, 0.697368]  pid=172340  ← identical
frames [12-15]: randoms=[0.773956, 0.438878, 0.858598, 0.697368]  pid=170344  ← identical
```

Each worker advances its RNG by the same amount per batch (all images are the same size), so workers remain synchronized indefinitely. With `N` workers and `batch_size=4`, frames `t` and `t+4` always share random thresholds (from different workers in the same round).

## Fix in visionsim

Fixed in commit `bd50b1c` ("Evolving the dataset formats") by replacing the DataLoader with a sequential loop:

```python
rng = np.random.default_rng(int(seed))
for i, (data, transform) in enumerate(dataset):
    imgs = emulate_spc(data, factor=factor, rng=rng)  # single rng, advances properly
```

## Evidence

### Gap analysis

For each chunk, we measure the fraction of packed bytes that differ at gap=1, gap=4, and gap=8. In clean data, all gaps should give similar diff rates. The bug produces sharp dips at gap=4 (cross-worker correlation) and gap=8 (same-worker, next batch).

**train-passby** — gap=1 vs gap=4 vs gap=8:

```
chunk  0: g1=73.5%  g4= 0.6%  g8= 1.2%   ← strong period-4 and period-8
chunk  5: g1=32.7%  g4= 0.3%  g8= 0.5%   ← strong
chunk  9: g1=29.8%  g4=13.3%  g8=20.2%   ← masked by scene motion, but gap-8 reveals it
chunk 10: g1=35.7%  g4=35.7%  g8=23.5%   ← gap-4 looks clean, gap-8 still shows dip!
chunk 11: g1=34.6%  g4=26.2%  g8=34.7%   ← gap-4 affected, gap-8 clean (transitional)
chunk 12: g1=32.5%  g4=29.8%  g8=32.6%   ← clean
chunk 20: g1=40.5%  g4=40.5%  g8=40.5%   ← clean
```

**blue-bathroom** — all 37 chunks affected:

```
chunk  0: g1=75.9%  g4= 0.2%  g8= 0.3%   ← extreme period-4
chunk 20: g1=87.7%  g4=23.1%  g8=31.2%   ← masked at gap-4, gap-8 still shows dip
chunk 31: g1=69.3%  g4=66.6%  g8=47.0%   ← gap-4 looks clean (r4=0.96), gap-8 reveals bug (r8=0.68)
chunk 35: g1=72.7%  g4=70.7%  g8=72.8%   ← truly clean (one of only ~2 clean chunks)
```

**basement (clean reference)**: all gaps equal at 16.2% — no periodicity.

### Why some chunks appear clean at gap-4 but aren't

The bug gives frames 4 apart the same random thresholds, but they sample from different input images (animated scene). When the scene changes slowly, same thresholds + same image = identical binary output (strong period-4). When the scene changes fast (train passing, camera moving), same thresholds + different images = different binary output, masking the gap-4 correlation.

But the gap-8 correlation (same worker's second batch) persists longer because it reflects a structural RNG correlation, not just threshold matching. This is why gap-8 catches chunks that gap-4 misses.

## Affected Chunks

### train-passby (12 of 37 chunks affected)

| Chunk | Frames | r4 | r8 | Status |
|-------|--------|------|------|--------|
| 0 | 0–1023 | 0.008 | 0.016 | FIXED |
| 1 | 1024–2047 | 0.022 | 0.041 | FIXED |
| 2 | 2048–3071 | 0.025 | 0.046 | FIXED |
| 3 | 3072–4095 | 0.008 | 0.015 | FIXED |
| 4 | 4096–5119 | 0.007 | 0.013 | FIXED |
| 5 | 5120–6143 | 0.009 | 0.016 | FIXED |
| 6 | 6144–7167 | 0.010 | 0.019 | FIXED |
| 7 | 7168–8191 | 0.006 | 0.011 | FIXED |
| 8 | 8192–9215 | 0.010 | 0.019 | FIXED |
| 9 | 9216–10239 | 0.360 | 0.416 | FIXED |
| 10 | 10240–11263 | 1.000 | 0.530 | FIXED (gap-8 only) |
| 11 | 11264–12287 | 0.804 | 0.809 | FIXED |
| 12–36 | 12288–37887 | ~1.0 | ~1.0 | Clean |

### blue-bathroom (37 of 37 chunks affected)

| Chunk | Frames | r4 | r8 | Status |
|-------|--------|------|------|--------|
| 0–19 | 0–20479 | 0.002–0.076 | 0.003–0.132 | FIXED (strong) |
| 20–30 | 20480–31743 | 0.199–0.432 | 0.202–0.444 | FIXED (moderate) |
| 31 | 31744–32767 | 0.968 | 0.613 | FIXED (gap-8 only) |
| 32 | 32768–33791 | 0.975 | 0.814 | FIXED (gap-8 only) |
| 33 | 33792–34815 | 0.368 | 0.570 | FIXED |
| 34 | 34816–35839 | 0.963 | 0.612 | FIXED (gap-8 only) |
| 35 | 35840–36863 | 0.978 | 0.808 | FIXED (gap-8 only) |
| 36 | 36864–37887 | 0.377 | 0.578 | FIXED |

### Clean scenes (not affected)

| Scene | r4 | r8 |
|-------|------|------|
| coffee-office-bar | ~1.0 | ~1.0 |
| basement | ~1.0 | ~1.0 |
| bedroom-interior | ~1.0 | ~1.0 |

## Fix Applied: Spatial Rate Estimation + Bernoulli Resampling (`fix_spad.py`)

For affected chunks, per-pixel firing rates are estimated using spatially-smoothed temporal windowing, then all frames are resampled as independent `Bernoulli(rate_t)` trials:

1. **Detect**: compute gap-4 and gap-8 ratios; flag if either < 0.9
2. **Estimate rates at anchor points**: divide the 1024 frames into overlapping temporal windows, compute per-window mean, then apply Gaussian spatial smoothing (sigma=2.0) to break the 5-level quantization ({0, 0.25, 0.50, 0.75, 1.00}) and recover continuous rate estimates
3. **Interpolate per-frame rates**: linearly interpolate between anchor points to get per-frame rate maps, preserving scene animation
4. **Resample**: replace all 1024 frames with independent `Bernoulli(rate_t)` draws using a fresh RNG

### Why spatial smoothing works

The bug quantizes rates to 5 levels because only 4 independent RNG draws exist per pixel. Neighboring pixels have similar true rates but different quantization outcomes, so spatial Gaussian smoothing recovers continuous rates while preserving edges. Temporal windowing (default: 8 windows) preserves scene animation that would be lost by collapsing to a single static rate.

### Verification

After correction, gap-4 and gap-8 ratios return to ~1.0, confirming complete elimination of temporal correlation.
