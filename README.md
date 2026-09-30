# v2vtrack

Online tracking of the receiver's array offset for position-aided vehicle-to-vehicle (V2V) millimetre-wave
beam prediction, evaluated as streams on DeepSense 6G scenarios 36-39.

The position-aided beam map for a V2V link (relative bearing between the two vehicles minus the receiver's
GPS-derived heading, mapped to a beam index on one of four arrays) fails across recordings: the receiver's
array offset relative to its heading differs by a few degrees between recordings and drifts within one,
and the heading is unobservable at low speed. This repository holds the analysis of that offset, a one-state
Kalman tracker fed by sparse beam sweeps or a local sweep that restores the map online without labels, the
baselines it is compared with (the 2023 challenge static map, beam hold, a beam-index offset tracker, a
windowed refit, a learned GPS-sequence network with test-time adaptation and supervised fine-tuning), a
continual multi-scenario stream, and a known-truth validation on DeepMIMO v4 in which a rotation applied
to a ray-traced array is recovered below the beam width.

## Layout

- `src/v2v/` — `data.py` (scenario index loader), `geo.py` (causal geometry and the codebook map), `track.py`
  (online methods, `step` / `feedback` protocol), `stream.py` (streaming evaluator and challenge metrics),
  `learned.py` (GPS-sequence network and its online variants)
- `scripts/` — `e0_challenge_baseline.py` (reproduces the 2023 challenge baseline), `e1*_*.py` (offset origin,
  transfer, sensor-lag checks), `e2_tracker.py`, `e3_learned.py`, `e4_known_truth.py`, `e5_continual.py`;
  `unpack_inner.py` (reads the dataset's split inner archives without concatenating them)
- `results/` — aggregated JSON outputs; `figures/` — figures at 600 dpi and as PDF; `tests/` — every number
  quoted in the paper is read from `results/` by a test

## Data

DeepSense 6G scenarios 36-39 (https://deepsense6g.net), licence as stated by the dataset; not redistributed.
Set the extracted root in `src/v2v/data.py` (`ROOT`). The scenario pickle files (`scenarioNN.p`) carry the
64-beam power vectors; `scripts/unpack_inner.py NN --modalities gps pwr` unpacks the GPS and power archives.
The known-truth arm uses DeepMIMO v4 (`pip install DeepMIMO`, scenario `city_3_houston_28`).

## Reproduce

```bash
pip install -r requirements.txt
python scripts/e0_challenge_baseline.py
python scripts/e1d_final.py
python scripts/e2_tracker.py
python scripts/e4_known_truth.py
python scripts/e5_continual.py
python -m pytest tests -q
python src/figures.py
```

## Citation

See `CITATION.cff`. Cite DeepSense 6G (Alkhateeb et al., IEEE Communications Magazine, 2023), the
DeepSense-V2V dataset paper (Morais et al., IEEE Transactions on Vehicular Technology, 2025) and DeepMIMO
(Alkhateeb, ITA 2019) for the data.

## Licence

MIT.
