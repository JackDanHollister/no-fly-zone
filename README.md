# NO-FLY-ZONE

A real fruit-fly connectome fighting for its life in a tank it was never
licensed to drive.

The complete Drosophila brain wiring diagram (FlyWire / MaleCNS, ~140-167k
neurons) runs as a spiking simulation and is seated as the **pilot** inside a
retro-futuristic 3D cockpit. The cockpit's screen shows Atari **Battlezone**;
the fly's compound eyes watch that screen; its descending motor neurons work
the cockpit controls; the controls drive the game. Fly → cockpit → tank →
robot war.

## Architecture

```
[ALE BattleZone-v5] ←— joystick actions —— [cockpit control state]
        |                                          ↑
    RGB frames                            control physics (MuJoCo)
        ↓                                          ↑
[MuJoCo cockpit: CRT screen geom ← frame texture upload]
        ↓  eye cameras (offscreen render)
[retina: hex ommatidia binning → photoreceptor drive]
        ↓
[LIF spiking brain over the real connectome (Shiu et al. 2024 params)]
        ↓  descending neurons (DNa02 steer, pIP10 fire, DNp01 startle, ...)
[motor mapping → levers / pedals / trigger]
```

Two pilot modes, same brain and eyes:
- **wired** — descending-neuron rates drive animated cockpit controls directly
  (the working baseline)
- **embodied** — the TuragaLab `flybody` model physically seated in the
  cockpit, motor neurons → leg actuators → contact forces on the levers
  (the over-the-top goal)

## Quickstart (any machine)

```bash
git clone https://github.com/JackDanHollister/no-fly-zone
cd no-fly-zone
python3 -m venv .venv                 # python >= 3.11
.venv/bin/pip install -e .            # ale-py bundles the Atari ROMs legally
.venv/bin/python scripts/fetch_assets.py   # fly body + connectome data, ~570 MB
.venv/bin/python scripts/smoke_battlezone.py
.venv/bin/python scripts/smoke_cockpit_screen.py
.venv/bin/python scripts/run_m2.py --video          # reflex-stub pilot, fast
.venv/bin/python scripts/validate_brain.py          # LIF core vs published results
.venv/bin/python scripts/run_m3.py --steps 150 --video  # the real connectome plays
```

Videos and renders land in `out/`. macOS-only gotcha: if `import noflyzone`
fails after installing, see the `.pth` note in `docs/architecture.md`.

## Machines

- **Dev**: Apple-silicon Mac — everything runs here (MuJoCo has native arm64
  wheels; ALE bundles the Atari ROMs legally; brain sim is sparse NumPy /
  PyTorch-MPS).
- **Final**: workstation (RTX 4090, 186 GB RAM) — CUDA brain backends and
  high-res rendering. All components are pure Python; only backends swap.

macOS note: keep the step→upload-texture→render loop single-threaded (GL must
stay on the main thread).

## Layout

- `src/noflyzone/` — package (game bridge, cockpit scene, retina, brain)
- `scripts/` — runnable smoke tests and demos
- `docs/architecture.md` — full design notes and decisions
- `data/` — connectome downloads (gitignored)

## Data & reference implementations

- MaleCNS v1.0 (Janelia/Google, CC-BY): https://male-cns.janelia.org
- FlyWire v783 dump (CC-BY, no login): https://zenodo.org/records/10676866
- LIF parameters: Shiu et al., Nature 634:210 (2024) — github.com/philshiu/Drosophila_brain_model
- Eye sampling: flygym Retina (NeLy-EPFL), aimbug (slickdomi/aimbug, MIT),
  DOOMFLY (nftechie/doomfly, MIT)
- Body model: TuragaLab/flybody (MuJoCo)
