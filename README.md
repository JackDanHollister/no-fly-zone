# NO-FLY-ZONE

**A fruit-fly brain. A tank cockpit. An extremely questionable driving licence.**

![The fly piloting Battlezone in its cockpit with physical levers, live brain activity and both retinal mosaics](docs/media/demo.gif)

*Closed-loop physical-controls recording, seed 0, zero RL. The 138,639-neuron
FlyWire v783 connectome watches Battlezone on the cockpit CRT, pushes physical
levers with its forelegs, presses the fire button with its proboscis and scores
three kills in the first 11 seconds.*

I wanted to see what happens if you put a simulated fruit-fly brain in a MuJoCo
tank cockpit and wire it up to Atari Battlezone. It started as a bit of fun and
turned into a proper rabbit hole of compound-eye optics, GPU spiking kernels,
wiring controls and watching out for shortcuts.

The default pilot runs **138,639 neurons and ~15.1 million directed connections**
from FlyWire v783 through a leaky integrate-and-fire model on custom
Triton/CUDA kernels — about **2.5× to 5× real time** wired on an RTX 4090, and
about **real time** with the physical levers and proboscis running.

## What it does

- **Full visual loop:** Battlezone renders onto a CRT inside the cockpit, two
  converged eye cameras watch the screen, two 721-ommatidium hex retinas drive
  the brain, and five descending-neuron pools steer, move, fire and dodge.
- **Physical controls (`--embodied`):** a tethered flybody model pushes hinged
  levers with its front claws and presses a sprung fire button with its
  proboscis. Their measured positions pick the Atari action.
- **Live dashboard:** cockpit view, game screen, spiking brain cloud, motor
  bars and what each compound eye actually sampled on that tick.
- **A proper test bench:** whole-brain and within-hemisphere wiring shuffles,
  a second connectome (MaleCNS v1.0, 166,700 neurons), moving-target tracking,
  plasticity tests and optional learned readouts.

![Mid-kill frame on physical controls: enemy exploding on the crosshair and across both retinal mosaics](docs/media/dashboard_turn.png)

## How it went (the highlights)

This turned out to be way more interesting once I stopped just looking at score
and started checking *why* the fly was doing what it did:

- **The crosshair blind spot:** early recordings spent 787 of 900 ticks doing
  `NOOP`. Parallel eye cameras + outward offsets had left **zero ommatidia** on
  the aiming crosshair — enemies vanished right as the fly lined them up.
- **Catching an RL shortcut:** a trained PPO GRU readout hit `6,000` points on
  seed 0, but watching the video back it was barely hunting — just running a
  clockwise circle-strafe exploit (`445` `RIGHT` turns vs `3` `LEFT`). The
  smoking gun: the same GRU trained on a **randomly scrambled brain** beat the
  real brain (`2.31` kills / `+0.78` utility vs `1.69` / `+0.41`). I pulled RL
  off the default launcher so the fly actually has to fly on its own wiring.
- **Finding where the fly actually looks:** mapping the steering neurons showed
  FlyWire's `LC10a` tracking fovea sits on the **anterior rim** of each eye, not
  the centre — so the fly was slightly cross-eyed. Uncrossing the foveas
  (`zoom = 1.75, eye_offset = 0.21`) cut moving-target tracking error in half to
  **`8.36°` RMS (`77%` crosshair lock, `0.99` turn-sign accuracy)**, beating
  even within-hemisphere shuffles (`12.8–14.5°` RMS) for the first time.
- **Curing "T-Rex vision" + Atari's 29-tick gun reload:** frame-to-frame
  luminance made stopped enemies invisible, and missed off-axis shots lock out
  Battlezone's cannon for ~2 seconds. Wiring the retina's `R8` blue channel
  into FlyWire's `3,630` real `Tm20 / Tm5a / Tm5b` chromatic synapses (`v11`,
  which stay dead silent at `0.0 Hz` in shuffled brains) and holding fire until
  centred took the pure frozen connectome to its best results yet.
- **Synaptic plasticity (honest negative):** a 576-episode controlled plasticity
  campaign finished with every learned brain scoring lower than measured weights
  and its matched shuffled-edge controls, so I closed that branch under its
  stopping rule.

### Latest zero-RL benchmark results (measured wiring vs controls)

| Benchmark | Measured FlyWire v783 | Whole-brain shuffles (`rewired`) | Within-hemisphere shuffles (`rewired_ipsi`) | Silenced |
|---|---|---|---|---|
| **Moving-target tracking** (`8` trajectories) | **`8.36°` RMS** (`77%` lock) | `22.5–29.7°` (`20–26%`) | `12.8–14.5°` (`38–55%`) | `22.3°` (`17%`) |
| **Battlezone combat** (`10 × 100 s` episodes) | **`3.40` kills, `3.60` lives (`+1.60` util)** | `0.90–1.60` kills (`-0.35` to `-1.60`) | `1.40–2.00` kills (`-1.15` to `-1.25`) | `0.70` kills (`-1.40`, `0` shots) |

On seed 0 (`60 s`), the default pure-connectome pilot scores **`3,000` points**
wired (`159` `LEFT` / `158` `RIGHT`, `3` lives lost) and **`3,000` points with
`1` life lost** pushing the physical levers and proboscis.

## Run it

Requires Python 3.11+ (tested on 3.12). NVIDIA CUDA runs at multi-real-time;
the SciPy CPU backend also works on Linux and macOS, just slower.

```bash
git clone https://github.com/JackDanHollister/no-fly-zone.git
cd no-fly-zone
python3.12 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]" -c constraints.txt
.venv/bin/python scripts/fetch_assets.py   # ~570 MB flybody + FlyWire data (gitignored)
```

On Linux with an NVIDIA GPU, install the CUDA extras and launch:

```bash
.venv/bin/python -m pip install -e ".[cuda]" -c constraints.txt
./no-fly-zone                              # live dashboard on desktop, or records MP4 headless
```

Useful options:

```bash
./no-fly-zone --embodied                   # physical foreleg levers + proboscis fire button
./no-fly-zone record --seconds 100         # save an MP4 + JSON trace in out/ without a window
./no-fly-zone --controller learned         # try an optional trained GRU readout instead
./no-fly-zone --controller legacy          # run the earlier September 25 preset
./no-fly-zone doctor                       # check assets, renderer, Battlezone and FFmpeg
.venv/bin/python -m pytest                 # full 56-test validation suite
```

## What's real biology vs engineering

The **wiring graph and anatomical connection counts** come directly from the
measured FlyWire v783 (and MaleCNS v1.0) connectomes, simulated with Shiu et
al.'s point-neuron LIF model using transmitter signs and synapse counts (not
physiological synaptic measurements). The visual input maps, descending-neuron
readout pools, foraging/escape motor rules, and embodied leg IK servos are
engineered interfaces around that untouched brain — there's no simulated
ventral nerve cord, no consciousness claim, and no token or coin.

## Want to look closer?

- [JOURNEY.md](JOURNEY.md) — the full build story, including all the dead ends and plot twists.
- [docs/architecture.md](docs/architecture.md) — how the loop, GPU kernels, physical levers and injection maps work.
- [docs/research/](docs/research/) — detailed experiment write-ups:
  [active vision, PPO audit & R8 foveas](docs/research/active-vision-and-tracking-2026-11-17.md),
  [wiring controls](docs/research/wiring-controls-2026-09-18.md),
  [plasticity campaign](docs/research/learning-campaign-2026-09-25.md),
  [male retinotopy](docs/research/male-retinotopy-2026-09-20.md),
  [size tuning](docs/research/size-tuning-2026-09-20.md).
- [CHANGELOG.md](CHANGELOG.md) — dated session history.

## Credits and licence

Code is [MIT licensed](LICENSE). The connectomes, body model and neuron model
belong to their respective research teams (see [CITATION.cff](CITATION.cff)):

- [FlyWire](https://zenodo.org/records/10676866) & [flywire_annotations](https://github.com/flyconnectome/flywire_annotations) (female connectome)
- [Shiu et al. Drosophila brain model](https://github.com/philshiu/Drosophila_brain_model) (LIF reference & v783 connectivity)
- [MaleCNS v1.0](https://male-cns.janelia.org/) (male connectome, CC-BY)
- [Flybody / MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/flybody) (Apache-2.0)
- [MuJoCo](https://github.com/google-deepmind/mujoco) & [Arcade Learning Environment](https://github.com/Farama-Foundation/Arcade-Learning-Environment) (no Atari ROM is committed here)

Jack Dan Hollister
