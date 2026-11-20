"""M3: the REAL connectome flies the tank.

Full closed loop with the 138,639-neuron FlyWire v783 LIF brain: screen ->
real R1-6 photoreceptors -> whole-brain spiking -> real descending neurons ->
levers -> Battlezone. Supports both the frozen-circuit reflex presets
(`pilot`, `active`) and a trained neural readout (`--readout-ckpt`) in either
wired or physical embodied lever mode.
"""

import argparse
from contextlib import ExitStack
import math
from pathlib import Path
import sys
import time

import numpy as np
from scipy import sparse

from noflyzone import dashboard
from noflyzone.brain_backend import BACKENDS, describe, make_brain
from noflyzone.brain_connectome import INJECTION_MAPS, READOUT_MAPS, BrainView, ConnectomeBrain
from noflyzone.brain_lif import ABLATIONS
from noflyzone.loop import PilotLoop
from noflyzone.plasticity import EDGE_SETS, MBPlasticity
from noflyzone.provenance import dump_json, git_state, savez

DATASET_LABELS = {"783": "FlyWire v783 (female)", "malecns": "MaleCNS v1.0 (male)"}
OUT = Path(__file__).resolve().parent.parent / "out"

PILOT_PRESET = {"injection": "v4", "readout": "pool4", "zoom": 2.0,
                "eye_offset": 0.15, "saccade_hold": 3, "startle_refractory": 15,
                "steer_mode": "li", "aminergic": "zero", "sensory": "external_only"}

ACTIVE_PRESET = {"injection": "v11", "readout": "pool4", "zoom": 1.75,
                 "eye_offset": 0.21, "center_y": 0.48, "eye_fovy": 54.0,
                 "converge_eyes": True, "suppress_turn_transients": True,
                 "saccade_hold": 1, "startle_ticks": 5, "startle_refractory": 18,
                 "startle_thresh": 0.80, "steer_mode": "li", "steer_min_hz": 8.0,
                 "sharp_steer_hz": 6.0, "search_steer": 1.0, "search_delay": 2,
                 "patrol_sweep": 6, "patrol_stride": 4, "fovea_li": 0.18,
                 "pip_lock_drive": 0.50, "no_off_axis_fire": True,
                 "direct_steer": True, "aminergic": "zero", "sensory": "external_only"}


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=["reference", "pilot", "active"], default="reference",
                    help="pilot: historical Sept 25 reflex; active: binocular active-foraging reflex")
    ap.add_argument("--steps", type=int, default=90,
                    help="game steps (15 per second of game time)")
    ap.add_argument("--video", action="store_true", help="write out/<video-name>.mp4")
    ap.add_argument("--backend", choices=BACKENDS, default="auto",
                    help="brain backend: cpu (sparse SciPy) or cuda (Triton)")
    ap.add_argument("--injection", choices=list(INJECTION_MAPS), default="v0",
                    help="named injection map from brain_connectome.INJECTION_MAPS")
    ap.add_argument("--readout", choices=list(READOUT_MAPS), default="v0",
                    help="named readout map from brain_connectome.READOUT_MAPS")
    ap.add_argument("--calibration", default=None,
                    help="dn.CALIBRATION_MAPS key (default: same as --readout)")
    ap.add_argument("--zoom", type=float, default=1.0,
                    help="retina zoom: lattice over the central 1/zoom of the eye")
    ap.add_argument("--eye-offset", type=float, default=0.0,
                    help="shift each eye's retina window sideways by this image fraction")
    ap.add_argument("--center-y", type=float, default=0.5,
                    help="vertical centre fraction of the retinal lattice on the eye image")
    ap.add_argument("--eye-fovy", type=float, default=None,
                    help="vertical field of view (degrees) of the cockpit eye cameras")
    ap.add_argument("--converge-eyes", action="store_true",
                    help="converge both eye cameras on the CRT centre (eliminates central blind spot)")
    ap.add_argument("--suppress-turn-transients", action="store_true",
                    help="efference-copy gate suppressing mountain transients on discrete tank turns")
    ap.add_argument("--idle-scan", action="store_true",
                    help="scripted forward-right idle scan (off by default)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup", action="store_true",
                    help="diversify game start with a seeded 75-150 tick warm-up before scoring")
    ap.add_argument("--dataset", default="783", help="brain_lif.DATASETS key: 783 (FlyWire) or malecns")
    ap.add_argument("--ablate", choices=list(ABLATIONS), default="none",
                    help="wiring control: rewired (whole-brain target shuffle), "
                         "rewired_ipsi (targets shuffled within each hemisphere), silenced")
    ap.add_argument("--ablate-seed", type=int, default=0, help="RNG seed of the rewiring")
    ap.add_argument("--weight-scale", type=float, default=1.0,
                    help="multiply every synaptic weight (gain-matched controls)")
    ap.add_argument("--sensory", choices=["free", "external_only"], default="free",
                    help="external_only: primary sensory neurons fire only from injected input")
    ap.add_argument("--signs", choices=["predicted", "known_first"], default="predicted")
    ap.add_argument("--aminergic", choices=["excitatory", "zero"], default="excitatory",
                    help="zero: dopamine/octopamine/serotonin carry no fast weight")
    ap.add_argument("--saccade-hold", type=int, default=0,
                    help="ticks of visual suppression after each movement")
    ap.add_argument("--startle-ticks", type=int, default=8,
                    help="ticks of giant-fibre evasive reverse on a DNp01 startle")
    ap.add_argument("--startle-refractory", type=int, default=0,
                    help="ticks after a startle during which loom is ignored")
    ap.add_argument("--startle-thresh", type=float, default=0.5,
                    help="normalised DNp01 drive threshold for giant-fibre startle")
    ap.add_argument("--steer-mode", choices=["diff", "li"], default="diff",
                    help="diff: normalised-drive difference; li: lateralisation index")
    ap.add_argument("--steer-min-hz", type=float, default=8.0,
                    help="minimum combined steer-pool Hz for LI steering")
    ap.add_argument("--sharp-steer-hz", type=float, default=0.0,
                    help="lower combined Hz gate when target is strongly unilateral (|LI| >= 0.60)")
    ap.add_argument("--search-steer", type=float, default=0.0,
                    help="active saccadic search turn magnitude when no target is visible")
    ap.add_argument("--search-delay", type=int, default=2,
                    help="quiet look ticks before active search initiates")
    ap.add_argument("--patrol-sweep", type=int, default=0,
                    help="saccades per alternating left/right patrol sweep (0 = continuous)")
    ap.add_argument("--patrol-stride", type=int, default=0,
                    help="forward optic-flow stride ticks after each patrol sweep")
    ap.add_argument("--fovea-li", type=float, default=0.0,
                    help="binocular fovea LI deadband inside which the pilot holds aim and fires")
    ap.add_argument("--pip-lock-drive", type=float, default=0.75,
                    help="normalised pIP10 drive for binocular crosshair lock in active mode")
    ap.add_argument("--no-off-axis-fire", action="store_true",
                    help="hold fire outside the fovea deadband so missed shots do not trigger Battlezone's 29-tick reload lock-out")
    ap.add_argument("--direct-steer", action="store_true",
                    help="actuate intended turns without low-pass lag and hold fixation only after turns")
    ap.add_argument("--readout-ckpt", default=None,
                    help="trained readout checkpoint (.pt from train_readout.py); drives wired or embodied controls")
    ap.add_argument("--readout-sample", action="store_true",
                    help="sample from trained readout distribution instead of greedy argmax")
    ap.add_argument("--trace", default=None,
                    help="write per-tick DN rates/actions to out/<name>.json")
    ap.add_argument("--learned-weights", default=None,
                    help="out/plasticity_<tag>_weights.npz from train_plasticity.py: install "
                         "those synaptic weights (rule off) and label the footer LEARNED WEIGHTS")
    ap.add_argument("--learn", default=None, choices=list(EDGE_SETS),
                    help="run the reward-modulated rule live on this edge set while the fly "
                         "plays (CUDA only): kills pulse PAM, hits pulse PPL1; footer shows drift")
    ap.add_argument("--learn-eta", type=float, default=2e-5)
    ap.add_argument("--learn-seed", type=int, default=0, help="seed of a shuffled edge set")
    ap.add_argument("--video-name", default="m3_demo", help="out/<name>.mp4 with --video")
    ap.add_argument("--live", action="store_true",
                    help="open a window showing the dashboard as the fly plays (tkinter; needs a display); "
                         "close it or press q to stop")
    ap.add_argument("--realtime", action="store_true",
                    help="pace the simulation at game speed when computation is fast enough")
    ap.add_argument("--quiet", action="store_true", help="omit per-second neuron diagnostics")
    ap.add_argument("--no-eyes", action="store_true", help="leave the eyes panel off the dashboard")
    ap.add_argument("--pilot", choices=["wired", "embodied"], default="wired",
                    help="wired: descending rates set the levers directly; embodied: they drive the "
                         "forelegs, the physical levers move by contact, the action reads the levers")
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument("--preset", default="reference")
    preset, _ = selector.parse_known_args(argv)
    if preset.preset == "pilot":
        ap.set_defaults(**PILOT_PRESET)
    elif preset.preset == "active":
        ap.set_defaults(**ACTIVE_PRESET)
    args = ap.parse_args(argv)
    if args.steps < 1 or args.seed < 0 or not math.isfinite(args.zoom) or args.zoom <= 0:
        ap.error("steps and zoom must be positive; seed must be nonnegative")
    if args.learn and args.learned_weights:
        ap.error("choose either live learning or a frozen learned checkpoint")
    if args.readout_ckpt and not Path(args.readout_ckpt).is_file():
        ap.error(f"readout checkpoint not found: {args.readout_ckpt}")
    for name, enabled, suffix in ((args.video_name, args.video, ".mp4"),
                                   (args.trace, bool(args.trace), ".json")):
        if enabled:
            if not name or Path(name).name != name:
                ap.error("output names must be filenames, without directories")
            if (OUT / f"{name}{suffix}").exists():
                ap.error(f"output exists: {OUT / f'{name}{suffix}'}; choose a new name")
    return args


def _build_readout_policy(ckpt_path: str, core, brain_cb: ConnectomeBrain, loop: PilotLoop, sample: bool):
    """Construct a PilotLoop policy_fn from a saved train_readout.py checkpoint."""
    import torch
    scripts_dir = str(Path(__file__).resolve().parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from train_readout import ActorCritic, RecurrentActorCritic, build_action_codec, motor_efference

    ck = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    dev = torch.device("cuda" if torch.cuda.is_available() and hasattr(core, "_device") else "cpu")
    arch = ck.get("arch", "mlp")
    hidden = int(ck.get("hidden", 256))
    feat_mode = ck["features"]
    mean = ck["norm_mean"]
    var = ck["norm_var"]
    var_floor = float(ck.get("norm_var_floor", 0.0))
    clip = ck.get("norm_clip", None)

    ann = core.annotations().drop_duplicates("root_id").set_index("root_id").reindex(core.root_ids)
    ct = ann["cell_type"].fillna("").astype(str).to_numpy()
    side = ann["side"].fillna("").astype(str).to_numpy()
    sc = ann["super_class"].fillna("").astype(str).to_numpy()
    keep = np.isin(sc, ("descending", "visual_projection", "visual_centrifugal"))

    if feat_mode in ("brain", "brain_bilateral"):
        ftypes = ck["feature_types"]
        rows, cols, vals = [], [], []
        for r, tok in enumerate(ftypes):
            if feat_mode == "brain_bilateral" and "/" in tok:
                t_name, s_name = tok.rsplit("/", 1)
                idx = np.flatnonzero(keep & (ct == t_name) & (side == s_name))
            elif feat_mode == "brain_bilateral":
                idx = np.flatnonzero(keep & (ct == tok) & (side == ""))
            else:
                idx = np.flatnonzero(keep & (ct == tok))
            if idx.size:
                rows += [r] * idx.size
                cols += idx.tolist()
                vals += [1.0 / idx.size] * idx.size
        pool = sparse.csr_matrix((vals, (rows, cols)), shape=(len(ftypes), core.n), dtype=np.float32)
        n_features = len(ftypes)
        retina_pool = None
    else:
        grid = (12, 9)
        xy = loop.retina.centers_xy
        gx = np.clip(((xy[:, 0] + 1) / 2 * grid[0]).astype(int), 0, grid[0] - 1)
        gy = np.clip(((xy[:, 1] + 1) / 2 * grid[1]).astype(int), 0, grid[1] - 1)
        cell = gy * grid[0] + gx
        n_cells = grid[0] * grid[1]
        cnts = np.bincount(cell, minlength=n_cells).astype(np.float32)
        retina_pool = sparse.csr_matrix((1.0 / cnts[cell], (cell, np.arange(len(cell)))),
                                        shape=(n_cells, len(cell)), dtype=np.float32)
        pool = None
        n_features = 4 * n_cells * 2

    a_to_f, _ = build_action_codec(loop.game.action_meanings)

    if arch == "mlp":
        net = ActorCritic(n_features, ck["n_actions"], hidden=hidden).to(dev)
        net.load_state_dict(ck["net"])
        net.eval()
        h_state = None
    else:
        net = RecurrentActorCritic(n_features, n_motor=4, hidden=hidden).to(dev)
        net.load_state_dict(ck["net"])
        net.eval()
        h_state = torch.zeros((1, hidden), dtype=torch.float32, device=dev)

    prev_fact = np.array([[1, 1, 0]], dtype=np.int64)
    since_fire = np.array([20], dtype=np.int64)
    prev_ret_lum = {"left": None, "right": None}

    def policy_fn(sensed: dict, dn_rates: dict, episode_end: bool) -> tuple[float, float, bool]:
        nonlocal prev_fact, since_fire
        if episode_end:
            if h_state is not None:
                h_state.zero_()
            prev_fact[:] = (1, 1, 0)
            since_fire[:] = 20
            prev_ret_lum["left"] = prev_ret_lum["right"] = None

        if pool is not None:
            hz = (pool @ brain_cb.window_counts.astype(np.float32)) / loop.dt
            obs = np.log1p(hz).astype(np.float32)[None, :]
        else:
            drives = sensed["drives"]
            parts = []
            for s in ("left", "right"):
                lum = drives[s][:, 0]
                dl = (lum - prev_ret_lum[s]) if prev_ret_lum[s] is not None else np.zeros_like(lum)
                prev_ret_lum[s] = lum
                full_d = np.stack([dl, dl], axis=1)
                parts.append(retina_pool @ drives[s])
                parts.append(retina_pool @ full_d)
            obs = np.concatenate([p.ravel() for p in parts]).astype(np.float32)[None, :]

        z = (obs - mean) / np.sqrt(np.maximum(var, var_floor) + 1e-8)
        if clip is not None:
            z = np.clip(z, -float(clip), float(clip))
        x_t = torch.from_numpy(z.astype(np.float32)).to(dev)

        with torch.no_grad():
            if arch == "mlp":
                logits, _ = net(x_t)
                a = int((torch.distributions.Categorical(logits=logits).sample() if sample
                         else logits.argmax(-1)).item())
                th, st, fi = a_to_f[a]
            else:
                m_t = torch.from_numpy(motor_efference(prev_fact, since_fire)).to(dev)
                (l_th, l_st, l_fi), _, h_next = net.step(x_t, m_t, h_state)
                h_state.copy_(h_next)
                if sample:
                    th = int(torch.distributions.Categorical(logits=l_th).sample().item())
                    st = int(torch.distributions.Categorical(logits=l_st).sample().item())
                    fi = int(torch.distributions.Categorical(logits=l_fi).sample().item())
                else:
                    th = int(l_th.argmax(-1).item())
                    st = int(l_st.argmax(-1).item())
                    fi = int(l_fi.argmax(-1).item())

        prev_fact[:] = (th, st, fi)
        since_fire[:] = 0 if fi == 1 else (since_fire + 1)
        steer_val = float(st - 1)     # -1 LEFT, 0 NOOP, +1 RIGHT
        throttle_val = float(th - 1)  # -1 DOWN, 0 NOOP, +1 UP
        return steer_val, throttle_val, bool(fi == 1)

    return policy_fn, ck


def main(argv=None) -> None:
    args = parse_args(argv)
    with ExitStack() as resources:
        run(args, resources)


def run(args, resources) -> None:
    ckpt_meta = None
    if args.readout_ckpt:
        import torch
        ckpt_meta = torch.load(args.readout_ckpt, weights_only=False, map_location="cpu")
        ck_args = ckpt_meta.get("args", {})
        legacy_eyes = ck_args.get("legacy_eyes", "arch" not in ckpt_meta)
        args.injection = ck_args.get("injection", args.injection)
        args.zoom = float(ck_args.get("zoom", args.zoom))
        args.eye_offset = float(ck_args.get("eye_offset", args.eye_offset))
        args.center_y = float(ck_args.get("center_y", 0.5 if legacy_eyes else 0.48))
        if not legacy_eyes:
            args.eye_fovy = float(ck_args.get("eye_fovy", 54.0))
            args.converge_eyes = True
        args.aminergic = "zero"
        args.sensory = "external_only"
        args.suppress_turn_transients = bool(ck_args.get("suppress_turn_transients", False))

    calibration = args.calibration or args.readout

    print("building loop + connectome brain...", flush=True)
    t0 = time.time()
    loop_kw = dict(seed=args.seed, retina_zoom=args.zoom,
                   retina_center=(0.5, args.center_y), eye_offset=args.eye_offset,
                   pilot=args.pilot)
    if args.eye_fovy is not None:
        loop_kw["eye_fovy"] = args.eye_fovy
    if args.converge_eyes:
        loop_kw["converge_eyes"] = True
    if args.suppress_turn_transients:
        loop_kw["suppress_turn_transients"] = True
    loop = PilotLoop(**loop_kw)
    resources.callback(loop.close)
    if args.warmup:
        from noflyzone.rl_env import episode_seed
        loop.reset(seed=episode_seed(args.seed, 0, 0), warmup_seed=episode_seed(args.seed, 0, 1))
    core = make_brain(args.backend, version=args.dataset, seed=args.seed,
                      ablate=args.ablate, ablate_seed=args.ablate_seed,
                      sensory_input=args.sensory, signs=args.signs,
                      aminergic=args.aminergic, weight_scale=args.weight_scale)
    extra_sites = None
    if args.learn:
        ann = core.annotations().drop_duplicates("root_id").set_index("root_id").reindex(core.root_ids)
        ct = ann["cell_type"].fillna("").astype(str).to_numpy()
        extra_sites = {"pam": np.flatnonzero(np.char.startswith(ct.astype(str), "PAM")),
                       "ppl1": np.flatnonzero(np.char.startswith(ct.astype(str), "PPL1"))}
    brain = ConnectomeBrain(retina=loop.retina, brain=core,
                            injection=args.injection, readout=args.readout,
                            extra_sites=extra_sites)
    loop.brain = brain
    learned = None
    if args.learned_weights:
        z = np.load(args.learned_weights)
        if not np.allclose(core.W.data[z["pos"]], z["w0"]):
            raise SystemExit("learned weights were saved from a different brain "
                             "(dataset, wiring or corrections differ)")
        core.set_edge_weights(z["pos"], z["w"])
        learned = (f"LEARNED WEIGHTS {Path(args.learned_weights).stem.replace('plasticity_', '').replace('_weights', '')}"
                   f" ({int((z['w'] != z['w0']).sum()):,} of {z['w'].size:,} synapses changed)")
        print(learned)
    plast = None
    if args.learn:
        if not hasattr(core, "set_edge_weights"):
            raise SystemExit("--learn needs the CUDA backend")
        plast = MBPlasticity(core, dt=loop.dt, eta=args.learn_eta, edge_set=args.learn,
                             seed=args.learn_seed)
        print(f"plasticity on: {plast.n_edges:,} edges ({plast.edge_set}, {plast.eligibility}, "
              f"{plast.scale}); dopamine from PAM/PPL1 teaching pulses, applied as a global "
              f"modulator (ENGINEERED)")
    ctrl_kw = dict(calibration=calibration, saccade_hold=args.saccade_hold,
                   startle_refractory=args.startle_refractory, steer_mode=args.steer_mode)
    if args.startle_ticks != 8:
        ctrl_kw["startle_ticks"] = args.startle_ticks
    if args.startle_thresh != 0.5:
        ctrl_kw["startle_thresh"] = args.startle_thresh
    if args.steer_min_hz != 8.0:
        ctrl_kw["steer_min_hz"] = args.steer_min_hz
    if args.sharp_steer_hz != 0.0:
        ctrl_kw["sharp_steer_hz"] = args.sharp_steer_hz
    if args.search_steer != 0.0:
        ctrl_kw["search_steer"] = args.search_steer
        ctrl_kw["search_delay"] = args.search_delay
    if args.patrol_sweep != 0:
        ctrl_kw["patrol_sweep"] = args.patrol_sweep
    if args.patrol_stride != 0:
        ctrl_kw["patrol_stride"] = args.patrol_stride
    if args.fovea_li != 0.0:
        ctrl_kw["fovea_li"] = args.fovea_li
        ctrl_kw["pip_lock_drive"] = args.pip_lock_drive
    if args.no_off_axis_fire:
        ctrl_kw["off_axis_fire"] = False
    if args.direct_steer:
        ctrl_kw["direct_steer"] = True
    loop.controls = type(loop.controls)(loop.game.action_meanings, **ctrl_kw)

    readout_tag = None
    if args.readout_ckpt:
        policy_fn, ck_loaded = _build_readout_policy(
            args.readout_ckpt, core, brain, loop, sample=args.readout_sample
        )
        loop.policy_fn = policy_fn
        readout_tag = f"{ck_loaded.get('arch', 'mlp').upper()}:{Path(args.readout_ckpt).stem}"

    if not args.quiet:
        print(f"brain backend: {describe(brain.brain)}")
    if args.idle_scan:
        loop.controls.idle_throttle = 0.35
        loop.controls.idle_steer = 0.5
    mode = (f"injection {args.injection} | readout {readout_tag or args.readout} | "
            f"calibration {calibration} | zoom {args.zoom:g}"
            + (f" eyes +-{args.eye_offset:g}" if args.eye_offset else "")
            + (f" fovy {args.eye_fovy:g} converged" if args.converge_eyes and args.eye_fovy else "")
            + " | "
            + (f"TRAINED READOUT ({readout_tag})" if readout_tag else
               ("IDLE SCAN (scripted)" if args.idle_scan else "CONNECTOME STEERING"))
            + (f" | search {args.search_steer:g}" if args.search_steer and not readout_tag else "")
            + (f" | saccade hold {args.saccade_hold}" if args.saccade_hold and not readout_tag else "")
            + (f" | startle refractory {args.startle_refractory}"
               if args.startle_refractory and not readout_tag else "")
            + (" | steer LI" if args.steer_mode == "li" and not readout_tag else "")
            + (f" | WIRING {args.ablate.upper()}"
               + (f" seed {args.ablate_seed}" if args.ablate_seed else "")
               if args.ablate != "none" else "")
            + (f" | weights x{args.weight_scale:g}" if args.weight_scale != 1.0 else "")
            + (f" | sensory {args.sensory}" if args.sensory != "free" else "")
            + (f" | signs {args.signs}" if args.signs != "predicted" else "")
            + (f" | aminergic {args.aminergic}" if args.aminergic != "excitatory" else "")
            + (f" | {learned}" if learned else "")
            + (" | EMBODIED: forelegs push physical levers (IK targets, claw adhesion); action reads the levers"
               if args.pilot == "embodied" else ""))
    g = git_state()
    if g["short"]:
        mode += f" | {g['short']}{'*' if g['dirty'] else ''}"
    if not args.quiet:
        print(mode)
    view = BrainView(brain.brain)
    diag_types = ["Mi1", "T4a", "T4b", "T5a", "LC4", "LPLC2", "LC10a"]
    diag = {t: brain.brain.indices_for_type(t) for t in diag_types}
    if args.quiet:
        print(f"Ready in {time.time() - t0:.0f} s: {core.n:,} neurons, {args.pilot} controls.", flush=True)
    else:
        print(f"ready in {time.time() - t0:.0f} s | {brain.describe()}", flush=True)

    live = None
    if args.live:
        from noflyzone.liveview import LiveView
        live = LiveView(title="NO-FLY-ZONE")
        resources.callback(live.close)
    writer = None
    if args.video:
        import imageio.v2 as imageio
        OUT.mkdir(exist_ok=True)
        writer = imageio.get_writer(OUT / f"{args.video_name}.mp4",
                                    fps=round(1 / loop.dt),
                                    codec="libx264", quality=8)
        resources.callback(writer.close)

    t0 = time.time()
    next_frame = time.monotonic()
    trace = []
    names = loop.game.action_meanings
    peak_spiking = 0
    learn_label = ""
    prev_rgb = None
    try:
        for i in range(args.steps):
            if plast is not None:
                for name, hz in plast.teaching_rates().items():
                    brain.set_extra_rate(name, hz)
            state = loop.step()
            spiking = int((brain.window_counts > 0).sum())
            peak_spiking = max(peak_spiking, spiking)
            if plast is not None:
                d = plast.step(brain.window_counts[None, :], kills=np.array([state["reward"] > 0]),
                               hits=np.array([bool(state["life_lost"])]))
                learn_label = (f" | PLASTICITY ON {plast.edge_set}: {d['frac_changed'] * 100:.0f}% of "
                               f"{plast.n_edges:,} synapses moved, drift {d['drift'] * 100:.1f}% "
                               f"(dopamine ENGINEERED)")
            peek_act = (loop.controls.to_action(commit=False)
                        if "commit" in loop.controls.to_action.__code__.co_varnames
                        else loop.controls.to_action())
            trace.append({"tick": i, "dn": state["dn"], "drive": dict(loop.controls.drive),
                          "suppressed": bool(loop.controls.suppressed),
                          "steer": loop.controls.steer, "throttle": loop.controls.throttle,
                          "levers": list(loop.controls.measured) if loop.controls.measured else None,
                          "fire": bool(loop.controls.fire),
                          "action": names[peek_act],
                          "reward": float(state["reward"]),
                          "life_lost": bool(state["life_lost"]),
                          "episode_end": bool(state["episode_end"]),
                          "spiking": spiking,
                          "retina_lr": float(state["drives"]["right"][:, 0].mean()
                                             - state["drives"]["left"][:, 0].mean())})
            if writer is not None or live is not None:
                eyes = None if args.no_eyes else dashboard.eyes_panel(
                    loop.retina, loop.retina_right, state["rgb"], prev_rgb)
                frame_img = dashboard.compose(
                    loop.cockpit.render("beauty"), loop.game.frame, loop.controls,
                    state["dn"], loop.total_reward, (i + 1) * loop.dt,
                    brain=view.render(brain.window_counts),
                    brain_label=f"{loop.brain.brain.n:,} neurons | {spiking:,} spiking",
                    footer=display_label(args, mode, learn_label, ckpt_meta),
                    eyes=eyes)
                if writer is not None:
                    writer.append_data(frame_img)
                if live is not None and not live.show(frame_img):
                    print("live view closed", flush=True)
                    break
            prev_rgb = state["rgb"]
            if args.realtime:
                next_frame = max(next_frame + loop.dt, time.monotonic())
                time.sleep(max(0.0, next_frame - time.monotonic()))
            if not args.quiet and (i + 1) % 15 == 0:
                dn_str = "  ".join(f"{k} {v:5.1f}" for k, v in state["dn"].items())
                cascade = "  ".join(
                    f"{t} {brain.window_counts[ix].mean() / loop.dt:4.0f}"
                    for t, ix in diag.items())
                print(f"t={(i + 1) * loop.dt:5.1f}s  spiking {spiking:6,}  {dn_str}",
                      flush=True)
                print(f"        cascade Hz: {cascade}", flush=True)
            elif args.quiet and live is None and (i + 1) % 150 == 0:
                print(f"Recorded {(i + 1) * loop.dt:.0f}/{args.steps * loop.dt:g} game seconds; "
                      f"score {loop.total_reward:.0f}", flush=True)
    except KeyboardInterrupt:
        print("\nStopped; saving the completed recording and trace.", flush=True)

    wall = time.time() - t0
    if writer is not None:
        print(f"video: {OUT / f'{args.video_name}.mp4'}")
    completed = len(trace)
    game_t = completed * loop.dt
    speed = game_t / max(wall, 1e-6)
    print(f"\n{completed}/{args.steps} steps = {game_t:.1f} s game time in {wall:.1f} s wall "
          f"({speed:.1f}x real time)")
    print("latency:")
    print(loop.latency_report())
    if trace:
        assert peak_spiking > 0, "brain silent for the whole run — injection broken"
    if args.pilot == "embodied" and trace:
        lv = np.array([t["levers"] for t in trace if t["levers"] is not None])
        print(f"embodied: levers reached left {lv[:, 0].min():+.2f}..{lv[:, 0].max():+.2f}, "
              f"right {lv[:, 1].min():+.2f}..{lv[:, 1].max():+.2f} of the fly's push range; "
              f"physics {1000 * np.mean(loop.timings['physics']):.0f} ms per tick")
    if plast is not None:
        OUT.mkdir(exist_ok=True)
        wpath = OUT / f"m3_learned_{plast.edge_set}_s{args.seed}_weights.npz"
        savez(wpath, args=args, brain=core, pos=plast.pos, w0=plast.w0, w=plast.w, pre=plast.pre, post=plast.post)
        print(f"plasticity: {learn_label.strip(' |')}; weights -> {wpath}")
    if args.trace:
        OUT.mkdir(exist_ok=True)
        dump_json(OUT / f"{args.trace}.json", {"mode": mode, "steps": completed,
                  "requested_steps": args.steps, "stopped_early": completed < args.steps, "trace": trace},
                  args=args, brain=core)
        from collections import Counter
        acts = Counter(t["action"] for t in trace)
        print(f"actions: {dict(acts.most_common())}")
        print(f"total reward {loop.total_reward:.0f}, lives lost "
              f"{sum(t['life_lost'] for t in trace)}, fire ticks "
              f"{sum(t['fire'] for t in trace)}; trace -> out/{args.trace}.json")

    if args.quiet or not trace:
        return
    ann = loop.brain.brain.annotations().set_index("root_id")
    ann = ann[~ann.index.duplicated()]
    import pandas as pd
    classes = ann.reindex(loop.brain.brain.root_ids)["super_class"]
    active = pd.Series(brain.window_counts > 0)
    print("\nspiking neurons by super_class (final tick):")
    print(classes[active.to_numpy()].value_counts().head(10).to_string())
    print("PASS: the connectome flew the tank")


def display_label(args, mode, learn_label, ckpt_meta=None):
    if args.readout_ckpt:
        arch = (ckpt_meta or {}).get("arch", "mlp").upper()
        weights = f"WIRING: {args.ablate.upper()}" if args.ablate != "none" else "MEASURED WEIGHTS"
        controls = "IK + physical levers" if args.pilot == "embodied" else "frozen connectome features"
        return f"TRAINED READOUT ({arch}) | {weights} | {controls}"
    if args.preset not in ("pilot", "active"):
        return f"NO-FLY-ZONE // {DATASET_LABELS.get(args.dataset, args.dataset)} // {mode}{learn_label}"
    if args.learn:
        weights = "PLASTICITY ON (ENGINEERED)"
    elif args.learned_weights:
        weights = "LEARNED WEIGHTS"
    elif args.ablate != "none":
        weights = f"WIRING: {args.ablate.upper()}"
    else:
        weights = "MEASURED WEIGHTS"
    if args.pilot == "embodied":
        controls = "IK + physical levers"
    elif args.preset == "active":
        controls = "binocular fovea + foraging reflex"
    else:
        controls = "engineered visual + motor interfaces"
    return f"REFLEX PILOT | {weights} | {controls}"


if __name__ == "__main__":
    main()
