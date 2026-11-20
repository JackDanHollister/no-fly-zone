"""Train a readout policy on the frozen connectome with PPO (MLP or Recurrent GRU).

The spiking connectome itself stays frozen. Either:
  - `mlp` (legacy 2-layer tanh MLP over single-tick features), or
  - `gru` (256-unit GRU with motor efference copy and factored 3x3x2 throttle/steer/fire
    heads over bilateral `(cell_type, side)` connectome features).

Compare `--brain connectome | rewired | rewired_ipsi | silenced`, plus
`--features retina` for the no-brain ceiling, under identical seeds and budgets.

Outputs:
  out/readout_<tag>.pt       latest checkpoint (policy + normaliser + provenance)
  out/readout_<tag>_best.pt  best rolling-utility checkpoint during RL phase
  out/readout_<tag>.jsonl    one JSON line per update with episode records and stats
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from noflyzone.brain_lif import ABLATIONS
from noflyzone.controls import CockpitControls
from noflyzone.provenance import attach
from noflyzone.rl_env import FlySquad

OUT = Path(__file__).resolve().parent.parent / "out"


def build_action_codec(action_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Exact bijection between ALE's 18 Battlezone actions and (throttle, steer, fire)
    in {0:DOWN, 1:NOOP, 2:UP} x {0:LEFT, 1:NOOP, 2:RIGHT} x {0:NOOP, 1:FIRE}."""
    a_to_f = np.zeros((len(action_names), 3), dtype=np.int64)
    f_to_a = np.zeros((3, 3, 2), dtype=np.int64)
    for idx, name in enumerate(action_names):
        th = 0 if "DOWN" in name else (2 if "UP" in name else 1)
        st = 0 if "LEFT" in name else (2 if "RIGHT" in name else 1)
        fi = 1 if "FIRE" in name else 0
        a_to_f[idx] = (th, st, fi)
        f_to_a[th, st, fi] = idx
    return a_to_f, f_to_a


def motor_efference(factored: np.ndarray, since_fire: np.ndarray) -> np.ndarray:
    """Internal corollary-discharge vector (B, 4) from the agent's own previous motor command."""
    th = (factored[:, 0] - 1).astype(np.float32)
    st = (factored[:, 1] - 1).astype(np.float32)
    fi = factored[:, 2].astype(np.float32)
    cd = np.minimum(1.0, since_fire.astype(np.float32) / 20.0)
    return np.stack([th, st, fi, cd], axis=-1)


class RunningNorm:
    def __init__(self, n: int, clip: float | None = 10.0, var_floor: float = 0.0):
        self.mean = np.zeros(n, np.float64)
        self.var = np.ones(n, np.float64)
        self.count = 1e-4
        self.clip = clip
        self.var_floor = float(var_floor)

    def update(self, x: np.ndarray) -> None:
        bm, bv, bc = x.mean(0), x.var(0), x.shape[0]
        d = bm - self.mean
        tot = self.count + bc
        self.mean += d * bc / tot
        self.var = (self.var * self.count + bv * bc + d ** 2 * self.count * bc / tot) / tot
        self.count = tot

    def __call__(self, x: np.ndarray) -> np.ndarray:
        denom = np.sqrt(np.maximum(self.var, self.var_floor) + 1e-8)
        z = (x - self.mean) / denom
        if self.clip is not None:
            z = np.clip(z, -self.clip, self.clip)
        return z.astype(np.float32)


class ActorCritic(nn.Module):
    """Legacy single-tick feedforward actor-critic (retained for historical checkpoints)."""

    def __init__(self, n_in: int, n_actions: int, hidden: int = 256):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(n_in, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.pi = nn.Linear(hidden, n_actions)
        self.v = nn.Linear(hidden, 1)
        nn.init.orthogonal_(self.pi.weight, 0.01)
        nn.init.zeros_(self.pi.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.body(x)
        return self.pi(h), self.v(h).squeeze(-1)


class RecurrentActorCritic(nn.Module):
    """GRU actor-critic with motor efference copy and factored 3x3x2 action heads."""

    def __init__(self, n_in: int, n_motor: int = 4, hidden: int = 256):
        super().__init__()
        self.n_in = n_in
        self.n_motor = n_motor
        self.hidden = hidden
        self.enc = nn.Sequential(
            nn.Linear(n_in + n_motor, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )
        self.gru = nn.GRUCell(hidden, hidden)
        self.pi_throttle = nn.Linear(hidden, 3)
        self.pi_steer = nn.Linear(hidden, 3)
        self.pi_fire = nn.Linear(hidden, 2)
        self.v = nn.Linear(hidden, 1)
        for head in (self.pi_throttle, self.pi_steer, self.pi_fire):
            nn.init.orthogonal_(head.weight, 0.01)
            nn.init.zeros_(head.bias)
        nn.init.orthogonal_(self.v.weight, 1.0)
        nn.init.zeros_(self.v.bias)

    def step(self, x: torch.Tensor, motor: torch.Tensor, h: torch.Tensor
             ) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
        z = self.enc(torch.cat([x, motor], dim=-1))
        h_next = self.gru(z, h)
        logits = (self.pi_throttle(h_next), self.pi_steer(h_next), self.pi_fire(h_next))
        val = self.v(h_next).squeeze(-1)
        return logits, val, h_next

    def evaluate_chunk(self, x_seq: torch.Tensor, m_seq: torch.Tensor,
                       done_prev_seq: torch.Tensor, h0: torch.Tensor,
                       factored_act: torch.Tensor
                       ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
        """Unroll chunk of shape (L, N, ...) starting from h0 (N, H), zeroing h where done_prev_seq == 1."""
        L = x_seq.shape[0]
        h = h0
        logps, ents, vals = [], [], []
        th_l, st_l, fi_l = [], [], []
        for t in range(L):
            if t > 0:
                h = h * (1.0 - done_prev_seq[t].unsqueeze(-1))
            (l_th, l_st, l_fi), v_t, h = self.step(x_seq[t], m_seq[t], h)
            d_th = torch.distributions.Categorical(logits=l_th)
            d_st = torch.distributions.Categorical(logits=l_st)
            d_fi = torch.distributions.Categorical(logits=l_fi)
            lp = (
                d_th.log_prob(factored_act[t, :, 0])
                + d_st.log_prob(factored_act[t, :, 1])
                + d_fi.log_prob(factored_act[t, :, 2])
            )
            ent = d_th.entropy() + d_st.entropy() + d_fi.entropy()
            logps.append(lp)
            ents.append(ent)
            vals.append(v_t)
            th_l.append(l_th)
            st_l.append(l_st)
            fi_l.append(l_fi)
        return (
            torch.stack(logps, dim=0),
            torch.stack(ents, dim=0),
            torch.stack(vals, dim=0),
            (torch.stack(th_l, dim=0), torch.stack(st_l, dim=0), torch.stack(fi_l, dim=0)),
        )


def make_teacher(action_names: list[str], seed: int = 0) -> CockpitControls:
    ctrl = CockpitControls(
        action_names,
        calibration="pool4",
        saccade_hold=1,
        startle_ticks=6,
        startle_refractory=15,
        steer_mode="li",
        steer_min_hz=6.0,
        search_steer=1.0,
        search_delay=2,
        fovea_li=0.20,
        direct_steer=True,
    )
    ctrl._search_sign = 1.0 if (seed % 2 == 0) else -1.0
    return ctrl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain", choices=["connectome", "rewired", "rewired_ipsi", "silenced"], default="connectome")
    ap.add_argument("--ablate-seed", type=int, default=0)
    ap.add_argument("--weight-scale", type=float, default=1.0)
    ap.add_argument("--features", choices=["brain_bilateral", "brain", "retina"], default="brain_bilateral")
    ap.add_argument("--arch", choices=["gru", "mlp"], default="gru")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--updates", type=int, default=300)
    ap.add_argument("--rollout", type=int, default=128, help="ticks per fly per update")
    ap.add_argument("--seq-len", type=int, default=16, help="BPTT chunk length for GRU")
    ap.add_argument("--episode-ticks", type=int, default=900)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatch", type=int, default=256, help="transitions per minibatch")
    ap.add_argument("--lr", type=float, default=2.5e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--entropy", type=float, default=0.002)
    ap.add_argument("--target-kl", type=float, default=0.025)
    ap.add_argument("--weight-decay", type=float, default=1e-4,
                    help="L2 weight decay on input encoder linear weights")
    ap.add_argument("--norm-var-floor", type=float, default=None,
                    help="variance floor for RunningNorm (default: 1.0 for gru, 0.0 for mlp)")
    ap.add_argument("--life-penalty", type=float, default=0.35,
                    help="reward per life lost is minus this; each kill is +1.0")
    ap.add_argument("--raw-score-reward", action="store_true",
                    help="use raw Battlezone points/1000 instead of unit +1.0 per enemy destroyed")
    ap.add_argument("--idle-penalty", type=float, default=0.002,
                    help="small motor-only penalty per tick of prolonged NOOP (>6 ticks) during training")
    ap.add_argument("--warmstart-updates", type=int, default=0,
                    help="initial DAgger behaviour-cloning + critic warm-start updates from the active reflex")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--injection", default="v10")
    ap.add_argument("--zoom", type=float, default=1.15)
    ap.add_argument("--eye-offset", type=float, default=0.11)
    ap.add_argument("--center-y", type=float, default=0.46)
    ap.add_argument("--eye-fovy", type=float, default=54.0)
    ap.add_argument("--legacy-eyes", action="store_true",
                    help="use legacy non-converged fovy=45 eye cameras")
    ap.add_argument("--no-turn-gate", action="store_true",
                    help="disable discrete self-turn transient gating")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    if args.arch == "gru" and args.rollout % args.seq_len != 0:
        ap.error("--rollout must be divisible by --seq-len for GRU updates")

    tag = args.tag or f"{args.brain}_{args.features}_{args.arch}_s{args.seed}"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ablate = "none" if args.brain == "connectome" else args.brain
    assert ablate in ABLATIONS
    eye_fovy = None if args.legacy_eyes else args.eye_fovy
    converge_eyes = not args.legacy_eyes
    suppress_turn_transients = (not args.legacy_eyes) and (not args.no_turn_gate)
    args.suppress_turn_transients = suppress_turn_transients
    var_floor = (1.0 if args.arch == "gru" else 0.0) if args.norm_var_floor is None else float(args.norm_var_floor)
    args.norm_var_floor = var_floor

    squad = FlySquad(
        batch=args.batch,
        seeds=[args.seed * 100 + i for i in range(args.batch)],
        neural_seed=args.seed,
        injection=args.injection,
        zoom=args.zoom,
        eye_offset=args.eye_offset,
        retina_center=(0.5, args.center_y),
        eye_fovy=eye_fovy,
        converge_eyes=converge_eyes,
        suppress_turn_transients=suppress_turn_transients,
        brain_kwargs={"ablate": ablate, "ablate_seed": args.ablate_seed, "weight_scale": args.weight_scale},
        episode_ticks=args.episode_ticks,
        features=args.features,
        life_penalty=args.life_penalty,
    )
    print(
        f"squad of {args.batch}: brain {args.brain} (scale {args.weight_scale:g}), "
        f"features {args.features} ({squad.n_features} dims), arch {args.arch}, "
        f"{squad.n_actions} actions, injection {args.injection}, "
        f"eyes {'legacy' if args.legacy_eyes else f'fovy={eye_fovy} converged'}, "
        f"turn_gate={suppress_turn_transients}, var_floor={var_floor:g}, "
        f"warmstart {args.warmstart_updates}/{args.updates} updates",
        flush=True,
    )

    dev = torch.device("cuda")
    a_to_f, f_to_a = build_action_codec(squad.action_names)
    noop_id = squad.action_names.index("NOOP")

    if args.arch == "mlp":
        net = ActorCritic(squad.n_features, squad.n_actions, hidden=args.hidden).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=args.lr, eps=1e-5)
    else:
        net = RecurrentActorCritic(squad.n_features, n_motor=4, hidden=args.hidden).to(dev)
        decay_params, nodecay_params = [], []
        for pname, p in net.named_parameters():
            if pname.startswith("enc.") and pname.endswith(".weight") and p.ndim == 2:
                decay_params.append(p)
            else:
                nodecay_params.append(p)
        opt = torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": args.weight_decay},
                {"params": nodecay_params, "weight_decay": 0.0},
            ],
            lr=args.lr,
            eps=1e-5,
        )

    norm = RunningNorm(
        squad.n_features,
        clip=10.0 if args.arch == "gru" else None,
        var_floor=var_floor,
    )

    obs = squad.reset()
    norm.update(obs)

    B, T = args.batch, args.rollout
    h_cur = torch.zeros((B, args.hidden), dtype=torch.float32, device=dev)
    prev_factored = np.tile(np.array([1, 1, 0], dtype=np.int64), (B, 1))
    since_fire = np.full(B, 20, dtype=np.int64)
    noop_streak = np.zeros(B, dtype=np.int64)
    teachers = [make_teacher(squad.action_names, i) for i in range(B)] if args.warmstart_updates > 0 else None

    OUT.mkdir(exist_ok=True)
    log = open(OUT / f"readout_{tag}.jsonl", "w")
    episodes: list[dict] = []
    best_utility = -1e9

    def save_ckpt(path: Path, update_idx: int, extra_meta: dict | None = None) -> None:
        payload = {
            "net": net.state_dict(),
            "arch": args.arch,
            "hidden": args.hidden,
            "norm_mean": norm.mean,
            "norm_var": norm.var,
            "norm_clip": norm.clip,
            "norm_var_floor": norm.var_floor,
            "features": args.features,
            "feature_types": squad.feature_types,
            "brain": args.brain,
            "ablate_seed": args.ablate_seed,
            "weight_scale": args.weight_scale,
            "n_actions": squad.n_actions,
            "action_names": squad.action_names,
            "args": vars(args),
        }
        if extra_meta:
            payload.update(extra_meta)
        torch.save(attach(payload, args=args, brain=squad.brain, update=update_idx), path)

    rng = np.random.default_rng(args.seed + 99991)

    for update in range(1, args.updates + 1):
        t0 = time.time()
        in_warmstart = update <= args.warmstart_updates
        update_episodes = []

        buf_obs = np.zeros((T, B, squad.n_features), np.float32)
        buf_mot = np.zeros((T, B, 4), np.float32)
        buf_h = np.zeros((T, B, args.hidden), np.float32)
        buf_done_prev = np.zeros((T, B), np.float32)
        buf_act = np.zeros((T, B), np.int64)
        buf_fact = np.zeros((T, B, 3), np.int64)
        buf_teacher_fact = np.zeros((T, B, 3), np.int64)
        buf_logp = np.zeros((T, B), np.float32)
        buf_rew = np.zeros((T, B), np.float32)
        buf_done = np.zeros((T, B), np.float32)
        buf_val = np.zeros((T + 1, B), np.float32)

        done_prev = np.zeros(B, np.float32)

        for t in range(T):
            x = norm(obs)
            m = motor_efference(prev_factored, since_fire)
            buf_obs[t] = x
            buf_mot[t] = m
            buf_done_prev[t] = done_prev

            if args.arch == "mlp":
                with torch.no_grad():
                    logits, value = net(torch.from_numpy(x).to(dev))
                    dist = torch.distributions.Categorical(logits=logits)
                    act_t = dist.sample()
                    logp_t = dist.log_prob(act_t)
                a_student = act_t.cpu().numpy()
                f_student = a_to_f[a_student]
                logp_np = logp_t.cpu().numpy()
                val_np = value.cpu().numpy()
            else:
                buf_h[t] = h_cur.cpu().numpy()
                with torch.no_grad():
                    (l_th, l_st, l_fi), value, h_cur = net.step(
                        torch.from_numpy(x).to(dev),
                        torch.from_numpy(m).to(dev),
                        h_cur,
                    )
                    d_th = torch.distributions.Categorical(logits=l_th)
                    d_st = torch.distributions.Categorical(logits=l_st)
                    d_fi = torch.distributions.Categorical(logits=l_fi)
                    th_s, st_s, fi_s = d_th.sample(), d_st.sample(), d_fi.sample()
                    logp_t = d_th.log_prob(th_s) + d_st.log_prob(st_s) + d_fi.log_prob(fi_s)
                f_student = np.stack([th_s.cpu().numpy(), st_s.cpu().numpy(), fi_s.cpu().numpy()], axis=-1)
                a_student = f_to_a[f_student[:, 0], f_student[:, 1], f_student[:, 2]]
                logp_np = logp_t.cpu().numpy()
                val_np = value.cpu().numpy()

            if in_warmstart and teachers is not None:
                a_teach = np.array([teachers[i].to_action(commit=False) for i in range(B)], dtype=np.int64)
                f_teach = a_to_f[a_teach]
                buf_teacher_fact[t] = f_teach
                # DAgger schedule: anneal teacher execution probability from 0.85 -> 0.25
                beta = 0.85 - 0.60 * ((update - 1) / max(1, args.warmstart_updates - 1))
                use_t = rng.random(B) < beta
                a = np.where(use_t, a_teach, a_student)
                f_exec = a_to_f[a]
                for i in range(B):
                    if f_exec[i, 1] != 1 and teachers[i].saccade_hold > 0:
                        teachers[i]._hold = teachers[i].saccade_hold
            else:
                a = a_student
                f_exec = f_student

            nobs, rew, done, infos = squad.step(a)

            # Align RL signal directly with enemy kills (avoiding 5x flying-saucer critic spikes)
            if args.arch == "gru" and not args.raw_score_reward:
                shaped_rew = np.array(
                    [float(inf["kill"]) - args.life_penalty * float(inf["life_lost"]) for inf in infos],
                    dtype=np.float32,
                )
            else:
                shaped_rew = rew.copy()
            if args.idle_penalty > 0.0:
                is_noop = (a == noop_id)
                noop_streak = np.where(is_noop, noop_streak + 1, 0)
                shaped_rew -= args.idle_penalty * (noop_streak > 6).astype(np.float32)

            buf_act[t] = a
            buf_fact[t] = f_exec
            buf_logp[t] = logp_np
            buf_rew[t] = shaped_rew
            buf_done[t] = done.astype(np.float32)
            buf_val[t] = val_np

            # Update internal motor efference copy & optional teacher controllers
            since_fire = np.where(f_exec[:, 2] == 1, 0, since_fire + 1)
            prev_factored = f_exec.copy()
            done_prev = done.astype(np.float32)

            for i, inf in enumerate(infos):
                if "episode" in inf:
                    episode = {"fly": i, **inf["episode"]}
                    episodes.append(episode)
                    update_episodes.append(episode)
                if done[i]:
                    h_cur[i].zero_()
                    prev_factored[i] = (1, 1, 0)
                    since_fire[i] = 20
                    noop_streak[i] = 0
                    if teachers is not None:
                        teachers[i] = make_teacher(squad.action_names, i + update)
                elif teachers is not None:
                    teachers[i].update(squad.last_dn[i], squad.dt)

            obs = nobs
            norm.update(obs)

        with torch.no_grad():
            x_last = norm(obs)
            if args.arch == "mlp":
                _, v_last = net(torch.from_numpy(x_last).to(dev))
            else:
                m_last = motor_efference(prev_factored, since_fire)
                _, v_last, _ = net.step(
                    torch.from_numpy(x_last).to(dev),
                    torch.from_numpy(m_last).to(dev),
                    h_cur,
                )
        buf_val[T] = v_last.cpu().numpy()

        # GAE
        adv = np.zeros((T, B), np.float32)
        last = np.zeros(B, np.float32)
        for t in reversed(range(T)):
            nonterm = 1.0 - buf_done[t]
            delta = buf_rew[t] + args.gamma * buf_val[t + 1] * nonterm - buf_val[t]
            last = delta + args.gamma * args.lam * nonterm * last
            adv[t] = last
        ret = adv + buf_val[:T]

        stats = {"pg": 0.0, "vf": 0.0, "ent": 0.0, "kl": 0.0}
        k = 0

        if args.arch == "mlp":
            n = T * B
            f_obs = torch.from_numpy(buf_obs.reshape(n, -1)).to(dev)
            f_act = torch.from_numpy(buf_act.reshape(n)).to(dev)
            f_logp = torch.from_numpy(buf_logp.reshape(n)).to(dev)
            f_adv = torch.from_numpy(adv.reshape(n)).to(dev)
            f_adv = (f_adv - f_adv.mean()) / (f_adv.std() + 1e-8)
            f_ret = torch.from_numpy(ret.reshape(n)).to(dev)
            for _ in range(args.epochs):
                perm = torch.randperm(n, device=dev)
                for s in range(0, n, args.minibatch):
                    mb = perm[s : s + args.minibatch]
                    logits, value = net(f_obs[mb])
                    dist = torch.distributions.Categorical(logits=logits)
                    logp = dist.log_prob(f_act[mb])
                    ratio = torch.exp(logp - f_logp[mb])
                    pg = -torch.min(
                        ratio * f_adv[mb],
                        torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * f_adv[mb],
                    ).mean()
                    vf = 0.5 * (value - f_ret[mb]).pow(2).mean()
                    ent = dist.entropy().mean()
                    loss = pg + 0.5 * vf - args.entropy * ent
                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                    opt.step()
                    stats["pg"] += pg.item()
                    stats["vf"] += vf.item()
                    stats["ent"] += ent.item()
                    stats["kl"] += (f_logp[mb] - logp).mean().item()
                    k += 1
        else:
            # Chunked BPTT updates of shape (L, N_chunks, ...)
            L = args.seq_len
            n_chunks_per_fly = T // L
            N = n_chunks_per_fly * B

            def to_chunks(arr: np.ndarray) -> torch.Tensor:
                # (T, B, ...) -> (n_chunks_per_fly, L, B, ...) -> (L, N, ...)
                rest = arr.shape[2:]
                reshaped = arr.reshape(n_chunks_per_fly, L, B, *rest)
                transposed = np.swapaxes(reshaped, 0, 1).reshape(L, N, *rest)
                return torch.from_numpy(np.ascontiguousarray(transposed)).to(dev)

            c_obs = to_chunks(buf_obs)
            c_mot = to_chunks(buf_mot)
            c_done_prev = to_chunks(buf_done_prev)
            c_fact = to_chunks(buf_fact)
            c_teach = to_chunks(buf_teacher_fact)
            c_logp = to_chunks(buf_logp)
            c_adv = to_chunks(adv)
            c_adv = (c_adv - c_adv.mean()) / (c_adv.std() + 1e-8)
            c_ret = to_chunks(ret)
            c_val_old = to_chunks(buf_val[:T])
            # Initial hidden state at t=0 of each chunk: (N, H)
            c_h0 = to_chunks(buf_h)[0]

            chunks_per_mb = max(1, args.minibatch // L)
            stop_early = False
            for _ in range(args.epochs):
                if stop_early:
                    break
                perm = torch.randperm(N, device=dev)
                for s in range(0, N, chunks_per_mb):
                    mb = perm[s : s + chunks_per_mb]
                    logp, ent, val, (l_th, l_st, l_fi) = net.evaluate_chunk(
                        c_obs[:, mb], c_mot[:, mb], c_done_prev[:, mb], c_h0[mb], c_fact[:, mb]
                    )
                    v_clipped = c_val_old[:, mb] + torch.clamp(
                        val - c_val_old[:, mb], -args.clip, args.clip
                    )
                    vf = 0.5 * torch.max((val - c_ret[:, mb]).pow(2), (v_clipped - c_ret[:, mb]).pow(2)).mean()
                    ent_m = ent.mean()

                    if in_warmstart:
                        # Supervised imitation of active reflex + critic warm-start
                        t_mb = c_teach[:, mb]
                        bc_loss = (
                            F.cross_entropy(l_th.reshape(-1, 3), t_mb[:, :, 0].reshape(-1))
                            + F.cross_entropy(l_st.reshape(-1, 3), t_mb[:, :, 1].reshape(-1))
                            + F.cross_entropy(l_fi.reshape(-1, 2), t_mb[:, :, 2].reshape(-1))
                        )
                        pg = bc_loss
                        approx_kl = 0.0
                    else:
                        logratio = logp - c_logp[:, mb]
                        ratio = torch.exp(logratio)
                        approx_kl = float(((ratio - 1.0) - logratio).mean().item())
                        if args.target_kl > 0 and approx_kl > 1.5 * args.target_kl:
                            stop_early = True
                            break
                        pg = -torch.min(
                            ratio * c_adv[:, mb],
                            torch.clamp(ratio, 1.0 - args.clip, 1.0 + args.clip) * c_adv[:, mb],
                        ).mean()

                    loss = pg + 0.5 * vf - args.entropy * ent_m
                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                    opt.step()
                    stats["pg"] += float(pg.item())
                    stats["vf"] += float(vf.item())
                    stats["ent"] += float(ent_m.item())
                    stats["kl"] += approx_kl
                    k += 1

        recent = episodes[-32:]
        ep_kills = float(np.mean([e["kills"] for e in recent])) if recent else None
        ep_lives = float(np.mean([e["lives_lost"] for e in recent])) if recent else None
        ep_rew = float(np.mean([e["reward"] for e in recent])) if recent else None
        ep_util = (ep_kills - 0.5 * ep_lives) if (ep_kills is not None and ep_lives is not None) else None

        row = {
            "update": update,
            "phase": "warmstart" if in_warmstart else "ppo",
            "ticks": update * T * B,
            "fly_ticks_per_s": T * B / (time.time() - t0),
            "rollout_reward_per_1k": float(buf_rew.sum() / (T * B) * 1000),
            "episodes": len(episodes),
            "episode_records": update_episodes,
            "ep_kills": ep_kills,
            "ep_lives": ep_lives,
            "ep_utility": ep_util,
            "ep_reward": ep_rew,
            "entropy": stats["ent"] / max(k, 1),
            "kl": stats["kl"] / max(k, 1),
            "vf": stats["vf"] / max(k, 1),
        }
        log.write(json.dumps(row) + "\n")
        log.flush()
        print(
            f"upd {update:4d} ({row['phase']:9s}) | {row['ticks']:8d} ticks | "
            f"{row['fly_ticks_per_s']:5.0f} fly-ticks/s | "
            f"ep kills {str(round(ep_kills, 2)) if ep_kills is not None else 'None':>5s} "
            f"lives {str(round(ep_lives, 2)) if ep_lives is not None else 'None':>5s} "
            f"util {str(round(ep_util, 2)) if ep_util is not None else 'None':>6s} "
            f"({len(episodes)} eps) | ent {row['entropy']:.2f} kl {row['kl']:.4f}",
            flush=True,
        )

        if update % 10 == 0 or update == args.updates:
            save_ckpt(OUT / f"readout_{tag}.pt", update, {"rolling_utility": ep_util})

        if (not in_warmstart) and len(recent) >= 32 and ep_util is not None and ep_util > best_utility:
            best_utility = ep_util
            save_ckpt(
                OUT / f"readout_{tag}_best.pt",
                update,
                {"best_rolling_utility": best_utility, "best_ep_kills": ep_kills, "best_ep_lives": ep_lives},
            )

    log.close()
    squad.close()
    print(f"saved out/readout_{tag}.pt (and _best.pt if applicable) and out/readout_{tag}.jsonl", flush=True)


if __name__ == "__main__":
    main()
