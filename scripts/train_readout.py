"""Train a readout policy on the frozen connectome with PPO.

The brain is never trained. A small MLP maps the pooled spike counts of the
descending, visual-projection and visual-centrifugal neurons to Battlezone
actions and is trained against the score. Run it three times with the same
seeds and compare: --brain connectome | rewired | silenced, plus
--features retina for the no-brain ceiling. That comparison, not the score,
is the point: does the measured wiring give a better feature map than a
random one?

Outputs: out/readout_<tag>.pt (policy + normaliser), out/readout_<tag>.jsonl
(one line per update with kills, lives, reward, throughput).
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from noflyzone.rl_env import FlySquad

OUT = Path(__file__).resolve().parent.parent / "out"


class RunningNorm:
    def __init__(self, n):
        self.mean = np.zeros(n, np.float64); self.var = np.ones(n, np.float64); self.count = 1e-4

    def update(self, x):
        bm, bv, bc = x.mean(0), x.var(0), x.shape[0]
        d = bm - self.mean; tot = self.count + bc
        self.mean += d * bc / tot
        self.var = (self.var * self.count + bv * bc + d ** 2 * self.count * bc / tot) / tot
        self.count = tot

    def __call__(self, x):
        return ((x - self.mean) / np.sqrt(self.var + 1e-8)).astype(np.float32)


class ActorCritic(nn.Module):
    def __init__(self, n_in, n_actions, hidden=256):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(n_in, hidden), nn.Tanh(),
                                  nn.Linear(hidden, hidden), nn.Tanh())
        self.pi = nn.Linear(hidden, n_actions)
        self.v = nn.Linear(hidden, 1)
        nn.init.orthogonal_(self.pi.weight, 0.01); nn.init.zeros_(self.pi.bias)

    def forward(self, x):
        h = self.body(x)
        return self.pi(h), self.v(h).squeeze(-1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain", choices=["connectome", "rewired", "silenced"], default="connectome")
    ap.add_argument("--features", choices=["brain", "retina"], default="brain")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--updates", type=int, default=100)
    ap.add_argument("--rollout", type=int, default=128, help="ticks per fly per update")
    ap.add_argument("--episode-ticks", type=int, default=900)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--entropy", type=float, default=0.01)
    ap.add_argument("--life-penalty", type=float, default=0.5,
                    help="reward per life lost is minus this; a kill is +1. 0.1 is the kill-shaped setting")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--injection", default="v4")
    ap.add_argument("--zoom", type=float, default=2.0)
    ap.add_argument("--eye-offset", type=float, default=0.15)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    tag = args.tag or f"{args.brain}_{args.features}_s{args.seed}"
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    ablate = "none" if args.brain == "connectome" else args.brain
    squad = FlySquad(batch=args.batch, seeds=[args.seed * 100 + i for i in range(args.batch)],
                     injection=args.injection, zoom=args.zoom, eye_offset=args.eye_offset,
                     brain_kwargs={"ablate": ablate}, episode_ticks=args.episode_ticks,
                     features=args.features, life_penalty=args.life_penalty)
    print(f"squad of {args.batch}: brain {args.brain}, features {args.features} "
          f"({squad.n_features} dims), {squad.n_actions} actions, episode {args.episode_ticks} ticks",
          flush=True)
    dev = torch.device("cuda")
    net = ActorCritic(squad.n_features, squad.n_actions).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, eps=1e-5)
    norm = RunningNorm(squad.n_features)
    obs = squad.reset()
    norm.update(obs)
    OUT.mkdir(exist_ok=True)
    log = open(OUT / f"readout_{tag}.jsonl", "w")
    episodes = []
    T, B = args.rollout, args.batch
    for update in range(1, args.updates + 1):
        t0 = time.time()
        buf_obs = np.zeros((T, B, squad.n_features), np.float32)
        buf_act = np.zeros((T, B), np.int64); buf_logp = np.zeros((T, B), np.float32)
        buf_rew = np.zeros((T, B), np.float32); buf_done = np.zeros((T, B), np.float32)
        buf_val = np.zeros((T + 1, B), np.float32)
        for t in range(T):
            x = norm(obs)
            with torch.no_grad():
                logits, value = net(torch.from_numpy(x).to(dev))
                dist = torch.distributions.Categorical(logits=logits)
                act = dist.sample()
                logp = dist.log_prob(act)
            a = act.cpu().numpy()
            nobs, rew, done, infos = squad.step(a)
            buf_obs[t], buf_act[t], buf_logp[t] = x, a, logp.cpu().numpy()
            buf_rew[t], buf_done[t], buf_val[t] = rew, done, value.cpu().numpy()
            for inf in infos:
                if "episode" in inf:
                    episodes.append(inf["episode"])
            obs = nobs
            norm.update(obs)
        with torch.no_grad():
            _, v_last = net(torch.from_numpy(norm(obs)).to(dev))
        buf_val[T] = v_last.cpu().numpy()
        # GAE
        adv = np.zeros((T, B), np.float32); last = np.zeros(B, np.float32)
        for t in reversed(range(T)):
            nonterm = 1.0 - buf_done[t]
            delta = buf_rew[t] + args.gamma * buf_val[t + 1] * nonterm - buf_val[t]
            last = delta + args.gamma * args.lam * nonterm * last
            adv[t] = last
        ret = adv + buf_val[:T]
        # PPO update
        n = T * B
        f_obs = torch.from_numpy(buf_obs.reshape(n, -1)).to(dev)
        f_act = torch.from_numpy(buf_act.reshape(n)).to(dev)
        f_logp = torch.from_numpy(buf_logp.reshape(n)).to(dev)
        f_adv = torch.from_numpy(adv.reshape(n)).to(dev)
        f_adv = (f_adv - f_adv.mean()) / (f_adv.std() + 1e-8)
        f_ret = torch.from_numpy(ret.reshape(n)).to(dev)
        stats = {"pg": 0.0, "vf": 0.0, "ent": 0.0, "kl": 0.0}; k = 0
        for _ in range(args.epochs):
            perm = torch.randperm(n, device=dev)
            for s in range(0, n, args.minibatch):
                mb = perm[s: s + args.minibatch]
                logits, value = net(f_obs[mb])
                dist = torch.distributions.Categorical(logits=logits)
                logp = dist.log_prob(f_act[mb])
                ratio = torch.exp(logp - f_logp[mb])
                pg = -torch.min(ratio * f_adv[mb],
                                torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * f_adv[mb]).mean()
                vf = 0.5 * (value - f_ret[mb]).pow(2).mean()
                ent = dist.entropy().mean()
                loss = pg + 0.5 * vf - args.entropy * ent
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 0.5); opt.step()
                stats["pg"] += pg.item(); stats["vf"] += vf.item(); stats["ent"] += ent.item()
                stats["kl"] += (f_logp[mb] - logp).mean().item(); k += 1
        recent = episodes[-20:]
        row = {"update": update, "ticks": update * T * B, "fly_ticks_per_s": T * B / (time.time() - t0),
               "rollout_reward_per_1k": float(buf_rew.sum() / (T * B) * 1000),
               "episodes": len(episodes),
               "ep_kills": float(np.mean([e["kills"] for e in recent])) if recent else None,
               "ep_lives": float(np.mean([e["lives_lost"] for e in recent])) if recent else None,
               "ep_reward": float(np.mean([e["reward"] for e in recent])) if recent else None,
               "entropy": stats["ent"] / max(k, 1), "kl": stats["kl"] / max(k, 1),
               "vf": stats["vf"] / max(k, 1)}
        log.write(json.dumps(row) + "\n"); log.flush()
        print(f"upd {update:4d} | {row['ticks']:8d} ticks | {row['fly_ticks_per_s']:5.0f} fly-ticks/s | "
              f"reward/1k {row['rollout_reward_per_1k']:+6.2f} | ep kills {row['ep_kills']} lives {row['ep_lives']} "
              f"({len(episodes)} eps) | ent {row['entropy']:.2f} kl {row['kl']:.4f}", flush=True)
        if update % 10 == 0 or update == args.updates:
            torch.save({"net": net.state_dict(), "norm_mean": norm.mean, "norm_var": norm.var,
                        "features": args.features, "feature_types": squad.feature_types,
                        "brain": args.brain, "n_actions": squad.n_actions, "args": vars(args)},
                       OUT / f"readout_{tag}.pt")
    squad.close()
    print(f"saved out/readout_{tag}.pt and out/readout_{tag}.jsonl")


if __name__ == "__main__":
    main()
