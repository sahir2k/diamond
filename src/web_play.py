import argparse
from pathlib import Path

from huggingface_hub import snapshot_download
from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import torch

from agent import Agent
from envs import WorldModelEnv
from game import PlayEnv
from web.web_game import WebGame

OmegaConf.register_new_resolver("eval", eval)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--mouse-multiplier", type=int, default=10)
    parser.add_argument("--size-multiplier", type=int, default=2)
    parser.add_argument("--fps", type=int, default=15)
    return parser.parse_args()


def prepare_play_mode(cfg: DictConfig) -> PlayEnv:
    path_hf = Path(snapshot_download(repo_id="eloialonso/diamond", allow_patterns="csgo/*"))
    path_ckpt = path_hf / "csgo/model/csgo.pt"
    spawn_dir = path_hf / "csgo/spawn"

    cfg.agent = OmegaConf.load(path_hf / "csgo/config/agent/csgo.yaml")
    cfg.env = OmegaConf.load(path_hf / "csgo/config/env/csgo.yaml")

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    num_actions = cfg.env.num_actions
    agent = Agent(instantiate(cfg.agent, num_actions=num_actions)).to(device).eval()
    agent.load(path_ckpt)

    sl = cfg.agent.denoiser.inner_model.num_steps_conditioning
    if agent.upsampler is not None:
        sl = max(sl, cfg.agent.upsampler.inner_model.num_steps_conditioning)
    wm_env_cfg = instantiate(cfg.world_model_env, num_batches_to_preload=1)
    wm_env = WorldModelEnv(agent.denoiser, agent.upsampler, agent.rew_end_model, spawn_dir, 1, sl, wm_env_cfg, return_denoising_trajectory=True)

    play_env = PlayEnv(agent, wm_env, False, False, False)
    return play_env


@torch.no_grad()
def main():
    args = parse_args()
    with initialize(version_base="1.3", config_path="../config"):
        cfg = compose(config_name="trainer")

    h, w = (cfg.env.train.size,) * 2 if isinstance(cfg.env.train.size, int) else cfg.env.train.size
    size_h, size_w = h * args.size_multiplier, w * args.size_multiplier
    env = prepare_play_mode(cfg)
    game = WebGame(env, (size_h, size_w), args.mouse_multiplier, fps=args.fps, host=args.host, port=args.port)
    game.run()


if __name__ == "__main__":
    main()
