from mjrl.utils.gym_env import GymEnv
from mjrl.policies.gaussian_mlp import MLP
from mjrl.baselines.quadratic_baseline import QuadraticBaseline
from mjrl.baselines.mlp_baseline import MLPBaseline
from mjrl.algos.npg_cg import NPG
from mjrl.algos.dapg import DAPG
from mjrl.algos.behavior_cloning import BC
from mjrl.utils.train_agent import train_agent
from mjrl.samplers.core import sample_paths
import os
import json
import mjrl.envs
import mj_envs
import time as timer
import pickle
import argparse
import mj_allegro_envs

import hydra
import logging
import os
import torch
import wandb

from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


def set_device(cuda : bool):
    device = "cpu"
    if cuda:
        if torch.cuda.is_available():
            device = "cuda"
        else:
            log.info("no gpu found, defaulting to cpu")
    device = torch.device(device)


def train(flags : DictConfig):
    pass


def valid(flags : DictConfig):
    pass


def main(flags : DictConfig):
    set_device(flags.cuda)
    torch.manual_seed(flags.random_seed)
    logging.info("Running...")

    # assert 'algorithm' in flags.job_data.keys()
    assert any([flags.job_data.algorithm == a for a in ['NPG', 'BCRL', 'DAPG']])
    # flags.job_data.lam_0 = 0.0 if 'lam_0' not in flags.job_data.keys() else flags.job_data.lam_0
    # flags.job_data.lam_1 = 0.0 if 'lam_1' not in flags.job_data.keys() else flags.job_data.lam_1
    # EXP_FILE = JOB_DIR + '/job_config.json'
    # with open(EXP_FILE, 'w') as f:
    #     json.dump(flags.job_data, f, indent=4)
    # logging.info(json.dump(flags.job_data, f, indent=4))

    # ===============================================================================
    # Train Loop
    # ===============================================================================
    wandb.init(project=flags.wbproject,group=flags.job_data.algorithm,config=flags.job_data, reinit=True)

    e = GymEnv(flags.job_data.env)
    policy = MLP(e.spec, hidden_sizes=tuple(flags.job_data.policy_size), seed=flags.job_data.seed)
    baseline = MLPBaseline(e.spec, reg_coef=1e-3, batch_size=flags.job_data.vf_batch_size,
                           epochs=flags.job_data.vf_epochs, learn_rate=flags.job_data.vf_learn_rate)

    # Get demonstration data if necessary and behavior clone
    if flags.job_data.algorithm != 'NPG':
        print("========================================")
        print("Collecting expert demonstrations")
        print("========================================")
        demo_paths = pickle.load(open(flags.job_data.demo_file, 'rb'))

        bc_agent = BC(demo_paths, policy=policy, epochs=flags.job_data.bc_epochs, batch_size=flags.job_data.bc_batch_size,
                      lr=flags.job_data.bc_learn_rate, loss_type='MSE', set_transforms=False)
        in_shift, in_scale, out_shift, out_scale = bc_agent.compute_transformations()
        bc_agent.set_transformations(in_shift, in_scale, out_shift, out_scale)
        bc_agent.set_variance_with_data(out_scale)

        ts = timer.time()
        print("========================================")
        print("Running BC with expert demonstrations")
        print("========================================")
        bc_agent.train()
        print("========================================")
        print("BC training complete !!!")
        print("time taken = %f" % (timer.time() - ts))
        print("========================================")

        if flags.job_data.eval_rollouts >= 1:
            score = e.evaluate_policy(policy, num_episodes=flags.job_data.eval_rollouts, mean_action=True)
            print("Score with behavior cloning = %f" % score[0][0])

    if flags.job_data.algorithm != 'DAPG':
        # We throw away the demo data when training from scratch or fine-tuning with RL without explicit augmentation
        demo_paths = None

    # ===============================================================================
    # RL Loop
    # ===============================================================================

    rl_agent = DAPG(e, policy, baseline, demo_paths,
                    normalized_step_size=flags.job_data.rl_step_size,
                    lam_0=flags.job_data.lam_0, lam_1=flags.job_data.lam_1,
                    seed=flags.job_data.seed, save_logs=True
                    )

    print("========================================")
    print("Starting reinforcement learning phase")
    print("========================================")

    ts = timer.time()
    train_agent(job_name=os.getcwd(),
                agent=rl_agent,
                seed=flags.job_data.seed,
                niter=flags.job_data.rl_num_iter,
                gamma=flags.job_data.rl_gamma,
                gae_lambda=flags.job_data.rl_gae,
                num_cpu=flags.job_data.num_cpu,
                sample_mode='trajectories',
                num_traj=flags.job_data.rl_num_traj,
                save_freq=flags.job_data.save_freq,
                evaluation_rollouts=flags.job_data.eval_rollouts)
    print("time taken = %f" % (timer.time()-ts))


# use hydra for config / savings outputs
@hydra.main(config_path=".", config_name="config")
def setup(flags : DictConfig):
    if os.path.exists("config.yaml"):
        # this lets us requeue runs without worrying if we changed our local config since then
        logging.info("loading pre-existing configuration, we're continuing a previous run")
        new_flags = OmegaConf.load("config.yaml")
        cli_conf = OmegaConf.from_cli()
        # however, you can override parameters from the cli still
        # this is useful e.g. if you did total_epochs=N before and want to increase it
        flags = OmegaConf.merge(new_flags, cli_conf)

    # log config + save it to local directory
    log.info(OmegaConf.to_yaml(flags))
    OmegaConf.save(flags, "config.yaml")

    if flags.wandb:
        wandb.init(project=flags.wbproject, entity=flags.wbentity, group=flags.group, config=flags)
    
    main(flags)


if __name__ == "__main__":
    setup()
