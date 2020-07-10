import torch
import numpy as np
from offlinerl.utils.args import get_args, get_params
from offlinerl.trainer import Trainer
from offlinerl.agent import get_agent
from offlinerl.datasets import get_dataset_env
from offlinerl.utils.logger import Logger
# from 


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.cuda:
        torch.backends.cudnn.deterministic = True

    params = get_params(args.config)
    env, dataset = get_dataset_env(params["env_id"])
    env.seed(args.seed)

    logger = Logger(
        experiment_id=args.id,
        env_name=params["env_id"],
        seed=args.seed,
        params=params,
        log_dir=args.log_dir)

    device = torch.device("cuda:{}".format(args.device) if args.cuda else "cpu")
    params["agent"]["device"] = device

    agent = get_agent(params["agent_id"], env, params["agent"])
    # agent = algo.DDPG(
    #     env, **params["agent"])

    trainer = Trainer(
        env, dataset, agent, logger,
        **params["training"])
    trainer.train()


if __name__ == "__main__":
    args = get_args()
    main(args)