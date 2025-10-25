import argparse
import importlib
import yaml


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alg', type=str, help='Algorithm')

    # ===== Basic Setting ======
    parser.add_argument('--suffix', type=str, help="Suffix for file")
    parser.add_argument('--device', type=int, help="Device to use")
    parser.add_argument('--dataset', type=str, help="Dataset")
    parser.add_argument('--model', type=str, help="Model")

    # ===== Federated Setting =====
    parser.add_argument('--total_num', type=int, help="Total clients num")
    parser.add_argument('--sr', type=float, help="Clients sample rate")
    parser.add_argument('--esr', type=float, help="Edge sample rate")
    parser.add_argument('--rnd', type=int, help="Communication rounds")
    parser.add_argument('--edge_rnd', type=int, help="Edge communication rounds")
    parser.add_argument('--test_gap', type=int, help='Rounds between test phases')
    parser.add_argument('--delta_time', type=float, help='Minutes required for aggregation')

    # ===== Mobile Setting =====
    parser.add_argument('--traj', type=str, help="Trajectory")
    parser.add_argument('--comm_range', type=float, help="Communication range")
    parser.add_argument('--grid_range', type=float, help="Range of the whole grid")

    # ===== Local Training Setting =====
    parser.add_argument('--bs', type=int, help="Batch size")
    parser.add_argument('--epoch', type=int, help="Epoch num")
    parser.add_argument('--lr', type=float, help="Learning rate")
    parser.add_argument('--gamma', type=float, help="Exponential decay of learning rate")

    # === read args from yaml ===
    with open('config.yaml', 'r') as f:
        yaml_config = yaml.load(f.read(), Loader=yaml.Loader)
    for k, v in yaml_config.items():
        parser.set_defaults(**{k: v})

    # === read args from command ===
    args, _ = parser.parse_known_args()

    # === read specific args from each method
    alg_module = importlib.import_module(f'alg.{args.alg}')
    spec_args = alg_module.add_args(parser) if hasattr(alg_module, 'add_args') else args
    return spec_args