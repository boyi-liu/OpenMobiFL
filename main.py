import importlib
import sys
import numpy as np
import os

from utils.options import args_parser
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class FedSim:
    def __init__(self, args):
        self.args = args
        args.suffix = f'exp/{args.suffix}'
        if not os.path.exists(f'./{args.suffix}'):
            os.makedirs(f'./{args.suffix}')

        output_path = f'{args.suffix}/{args.alg}_{args.dataset}_{args.model}_' \
                      f'{args.total_num}c_{args.epoch}E_lr{args.lr}'

        # === load trainer ===
        alg_module = importlib.import_module(f'alg.{args.alg}')

        # === init clients & server ===
        self.clients = [alg_module.Client(idx, args) for idx in tqdm(range(args.total_num))]
        self.server = alg_module.Server(0, args, self.clients)
        self.server.output = open(f'./{output_path}.txt', 'a')

    def simulate(self):
        acc_list = []
        TEST_GAP = self.args.test_gap

        try:
            for rnd in tqdm(range(0, self.server.total_round), desc='Communication Round', leave=False):
                # ===================== train =====================
                self.server.round = rnd
                self.server.run()

                # ===================== test =====================
                if (self.server.total_round - rnd <= 10) or (rnd % TEST_GAP == (TEST_GAP-1)):
                    ret_dict = self.server.test_all()
                    acc = ret_dict['acc']
                    acc_list.append(acc)

                    self.server.output.write(f'[Round {rnd}] Acc: {acc:.2f} | Time: {self.server.wall_clock_time:.2f}s\n')
                    self.server.output.flush()

        except KeyboardInterrupt:
            ...
        finally:
            avg_count = 10
            acc_avg = np.mean(acc_list[-avg_count:]).item()
            acc_max = np.max(acc_list).item()

            self.server.output.write('==========Summary==========\n')
            self.server.output.write(f'[Total] Acc: {acc_avg:.2f} | Max Acc: {acc_max:.2f}\n')


if __name__ == '__main__':
    args = args_parser()
    fed = FedSim(args=args)
    fed.simulate()