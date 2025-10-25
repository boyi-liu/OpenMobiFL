import random
from collections import deque

import numpy as np
import torch
from torch.utils.data import DataLoader

from alg.base import BaseClient, BaseServer
from utils.data_utils import read_client_data

class MobClient(BaseClient):
    def __init__(self, id, args):
        super().__init__(id, args)
        self.is_covered = False

        # load traj data & init position
        traj_data = np.load(f'./trajectory/{args.traj}/{self.id}.npz',
                            allow_pickle=True)
        self.positions = traj_data['position']
        self.timestamps = traj_data['timestamp']
        self.current_pos = self.positions[0]

    def init_data(self):
        self.train_threshold = 600
        self.test_threshold = 200
        self.dataset_train = deque(maxlen=self.train_threshold)
        self.dataset_test = deque(maxlen=self.test_threshold)

    def update_position(self, global_time):
        current_pos = self.positions[0] 
        for pos, timestamp in zip(self.positions, self.timestamps):
            if timestamp > global_time: 
                break
            current_pos = pos 
        self.current_pos = current_pos

    def load_dataset(self):
        SAMPLE_GAP = 3
        train_samples = random.sample(self.current_region.dataset_train,
                                      self.train_threshold // SAMPLE_GAP)
        self.dataset_train.extend(train_samples)

        test_samples = random.sample(self.current_region.dataset_test,
                                     self.test_threshold // SAMPLE_GAP)
        self.dataset_test.extend(test_samples)

        self.loader_train = DataLoader(
            dataset=list(self.dataset_train),
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=None,
        )

        self.loader_test = DataLoader(
            dataset=list(self.dataset_test),
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=None,
        )


class MobServer(BaseServer, MobClient):
    def __init__(self, id, args, clients):
        super().__init__(id, args, clients)

        self.position = np.array([
            args.server_lon if hasattr(args, 'server_lon') else 116.41667,
            args.server_lat if hasattr(args, 'server_lat') else 39.91667
        ])
        self.communication_range = args.comm_range
        self.DELTA_TIME = args.delta_time

        self.regions = init_regions(args, self.position)

    def sample(self):
        available_clients = [c for c in self.clients if c.is_covered]

        if len(available_clients) == 0:
            self.sampled_clients = []
        else:
            sample_num = min(int(self.sample_rate * self.client_num), len(available_clients))
            self.sampled_clients = sorted(random.sample(available_clients, sample_num), key=lambda x: x.id)

    def downlink(self):
        super().downlink()

    def client_update(self):
        for client in self.sampled_clients:
            assert client.is_covered
            client.model.train()
            client.reset_optimizer()
            client.run()

    def update_global_state(self):
        # 1. update wall clock time
        self.wall_clock_time += self.DELTA_TIME

        # 2. update client position
        for client in self.clients:
            # 2-1. update position
            client.update_position(self.wall_clock_time)
            client.is_covered = np.linalg.norm(
                np.array(client.current_pos) - np.array(self.position)
            ) <= self.communication_range

            # 2-2. check region update
            client.current_region = None
            for region in self.regions:
                x, y = client.current_pos
                xmin, xmax, ymin, ymax = region.bounds
                if xmin <= x <= xmax and ymin <= y <= ymax:
                    client.current_region = region

            # 2-3. load dataset
            if client.current_region is not None: client.load_dataset()

    # NOTE: the client not covered will not upload parameters
    def uplink(self):
        assert (len(self.sampled_clients) > 0)
        def nan_to_zero(tensor):
            return torch.where(torch.isnan(tensor), torch.zeros_like(tensor), tensor)
        self.received_params = [nan_to_zero(client.model2tensor()) for client in self.sampled_clients if client.is_covered]

    # NOTE: the client not covered will not upload parameters
    def aggregate(self):
        assert (len(self.sampled_clients) > 0)
        uploaded_clients = [client for client in self.sampled_clients if client.is_covered]
        total_samples = sum(len(client.dataset_train) for client in uploaded_clients)
        weights = [len(client.dataset_train) / total_samples for client in uploaded_clients]

        self.received_params = [params * weight for weight, params in zip(weights, self.received_params)]
        avg_tensor = sum(self.received_params)
        self.tensor2model(avg_tensor)

def init_regions(args, position):
    regions = []
    x, y = position
    r = args.grid_range
    xmin, xmax, ymin, ymax = x - r, x + r, y - r, y + r

    cols, rows = 4, 4
    lon_step = (xmax - xmin) / cols
    lat_step = (ymax - ymin) / rows

    region_id = 0
    for i in range(cols):
        for j in range(rows):
            bounds = (
                xmin + i * lon_step, xmin + (i + 1) * lon_step,
                ymin + j * lat_step, ymin + (j + 1) * lat_step
            )
            regions.append(Region(args, region_id, bounds))
            region_id += 1
    return regions

class Region:
    def __init__(self, args, region_id, bounds):
        self.id = region_id
        self.bounds = bounds
        self.dataset_train = read_client_data(args.dataset, self.id, is_train=True)
        self.dataset_test = read_client_data(args.dataset, self.id, is_train=False)