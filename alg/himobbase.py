import importlib
import torch
import random

from alg.mobbase import MobClient, MobServer

class HiMobClient(MobClient):
    def __init__(self, id, args):
        super().__init__(id, args)


class HiMobEdgeServer(MobServer):
    def __init__(self, id, args, clients, region):
        super().__init__(id, args, clients)
        self.args = args
        self.region = region
        self.edge_round = args.edge_rnd
        self.covered_clients = []

    def run(self):
        for _ in range(self.edge_round):
            self.downlink()
            self.client_update()
            self.uplink()
            self.aggregate()

    def downlink(self):
        if len(self.sampled_clients) == 0: return
        super().downlink()

    def client_update(self):
        if len(self.sampled_clients) == 0: return
        super().client_update()

    # NOTE: the client may move to other regions
    def uplink(self):
        if len(self.sampled_clients) == 0: return

        def nan_to_zero(tensor):
            return torch.where(torch.isnan(tensor), torch.zeros_like(tensor), tensor)

        self.received_params = [nan_to_zero(client.model2tensor())
                                for client in self.sampled_clients
                                if client.current_region == self.region]

    # NOTE: the client may move to other regions
    def aggregate(self):
        assert (len(self.sampled_clients) > 0)
        uploaded_clients = [client for client in self.sampled_clients if client.current_region == self.region]
        total_samples = sum(len(client.dataset_train) for client in uploaded_clients)
        weights = [len(client.dataset_train) / total_samples for client in uploaded_clients]

        self.received_params = [params * weight for weight, params in zip(weights, self.received_params)]
        avg_tensor = sum(self.received_params)
        self.tensor2model(avg_tensor)

class HiMobServer(MobServer, HiMobClient):
    def __init__(self, id, args, clients):
        super().__init__(id, args, clients)
        alg_module = importlib.import_module(f'alg.{args.alg}')
        EdgeClass = getattr(alg_module, 'EdgeServer', HiMobEdgeServer)
        self.edges = [EdgeClass(e_id, args, clients, region)
                      for e_id, region in enumerate(self.regions)]

        self.active_edges = []
        self.edge_sample_rate = args.esr


    def sample_edges(self):
        sample_num = min(int(self.edge_sample_rate * len(self.edges)), len(self.edges))
        while True:
            self.active_edges = sorted(random.sample(self.edges, sample_num), key=lambda x: x.id)
            available_clients = [c for c in self.clients
                                 if c.current_region in [edge.region for edge in self.active_edges]]
            if len(available_clients) > 0: break

    def sample(self):
        available_clients = [c for c in self.clients
                             if c.current_region in [edge.region for edge in self.active_edges]]

        sample_num = min(int(self.sample_rate * self.client_num), len(available_clients))
        self.sampled_clients = sorted(random.sample(available_clients, sample_num), key=lambda x: x.id)

        for edge in self.edges:
            edge.sampled_clients = self.sampled_clients
        for client in self.sampled_clients:
            sampled_clients = self.edges[client.current_region.id].sampled_clients
            sampled_clients.append(client)

    def downlink(self):
        any(edge.clone_model(self) for edge in self.active_edges)

    def edge_update(self):
        any(edge.run() for edge in self.active_edges)

    def uplink(self):
        def nan_to_zero(tensor):
            return torch.where(torch.isnan(tensor), torch.zeros_like(tensor), tensor)

        self.received_params = [nan_to_zero(edge.model2tensor()) for edge in self.active_edges]

    def aggregate(self):
        assert (len(self.sampled_clients) > 0)
        total_samples = sum(len(edge.dataset_train) for edge in self.active_edges)
        weights = [len(edge.dataset_train) / total_samples for edge in self.active_edges]

        self.received_params = [params * weight for weight, params in zip(weights, self.received_params)]
        avg_tensor = sum(self.received_params)
        self.tensor2model(avg_tensor)

    def update_global_state(self):
        # 1. update wall clock time
        self.wall_clock_time += self.DELTA_TIME

        # 2. update client position
        for client in self.clients:
            # 2-1. update position
            client.update_position(self.wall_clock_time)

            # 2-2. check region update
            client.current_region = None
            for region in self.regions:
                x, y = client.current_pos
                xmin, xmax, ymin, ymax = region.bounds
                if xmin <= x <= xmax and ymin <= y <= ymax:
                    client.current_region = region

            # 2-3. load dataset
            if client.current_region is not None: client.load_dataset()