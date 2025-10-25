from alg.himobbase import HiMobClient, HiMobServer

class Client(HiMobClient):
    def run(self):
        self.train()

class Server(HiMobServer):
    def run(self):
        self.sample_edges()
        self.sample()
        self.downlink()
        self.edge_update()
        self.update_global_state()
        self.uplink()
        self.aggregate()