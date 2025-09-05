import asyncio
from wfMiniAPI import kernel as kern
from utils.metrics import log

class TrainingStage:
    """
    Periodic surrogate training step.
    Uses matMulGeneral + (optional) MPIallReduce to emulate data-parallel training.
    """
    def __init__(self, cfg):
        tr = cfg["training"]
        self.device = tr["device"]
        self.train_every = tr["train_every"]
        self.model_dim = tr["model_dim"]
        self.use_collective = tr.get("use_collective", False)
        self.collective_bytes = tr.get("collective_bytes", 4 * 1024 * 1024)

    async def run(self, q_in, q_out):
        k = 0
        while True:
            msg = await q_in.get()
            if msg is None:
                await q_out.put(None)
                log("TrainingStage: complete")
                return
            if k % self.train_every == 0:
                # emulate backprop compute
                kern.matMulGeneral(device=self.device, dim_list=[self.model_dim,
                                                                self.model_dim,
                                                                self.model_dim])
                # emulate gradient allreduce
                if self.use_collective:
                    kern.MPIallReduce(device=self.device, data_size=self.collective_bytes)
                model = {"version": k, "dim": self.model_dim}
                await q_out.put(model)
            k += 1
