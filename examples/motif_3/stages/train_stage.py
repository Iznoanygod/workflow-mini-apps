import asyncio
from wfMiniAPI import kernel as kern
from utils.metrics import log

class TrainingStage:
    """
    Optional surrogate training on a stream of evaluation results.
    - Compute via matMulGeneral
    - Optional MPIallReduce to emulate DP sync
    Emits lightweight model-version updates.
    """
    def __init__(self, cfg):
        tr = cfg["training"]
        self.device = tr["device"]
        self.model_dim = tr["model_dim"]
        self.train_every = tr["train_every"]
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
                kern.matMulGeneral(device=self.device,
                                   dim_list=[self.model_dim,
                                             self.model_dim,
                                             self.model_dim])
                if self.use_collective:
                    kern.MPIallReduce(device=self.device,
                                      data_size=self.collective_bytes)
                await q_out.put({"version": k, "dim": self.model_dim})
            k += 1