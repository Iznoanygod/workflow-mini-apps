import asyncio, math, random
from wfMiniAPI import kernel as kern
from utils.metrics import log

class EvaluateStage:
    """
    Embarrassingly-parallel evaluation pool:
    - read/write emulation (I/O)
    - main compute via matMulSimple2D (dominant)
    Produces result records sent to both training and selection branches.
    """
    def __init__(self, cfg):
        e = cfg["evaluate"]
        self.device = e["device"]
        self.read_size = e["read_size_bytes"]
        self.write_size = e["write_size_bytes"]
        self.matmul_dim = e["matmul_dim"]
        self.workers = e.get("workers", 4)

    async def worker(self, name, q_in, q_out_train, q_out_select):
        while True:
            cand = await q_in.get()
            if cand is None:
                # propagate sentinel to other workers and downstream
                await q_in.put(None)
                await q_out_train.put(None)
                await q_out_select.put(None)
                log(f"EvaluateStage[{name}]: complete")
                return

            # emulate evaluation
            kern.readNonMPI(nbytes=self.read_size)
            kern.RNG(device=self.device, data_size=cand["vec_len"])
            kern.matMulSimple2D(device=self.device, dim=self.matmul_dim)
            kern.writeNonMPI(nbytes=self.write_size)

            # fake objective: lower is better
            obj = 1.0 / (1.0 + self.matmul_dim) + random.random() * 0.01
            res = {"rid": cand["rid"], "cid": cand["cid"], "objective": obj}

            await q_out_train.put(res)
            await q_out_select.put(res)

    async def run(self, q_in, q_out_train, q_out_select):
        tasks = [asyncio.create_task(
                    self.worker(f"W{i}", q_in, q_out_train, q_out_select))
                 for i in range(self.workers)]
        await asyncio.gather(*tasks)