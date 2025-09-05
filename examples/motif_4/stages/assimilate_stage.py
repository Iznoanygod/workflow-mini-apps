import asyncio, collections
from wfMiniAPI import kernel as kern
from utils.metrics import log

class AssimilationStage:
    """
    Windowed assimilation (e.g., reductions / filtering).
    Emits to both training (for model updates) and inference (for online predictions).
    """
    def __init__(self, cfg):
        a = cfg["assimilation"]
        self.window = a["window"]
        self.device = a["device"]
        self.buf = collections.deque(maxlen=self.window)
        self.emit_every = a.get("emit_every", 1)

    async def run(self, q_in, q_out_train, q_out_infer):
        n = 0
        while True:
            msg = await q_in.get()
            if msg is None:
                await q_out_train.put(None)
                await q_out_infer.put(None)
                log("AssimilationStage: complete")
                return

            self.buf.append(msg["state_dim"])
            data_size = max(1, sum(self.buf))  # scale only
            kern.reduction(device=self.device, data_size=data_size)

            if n % self.emit_every == 0:
                state_summary = {"t": msg["t"], "feat": data_size}
                await q_out_train.put(state_summary)
                await q_out_infer.put(state_summary)
            n += 1
