import asyncio, random
from wfMiniAPI import kernel as kern
from utils.metrics import log

class ProposeStage:
    """
    Proposes candidate designs in rounds; consumes feedback to bias proposals.
    Emulates proposal compute & data staging via RNG + H2D/D2H copies.
    """
    def __init__(self, cfg):
        p = cfg["propose"]
        self.rounds = p["rounds"]
        self.batch = p["batch_size"]
        self.device = p["device"]
        self.vec_len = p.get("vec_len", 1_000_000)
        self.copy_bytes = p.get("copy_bytes", 2 * 1024 * 1024)

    async def run(self, q_out, q_fb):
        for r in range(self.rounds):
            # consume *latest* feedback if available (non-blocking)
            fb_latest = None
            try:
                while True:
                    fb_latest = q_fb.get_nowait()
                    if fb_latest is None:
                        # early termination signal
                        await q_out.put(None)
                        log("ProposeStage: early stop")
                        return
            except asyncio.QueueEmpty:
                pass

            for i in range(self.batch):
                # emulate cheap generator compute+copies
                kern.RNG(device=self.device, data_size=self.vec_len)
                kern.dataCopyH2D(data_size=self.copy_bytes)
                kern.dataCopyD2H(data_size=self.copy_bytes)
                cand = {
                    "rid": r, "cid": i,
                    "params": {"seed": random.random()},
                    "vec_len": self.vec_len
                }
                await q_out.put(cand)

        # finished proposing
        await q_out.put(None)
        log("ProposeStage: complete")