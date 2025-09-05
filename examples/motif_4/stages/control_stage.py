import asyncio, math
from wfMiniAPI import kernel as kern
from utils.metrics import log

class ControlStage:
    """
    Takes predictions and computes control signals to steer the sim.
    """
    def __init__(self, cfg):
        c = cfg["control"]
        self.device = c["device"]
        self.policy_cost = c.get("policy_cost", 256*256)

    async def run(self, q_in, q_out):
        while True:
            pred = await q_in.get()
            if pred is None:
                await q_out.put(None)
                log("ControlStage: complete")
                return
            # emulate tiny policy compute
            dim = int(math.sqrt(self.policy_cost))
            kern.matMulSimple2D(device=self.device, dim=dim)
            ctrl = {"gain": 0.01, "model_ver": pred["model_ver"], "t": pred["t"]}
            await q_out.put(ctrl)
