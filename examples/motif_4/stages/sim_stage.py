import asyncio, time, math, random
from wfMiniAPI import kernel as kern
from utils.metrics import log

class SimulationStage:
    """
    Emulates a time-stepping simulation producing state snapshots.
    Uses matMulSimple2D + read/write emulation to capture CPU/GPU + I/O.
    Accepts control updates (e.g., parameter steering) from the controller.
    """
    def __init__(self, cfg):
        sim = cfg["simulation"]
        self.steps = sim["steps"]
        self.state_dim = sim["state_dim"]
        self.read_size = sim["read_size_bytes"]
        self.write_size = sim["write_size_bytes"]
        self.device = sim["device"]
        self.sleep_ms = sim.get("sleep_ms", 0) 

    async def run(self, q_out, q_in):
        for step in range(self.steps):
            # apply pending control update (non-blocking)
            ctrl = None
            try:
                ctrl = q_in.get_nowait()
            except asyncio.QueueEmpty:
                pass

            # read/input emulate
            kern.readNonMPI(nbytes=self.read_size)

            # compute: simulate main step
            kern.RNG(device=self.device, data_size=self.state_dim)
            kern.matMulSimple2D(device=self.device, dim=self.state_dim)

            # write/output emulate
            kern.writeNonMPI(nbytes=self.write_size)

            # emit "state" token (size metadata; payload omitted for lightness)
            msg = {"t": step, "state_dim": self.state_dim, "ctrl": ctrl}
            await q_out.put(msg)

            if self.sleep_ms > 0:
                await asyncio.sleep(self.sleep_ms / 1000.0)

        # signal completion
        await q_out.put(None)
        log("SimulationStage: complete")
