import asyncio
from wfMiniAPI import kernel as kern
from utils.metrics import log

class InferenceStage:
    """
    Online inference using the surrogate.
    Models memory copies + axpy-like compute.
    """
    def __init__(self, cfg):
        inf = cfg["inference"]
        self.device = inf["device"]
        self.copy_bytes = inf.get("copy_bytes", 2 * 1024 * 1024)
        self.vec_len = inf.get("vec_len", 1_000_000)

    async def run(self, q_state, q_model, q_out):
        current_model = {"version": -1, "dim": 1}

        async def model_updater():
            nonlocal current_model
            while True:
                m = await q_model.get()
                if m is None:
                    return
                current_model = m

        updater_task = asyncio.create_task(model_updater())

        while True:
            s = await q_state.get()
            if s is None:
                await q_out.put(None)
                updater_task.cancel()
                log("InferenceStage: complete")
                return

            # emulate H2D/D2H copies around inference
            kern.dataCopyH2D(data_size=self.copy_bytes)
            kern.axpy(device=self.device, data_size=self.vec_len)  # y = a*x + y (emulated)
            kern.dataCopyD2H(data_size=self.copy_bytes)

            pred = {"t": s["t"], "model_ver": current_model["version"], "health": 1.0}
            await q_out.put(pred)
