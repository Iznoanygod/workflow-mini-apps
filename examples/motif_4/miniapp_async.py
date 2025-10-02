import argparse, asyncio, yaml, random
from utils.metrics import Timer, log

from radical.asyncflow import WorkflowEngine
from radical.asyncflow import ConcurrentExecutionBackend

from wfMiniAPI import kernel as kern

from concurrent.futures import ThreadPoolExecutor

def load_cfg(path): 
    with open(path, "r") as f: return yaml.safe_load(f)

async def workflow(cfg):
    backend = await ConcurrentExecutionBackend(ThreadPoolExecutor())
    flow = await WorkflowEngine.create(backend=backend)

    @flow.function_task
    async def simulation():
        sim = cfg["simulation"]
        steps = sim["steps"]
        state_dim = sim["state_dim"]
        read_size = sim["read_size_bytes"]
        write_size = sim["write_size_bytes"]
        device = sim["device"]
        sleep_ms = sim.get("sleep_ms", 0) 
        kern.readNonMPI(nbytes=read_size)

        kern.RNG(device=device, data_size=state_dim)
        kern.matMulSimple2D(device=device, dim=state_dim)

        kern.writeNonMPI(nbytes=write_size)

        msg = {"state_dim": state_dim}

        if sleep_ms > 0:
            await asyncio.sleep(sleep_ms / 1000.0)
        return msg

    @flow.function_task
    async def assimilation(state):
        assim = cfg["assimilation"]
        read_size = assim["read_size_bytes"]
        write_size = assim["write_size_bytes"]
        device = assim["device"]
        kern.readNonMPI(nbytes=read_size)
        kern.RNG(device=device, data_size=state["state_dim"])
        kern.matMulSimple2D(device=device, dim=state["state_dim"])
        kern.writeNonMPI(nbytes=write_size)
        msg = {"t": state["t"], "state_dim": state["state_dim"]}
        return msg
    
    @flow.function_task
    async def training(state):
        train = cfg["training"]
        read_size = train["read_size_bytes"]
        write_size = train["write_size_bytes"]
        device = train["device"]
        kern.readNonMPI(nbytes=read_size)
        kern.RNG(device=device, data_size=state["state_dim"])
        kern.matMulSimple2D(device=device, dim=state["state_dim"])
        kern.writeNonMPI(nbytes=write_size)
        msg = {"t": state["t"], "model_dim": state["state_dim"] // 2}
        return msg

    @flow.function_task
    async def inference(state, model):
        infer = cfg["inference"]
        read_size = infer["read_size_bytes"]
        write_size = infer["write_size_bytes"]
        device = infer["device"]
        kern.readNonMPI(nbytes=read_size)
        kern.RNG(device=device, data_size=state["state_dim"])
        kern.matMulSimple2D(device=device, dim=state["state_dim"])
        kern.matMulSimple2D(device=device, dim=model["model_dim"])
        kern.writeNonMPI(nbytes=write_size)
        msg = {"t": state["t"], "state_dim": state["state_dim"]}
        return msg

    @flow.function_task
    async def control(state):
        ctrl = cfg["control"]
        read_size = ctrl["read_size_bytes"]
        write_size = ctrl["write_size_bytes"]
        device = ctrl["device"]
        kern.readNonMPI(nbytes=read_size)
        kern.RNG(device=device, data_size=state["state_dim"])
        kern.matMulSimple2D(device=device, dim=state["state_dim"])
        kern.writeNonMPI(nbytes=write_size)
        msg = {"t": state["t"], "ctrl_param": random.random()}
        return msg
    
    sim_t = simulation()
    assim_t = assimilation(sim_t)
    train_t = training(assim_t)
    infer_t = inference(assim_t, train_t)
    control_t = await control(infer_t)

    await flow.shutdown()

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Digital Twin (Motif 4) Mini-App")
    ap.add_argument("--config", type=str, default="config.yaml")
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    t = Timer().start()
    asyncio.run(workflow(cfg))
    log(f"DONE in {t.stop():.3f}s")
