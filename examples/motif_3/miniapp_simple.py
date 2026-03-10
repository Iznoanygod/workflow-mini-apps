import argparse, asyncio, yaml, random, logging
import time
from utils.metrics import Timer, log

from radical.asyncflow import WorkflowEngine
from radical.asyncflow import DragonExecutionBackend
from radical.asyncflow.logging import init_default_logger

from wfMiniAPI import kernel as kern

def load_cfg(path): 
    with open(path, "r") as f: return yaml.safe_load(f)

async def workflow(cfg):
    logger = logging.getLogger(__name__)
    init_default_logger(logging.DEBUG)
    backend = await DragonExecutionBackend()
    init_default_logger(logging.DEBUG)
    flow = await WorkflowEngine.create(backend=backend)
    
    @flow.function_task
    async def simulate():
        e = cfg["simulate"]
        device = e["device"]
        read_size = e["read_size_bytes"]
        write_size = e["write_size_bytes"]
        matmul_dim = e["matmul_dim"]

        log(f"Simulating candidate with params...")
        log(time.strftime("%H:%M:%S", time.localtime()))
        kern.generateRandomNumber(device=device, size=matmul_dim)
        for i in range(8):
            kern.matMulSimple2D(device=device, size=matmul_dim)
        kern.writeNonMPI(num_bytes=write_size, data_root_dir="./")
        kern.readNonMPI(num_bytes=read_size, data_root_dir="./")
        log(f"Finished simulating candidate with params...")
        log(time.strftime("%H:%M:%S", time.localtime()))
        return random.random()

    @flow.function_task
    async def train(res):
        e = cfg["training"]
        device = e["device"]
        read_size = e["read_size_bytes"]
        write_size = e["write_size_bytes"]
        copy_size = e["data_copy_size_bytes"]
        matmul_dim = e["matmul_dim"]

        log(f"Training model with evaluation results...")
        log(time.strftime("%H:%M:%S", time.localtime()))
        kern.readNonMPI(num_bytes=read_size, data_root_dir="./")
        kern.dataCopyH2D(data_size=copy_size)
        for i in range(6):
            kern.matMulSimple2D(device="gpu", size=matmul_dim)
            kern.matMulSimple2D(device="cpu", size=matmul_dim)
        kern.dataCopyH2D(data_size=copy_size)
        kern.writeNonMPI(num_bytes=write_size, data_root_dir="./")
        log(f"Finished training model with evaluation results...")
        log(time.strftime("%H:%M:%S", time.localtime()))
        return random.random()

    for i in range(3):
        simulate_t= simulate()
        train_t = await train(simulate_t)

    await flow.shutdown()

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Inverse Design (Motif 3) Mini-App")
    ap.add_argument("--config", type=str, default="config.yaml")
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    t = Timer().start()
    asyncio.run(workflow(cfg))
    log(f"DONE in {t.stop():.3f}s")