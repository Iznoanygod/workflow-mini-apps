import argparse, asyncio, yaml, random, logging
from utils.metrics import Timer, log

from radical.asyncflow import WorkflowEngine
from radical.asyncflow import DragonExecutionBackend
from radical.asyncflow.logging import init_default_logger

from wfMiniAPI import kernel as kern


from concurrent.futures import ThreadPoolExecutor

def load_cfg(path): 
    with open(path, "r") as f: return yaml.safe_load(f)

async def workflow(cfg):
    logger = logging.getLogger(__name__)
    init_default_logger(logging.DEBUG)
    backend = await DragonExecutionBackend()
    init_default_logger(logging.DEBUG)
    flow = await WorkflowEngine.create(backend=backend)

    @flow.function_task
    async def propose():
        p = cfg["propose"]
        device = p["device"]
        vec_len = p.get("vec_len", 1_000_000)
        copy_bytes = p.get("copy_bytes", 2 * 1024 * 1024)
        kern.RNG(device=device, data_size=vec_len)
        kern.dataCopyH2D(data_size=copy_bytes)
        kern.dataCopyD2H(data_size=copy_bytes)
        cand = {
            "params": {"seed": random.random()},
            "vec_len": vec_len
        }
        return cand
    
    @flow.function_task
    async def evaluate(cand):
        e = cfg["evaluate"]
        device = e["device"]
        read_size = e["read_size_bytes"]
        write_size = e["write_size_bytes"]
        matmul_dim = e["matmul_dim"]
        kern.readNonMPI(nbytes=read_size)
        kern.RNG(device=device, data_size=cand["vec_len"])
        kern.matMulSimple2D(device=device, dim=matmul_dim)
        kern.writeNonMPI(nbytes=write_size)

        obj = 1.0 / (1.0 + matmul_dim) + random.random() * 0.01
        res = {"objective": obj}
        return res
    
    @flow.function_task
    async def train(res, k):
        tr = cfg["training"]
        device = tr["device"]
        model_dim = tr["model_dim"]
        use_collective = tr.get("use_collective", False)
        collective_bytes = tr.get("collective_bytes", 4 * 1024 * 1024)

        kern.matMulGeneral(device=device,
                           dim_list=[model_dim, model_dim, model_dim])
        if use_collective:
            kern.MPIallReduce(device=device, data_size=collective_bytes)
        model_update = {"version": k, "dim": model_dim}
        return model_update

    propose_t = propose()
    evaluate_t = evaluate(propose_t)
    train_t = await train(evaluate_t, k=1)

    await flow.shutdown()

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Inverse Design (Motif 3) Mini-App")
    ap.add_argument("--config", type=str, default="config.yaml")
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    t = Timer().start()
    asyncio.run(workflow(cfg))
    log(f"DONE in {t.stop():.3f}s")