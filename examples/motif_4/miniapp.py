import argparse, asyncio, os, time, json
from pathlib import Path

from stages.sim_stage import SimulationStage
from stages.assimilate_stage import AssimilationStage
from stages.train_stage import TrainingStage
from stages.infer_stage import InferenceStage
from stages.control_stage import ControlStage
from utils.metrics import Timer, log
import yaml

async def run_pipeline(cfg):
    q_sim_to_assim = asyncio.Queue(maxsize=cfg["queues"]["sim_to_assim"])
    q_assim_to_train = asyncio.Queue(maxsize=cfg["queues"]["assim_to_train"])
    q_assim_to_infer = asyncio.Queue(maxsize=cfg["queues"]["assim_to_infer"])
    q_train_to_infer = asyncio.Queue(maxsize=cfg["queues"]["train_to_infer"])
    q_infer_to_control = asyncio.Queue(maxsize=cfg["queues"]["infer_to_control"])
    q_control_to_sim = asyncio.Queue(maxsize=cfg["queues"]["control_to_sim"])

    # Stages
    sim = SimulationStage(cfg)
    assim = AssimilationStage(cfg)
    train = TrainingStage(cfg)
    infer = InferenceStage(cfg)
    control = ControlStage(cfg)

    # Tasks (concurrent, real-time style)
    tasks = [
        asyncio.create_task(sim.run(q_out=q_sim_to_assim, q_in=q_control_to_sim)),
        asyncio.create_task(assim.run(q_in=q_sim_to_assim,
                                      q_out_train=q_assim_to_train,
                                      q_out_infer=q_assim_to_infer)),
        asyncio.create_task(train.run(q_in=q_assim_to_train, q_out=q_train_to_infer)),
        asyncio.create_task(infer.run(q_state=q_assim_to_infer,
                                      q_model=q_train_to_infer,
                                      q_out=q_infer_to_control)),
        asyncio.create_task(control.run(q_in=q_infer_to_control, q_out=q_control_to_sim)),
    ]

    await asyncio.gather(*tasks)

def load_cfg(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Digital Twin (Motif 4) Mini-App")
    ap.add_argument("--config", type=str, default="config.yaml")
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    t = Timer().start()
    asyncio.run(run_pipeline(cfg))
    log(f"DONE in {t.stop():.3f}s")
