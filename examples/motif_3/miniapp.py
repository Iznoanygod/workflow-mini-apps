import argparse, asyncio, yaml
from stages.propose_stage import ProposeStage
from stages.evaluate_stage import EvaluateStage
from stages.train_stage import TrainingStage
from stages.select_stage import SelectStage
from utils.metrics import Timer, log

async def run_pipeline(cfg):
    # Queues wire the inverse-design loop
    q_candidates = asyncio.Queue(maxsize=cfg["queues"]["candidates"])
    q_results_train = asyncio.Queue(maxsize=cfg["queues"]["results"])
    q_results_select = asyncio.Queue(maxsize=cfg["queues"]["results"])
    q_model_updates = asyncio.Queue(maxsize=cfg["queues"]["models"])
    q_feedback = asyncio.Queue(maxsize=cfg["queues"]["feedback"])

    propose = ProposeStage(cfg)
    evaluate = EvaluateStage(cfg)
    train = TrainingStage(cfg)
    select = SelectStage(cfg)

    tasks = [
        asyncio.create_task(propose.run(q_out=q_candidates, q_fb=q_feedback)),
        asyncio.create_task(evaluate.run(q_in=q_candidates,
                                         q_out_train=q_results_train,
                                         q_out_select=q_results_select)),
        asyncio.create_task(train.run(q_in=q_results_train, q_out=q_model_updates)),
        asyncio.create_task(select.run(q_in_results=q_results_select,
                                       q_in_model=q_model_updates,
                                       q_out_fb=q_feedback)),
    ]
    await asyncio.gather(*tasks)

def load_cfg(path): 
    with open(path, "r") as f: return yaml.safe_load(f)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Inverse Design (Motif 3) Mini-App")
    ap.add_argument("--config", type=str, default="config.yaml")
    args = ap.parse_args()
    cfg = load_cfg(args.config)
    t = Timer().start()
    asyncio.run(run_pipeline(cfg))
    log(f"DONE in {t.stop():.3f}s")