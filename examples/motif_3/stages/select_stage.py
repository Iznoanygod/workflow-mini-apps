import asyncio, heapq
from wfMiniAPI import kernel as kern
from utils.metrics import log

class SelectStage:
    """
    Maintains top-K designs and periodically sends feedback to the proposer.
    Consumes (a) evaluation results and (b) model updates (optional).
    """
    def __init__(self, cfg):
        s = cfg["select"]
        self.top_k = s["top_k"]
        self.emit_every = s.get("emit_every", 1)
        self.device = s["device"]
        self.policy_cost = s.get("policy_cost", 256 * 256)

        # min-heap storing (-score, rid, cid) so we keep best (lowest objective)
        self.heap = []

    def _push(self, res):
        score = res["objective"]
        item = (score, res["rid"], res["cid"])
        if len(self.heap) < self.top_k:
            heapq.heappush(self.heap, (-score, item))
        else:
            worst = self.heap[0]
            if -score > worst[0]:
                heapq.heapreplace(self.heap, (-score, item))

    async def run(self, q_in_results, q_in_model, q_out_fb):
        # model updates are optional; listen asynchronously
        async def _model_listener():
            while True:
                m = await q_in_model.get()
                if m is None: return
                # tiny compute to emulate re-scoring policy
                kern.matMulSimple2D(device=self.device,
                                    dim=int(self.policy_cost ** 0.5))

        listener = asyncio.create_task(_model_listener())

        n = 0
        while True:
            res = await q_in_results.get()
            if res is None:
                await q_out_fb.put(None)   # tell proposer we’re done
                listener.cancel()
                log("SelectStage: complete")
                return

            self._push(res)

            if n % self.emit_every == 0 and self.heap:
                # Emit feedback with current best
                best_neg, (score, rid, cid) = max(self.heap), None
                # heap stores (-score, item); invert again:
                best = max(self.heap, key=lambda x: x[0])
                _, (top_score, top_rid, top_cid) = best
                fb = {"best": {"rid": top_rid, "cid": top_cid,
                               "objective": top_score}}
                await q_out_fb.put(fb)
            n += 1