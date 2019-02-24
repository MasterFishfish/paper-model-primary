
class Samplepack():
    def __init__(self):
        self.samples = []
        self.id2sample = {}

    def init_id2sample(self):
        if self.samples is None:
            raise Exception("Samples is None.", self.samples)
        for sample in self.samples:
            self.id2sample[sample.id] = sample

    def pack_preds(self, preds, ids):
        # preds 和 ids 是 list
        for i in range(len(ids)):
            self.id2sample[ids[i]].preds = preds[i]

    def update_best(self):
        for sample in self.samples:
            sample.best_pred = sample.pred
