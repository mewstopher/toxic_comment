import torch
import numpy as np

class ToxicEvaluation(object):

    def __init__(self, model, test_dataloader, device=device):
        self.model = model
        self.dataloader = test_dataloader
        self.device = device

        self.test_data, self.predictions = self._predict_test()

    def _predict_test(self):
        with torch.no_grad():
            dataiter = iter(self.dataloader)
            data_sample = dataiter.next()

            for i in range(len(data_sample)):
                data_sample[i] = data_sample[i].to(self.device)

            out = self.model(data_sample[0])

        return data_sample, out

