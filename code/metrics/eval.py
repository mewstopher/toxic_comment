import torch
import numpy as np

class ToxicEvaluation(object):

    def __init__(self, model, dataloader, device):
        self.model = model
        self.dataloader = test_dataloader
        self.device = device

        self.predicted, self.accuracy = self._predict_val()

    def _predict_val(self):
        with torch.no_grad():
            for data in self.dataloader:
                for i in range(len(data)):
                    data[i] = data[i].to(self.device)
                input_data, labels = data
                val_out = model(input_data)
                _, predicted = torch.max(val_out.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return predicted, correct/total

