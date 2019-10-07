import torch
import numpy as np

class ToxicEvaluation(object):

    def __init__(self, model, dataloader, device, nb_classes):
        self.model = model
        self.dataloader = dataloader
        self.device = device

        self.predicted, self.accuracy , self.confusion_matrix = self._predict_val()

    def _predict_val(self):
         confusion_matrix = torch.zeros(nb_classes, nb_classes)
        with torch.no_grad():
            for data in self.dataloader:
                for i in range(len(data)):
                    data[i] = data[i].to(self.device)
                input_data, labels = data
                val_out = model(input_data)
                predicted = torch.round(val_out.data)
                total += labels.size(0)*labels.size(1)
                correct += (predicted == labels).sum().item()
                for t, [ in zip(classes.view(-1), preds.view(-1)):
                        confusion_matrix[t.long(), p.long()] += 1
        return predicted, correct/total, confusion_matrix

    def _confusion_matrix(self):
        confusion_matrix = torch.zeros(nb_classes, nb_classes)
        with torch.no_grad():
            for i, (inputs, classes) in enumerate(dataloader):
                inputs = inputs.to(device)
                classes = classes.to(device)
                outputs = model_ft(inputs)
                _, preds = torch.max(outputs, 1)
                for t, p in zip(classes.view(-1), preds.view(-1)):
                    confusion_matrix[t.long(), p.long()] += 1

        return confusion_matrix
