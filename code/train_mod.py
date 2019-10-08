from imports import *
from metrics.eval import ToxicEvaluation
from helper_functions import speak

class Trainer:
    accuracies = {}
    losses = {}

    def __init__(self, model, train_dataloader, val_dataloader, num_epochs, Loss, model_path=None,
                 save_path=None, save=False, load_model=False):
        self.model = model
        self.save_choice(save, save_path)
        if load_model:
            self.model.load_state_dict(torch.load(model_path))
            self.model.train()
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.num_epochs = num_epochs
        self.Loss = Loss
        self.save = save
        self.save_path = save_path

    def save_choice(self, save, save_path):
        if not save:
            print("you have chosen not to save the model")
        elif save and save_path:
            print("saving model at: {}".format(save_path))

    def train(self):
        print("begining to train the machine")
        count = 0
        total = 0
        correct = 0
        for epoch in range(self.num_epochs):
            self.losses[epoch] = []
            self.accuracies[epoch] = []
            for data_sample in self.train_dataloader:
                for i in range(len(data_sample)):
                    data_sample[i] = data_sample[i].to(device)

                count += 1
                model.zero_grad()


                out = self.model(data_sample[0])
                loss = self.Loss(out, data_sample[1].float())

                loss.backward()
                optimizer.step()
                self.losses[epoch].append(loss.item())

                predicted = torch.round(out)
                total += data_sample[1].size(0) * data_sample[1].size(1)
                correct += (predicted == data_sample[1]).sum().item()
                accuracy = correct/total
                self.accuracies[epoch].append(accuracy)

                if count % 50 == 0:
                    print("loss: {} (at iteration {})".format(np.mean(losses[epoch]), count))
                    print("accuracy: {} (at iteration {})".format(np.mean(accuracies[epoch]), count))
            toxic_eval = ToxicEvaluation(model, val_dataloader, device)
            val_accuracy = toxic_eval.correct/toxic_eval.total
            print("accuracy for validation set: {}".format(val_accuracy))

            print("Average training loss for epoch {}: {}".format(epoch, np.mean(losses[epoch])))
            if save:
                torch.save(self.model.state_dict(), save_path)

    def print_losses(self):
        loss_list = [i for epoch in self.losses for i in self.losses[epoch]]
        return plt.plot(loss_list)

    def plot_accuracies(self):
        accuracy_list = [i for epoch in self.accuracies for i in self.accuracies[epoch]]
        return plt.plot(accuracy_list)

    def run(self):
        try:
            self.train()
        except KeyboardInterrupt:
            print('\n' * 8)
            speak()
            if self.save:
                torch.save(self.model.state_dict(), save_path + str(datetime.now()).split()[0])

