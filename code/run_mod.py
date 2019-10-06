from imports import *
from dataset import ToxicDataset
from model import LstmNet
from metrics.eval import ToxicEvaluation
# define Paths to datafile and embeddings
TOXIC_CSV_PATH = "../input/datasets/train.csv"
GLOVE_PATH = "../input/glove_embeddings/Embeddings/glove.6B.50d.txt"

# Initialize Dataset class
toxic_dataset = ToxicDataset(TOXIC_CSV_PATH, GLOVE_PATH)

batch_size = 64
nb_lstm_units = 64
random_seed = 42

val_split = 0.1
test_split = 0.1
shuffle_dataset = False

train_sampler, val_sampler, test_sampler = train_test_sampler(toxic_dataset, .8, .1, .1)

# Creating PT data samplers and loaders:

train_dataloader = DataLoader(toxic_dataset, batch_size, sampler=train_sampler)
val_dataloader = DataLoader(toxic_dataset, len(val_sampler.indices), sampler=val_sampler)
test_dataloader = DataLoader(toxic_dataset, len(test_sampler.indices), sampler=test_sampler)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = LstmNet(toxic_dataset.initial_embeddings, 200, device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
BCELoss = nn.BCELoss()
losses = {}
num_epochs = 1



for epoch in range(num_epochs):
    losses[epoch] = []
    for data_sample in train_dataloader:
        for i in range(len(data_sample)):
            idx = np.argsort(-sample_data[0])
            sample_data[i] = sample_data[i][idx].to(device)

        model.zero_grad()


        out = model(data_sample[0])
        loss = BCELoss(out, data_sample[1].float())

        loss.backward()
        optimizer.step()
        print(loss.data)
        losses[epoch].append(loss.data)

#    accuracy_score(lables, preds>.5)

#    torch.save(model.state_dict(), ../output/'model' + datetime.now().date())
toxic_eval =ToxicEvaluation(model, val_dataloader, device)
labels, preds = snopes_eval.claim_wise_accuracies()

val_outputs = model(data)
_, predicted = torch.max(outputs, 1)
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
100 * correct / total))

dataiter = iter(val_dataloader)
data, labels = dataiter.next()
true_claim_indices = np.where(labels==1)
false_claim_indices = np.where(labels==0)
accuracy_score(labels[true_claim_indices], preds[true_claim_indices]>0.5)
accuracy_score(labels[false_claim_indices], preds[false_claim_indices]>0.5)
