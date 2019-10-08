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
val_dataloader = DataLoader(toxic_dataset, 128, sampler=val_sampler)
test_dataloader = DataLoader(toxic_dataset, 128, sampler=test_sampler)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = LstmNet(toxic_dataset.initial_embeddings, 200, device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
BCELoss = nn.BCELoss()
losses = {}
num_epochs = 1
count = 0

trainer = ModelTrainer(model, train_dataloader, val_dataloader, BCELos)

