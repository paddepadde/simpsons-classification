# Testing the ignite functionality included with pyTorch 

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch 
import torchvision
import torchvision.transforms as transforms

from ignite.engine import Events, create_supervised_evaluator, \
    create_supervised_trainer
from ignite.metrics import Accuracy, Loss

sns.set_style('darkgrid')

# Hyperparameters for the training procedure
batch_size = 200
epochs = 25
learn_rate = 1e-3

# Use cuda if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load the data for the experiment
transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = torchvision.datasets.ImageFolder('./data/train', transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# Map for labels, so that we can visualize the results better
label_map = {
    0: 'Apu', 1: 'Bart', 2: 'Mr. Burns', 3: 'Chief Wiggum', 4: 'Edna', 5: 'Grandpa',
    6: 'Homer', 7: 'Krusty', 8: 'Lisa', 9: 'Marge', 10: 'Milhouse', 11: 'Moe', 
    12: 'Flanders', 13: 'Nelson', 14: 'Patty', 15: 'Skinner', 16: 'Selma', 17: 'Smithers'
}

# Load test dataset 
test_dataset = torchvision.datasets.ImageFolder('./data/test', transform=transform)
validation_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

from models import ConvModel

model = ConvModel(channels=3, num_classes=18).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

print(model)

trainer = create_supervised_trainer(model, optimizer, criterion, device=device)

metrics = {
    "accuracy": Accuracy(),
    "loss": Loss(criterion)
}
evaluator = create_supervised_evaluator(model,metrics, device=device)

@trainer.on(Events.ITERATION_COMPLETED(every=10))
def log_training_loss(engine):
    print(engine.state.output)

@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(engine):
    evaluator.run(dataloader)
    metrics = evaluator.state.metrics
    print(metrics)

@trainer.on(Events.EPOCH_COMPLETED)
def log_validation_resluts(engine):
    evaluator.run(validation_loader)
    metrics = evaluator.state.metrics
    print(metrics)

trainer.run(dataloader, max_epochs=20)