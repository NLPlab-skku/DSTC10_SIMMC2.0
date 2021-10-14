import torch
import torchvision

import numpy as np
import torchvision.transforms as T
import seaborn as sns
import torch.nn.functional as F

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import models
from torch import nn, optim
from torch.optim import lr_scheduler
from tqdm import tqdm
from matplotlib.ticker import MaxNLocator

from matplotlib import pyplot as plt
from glob import glob
from pathlib import Path
from collections import defaultdict
from os import listdir
from os.path import isfile, join

def load_json(filename):
    with open(filename,mode="r") as f:
        return json.load(f)

def save_json(data,filename):
    with open(filename, mode="w") as f:
        json.dump(data,f)
    return

sns.set(style='whitegrid', palette='muted', font_scale=1.2)

HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]

sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"

from pathlib import Path
cwd = Path().resolve()
query =cwd / "*"
print(query)
# glob(str(query))

scene_folder = "simmc2_scene_jsons_dstc10_public"
scenes = sorted(glob(str(scene_folder + "/*_scene.json")))
bboxes = sorted(glob(str(scene_folder + "/*_bbox.json")))

DATASETS = ['train', 'val'] # split only by two.

image_path1=str(cwd)+  "/simmc2_scene_images_dstc10_public_part2"
image_path2=str(cwd)+ "/simmc2_scene_images_dstc10_public_part1"
image_files1 = [f for f in listdir(image_path1) if isfile(join(image_path1, f))]
image_files2 = [f for f in listdir(image_path2) if isfile(join(image_path2, f))]

mean_nums = [0.485, 0.456, 0.406]
std_nums = [0.229, 0.224, 0.225]

transforms = {'train': T.Compose([
  T.RandomResizedCrop(size=[256,256]),
  T.RandomRotation(degrees=15),
  T.RandomHorizontalFlip(),
  T.ToTensor(),
  T.Normalize(mean_nums, std_nums)
]), 'val': T.Compose([
  T.Resize(size=[256,256]),
  T.CenterCrop(size=256),
  T.ToTensor(),
  T.Normalize(mean_nums, std_nums)
]), 'test': T.Compose([
  T.Resize(size=[256,256]),
  T.CenterCrop(size=256),
  T.ToTensor(),
  T.Normalize(mean_nums, std_nums)
]),
}


furniture_type_folder = Path("vision_data/furniture/type_data")
fashion_type_folder = Path("vision_data/fashion/type_data")
furniture_materials_folder = Path("vision_data/furniture/materials_data")
fashion_pattern_folder = Path("vision_data/fashion/pattern_data")
color_folder=Path("vision_data/color/color_data")


# Set input for model
# DATA_DIR=Path("vision_data/color/color_data")
DATA_DIR=color_folder
# ORG_DATA_DIR = Path("vision_data/color/original_color_data")
image_datasets = {
  d: ImageFolder(f'{DATA_DIR}/{d}', transforms[d]) for d in DATASETS
}

data_loaders = {
  d: DataLoader(image_datasets[d], batch_size=4, shuffle=True, num_workers=4) 
  for d in DATASETS
}

dataset_sizes = {d: len(image_datasets[d]) for d in DATASETS}
class_names = image_datasets['train'].classes


def create_model(n_classes):
#   model = models.resnet50(pretrained=True)
  model = models.resnext101_32x8d(pretrained=True)
  n_features = model.fc.in_features
  model.fc = nn.Linear(n_features, n_classes)

  return model.to(device)


def train_epoch(
  model, 
  data_loader, 
  loss_fn, 
  optimizer, 
  device, 
  scheduler, 
  n_examples
):
  model = model.train()

  losses = []
  correct_predictions = 0
  
  for inputs, labels in data_loader:
    inputs = inputs.to(device)
    labels = labels.to(device)

    outputs = model(inputs)

    _, preds = torch.max(outputs, dim=1)
    loss = loss_fn(outputs, labels)

    correct_predictions += torch.sum(preds == labels)
    losses.append(loss.item())

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

  scheduler.step()

  return correct_predictions.double() / n_examples, np.mean(losses)


def eval_model(model, data_loader, loss_fn, device, n_examples):
  model = model.eval()

  losses = []
  correct_predictions = 0

  with torch.no_grad():
    for inputs, labels in data_loader:
      inputs = inputs.to(device)
      labels = labels.to(device)

      outputs = model(inputs)

      _, preds = torch.max(outputs, dim=1)

      loss = loss_fn(outputs, labels)

      correct_predictions += torch.sum(preds == labels)
      losses.append(loss.item())

  return correct_predictions.double() / n_examples, np.mean(losses)


def train_model(model, data_loaders, dataset_sizes, device, n_epochs=3):
  optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
  scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
  loss_fn = nn.CrossEntropyLoss().to(device)

  history = defaultdict(list)
  best_accuracy = 0

  for epoch in tqdm(range(n_epochs),total=n_epochs):

    print(f'Epoch {epoch + 1}/{n_epochs}')
    print('-' * 10)

    train_acc, train_loss = train_epoch(
      model,
      data_loaders['train'],    
      loss_fn, 
      optimizer, 
      device, 
      scheduler, 
      dataset_sizes['train']
    )

    print(f'Train loss {train_loss} accuracy {train_acc}')

    val_acc, val_loss = eval_model(
      model,
      data_loaders['val'],
      loss_fn,
      device,
      dataset_sizes['val']
    )

    print(f'Val   loss {val_loss} accuracy {val_acc}')
    print()

    history['train_acc'].append(train_acc.cpu())
    history['train_loss'].append(train_loss)
    history['val_acc'].append(val_acc.cpu())
    history['val_loss'].append(val_loss)

    if val_acc > best_accuracy:
      torch.save(model.state_dict(), 'best_model_state.bin')
      best_accuracy = val_acc

  print(f'Best val accuracy: {best_accuracy}')
  
  model.load_state_dict(torch.load('best_model_state.bin'))

  return model, history

# Set model
base_model = create_model(len(class_names))

base_model, history = train_model(base_model, data_loaders, dataset_sizes, device,n_epochs=40)





