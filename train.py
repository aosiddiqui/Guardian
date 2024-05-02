import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, Subset, SubsetRandomSampler
from torchmetrics import Accuracy, Precision, Recall, F1Score, AUROC

import os
import h5py
import csv
import tqdm
import time
import pickle
import sys

random_seed = 42
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)

# Dataset class
class KidsGuardDataset(Dataset):
    def __init__(self):
        self.dataset = 'KidsGuard/kidsguard-dataset/kidsguard_complete.h5'
    
    def __len__(self):
        with h5py.File(self.dataset, 'r') as file:
            return len(file['labels'])

    def __getitem__(self, index):
        with h5py.File(self.dataset, 'r') as file:
            # Load the video segments, embeddings, and labels from the video folder
            video_embedding = torch.from_numpy(file['video_embeddings'][index])
            audio_embedding = torch.from_numpy(file['audio_embeddings'][index])
            label = file['labels'][index]

        return {'video_embeddings': video_embedding, 'audio_embeddings': audio_embedding, 'labels': label}

#-----------Architecture----------------
class EncoderModel(nn.Module):
    def __init__(self, d_model=768, nhead=8, num_layers=1):
        super().__init__()
        self.video_encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.video_encoder = nn.TransformerEncoder(self.video_encoder_layer, num_layers=num_layers)

        self.audio_encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.audio_encoder = nn.TransformerEncoder(self.audio_encoder_layer, num_layers=num_layers)

    def forward(self, x, y):
        x = self.video_encoder(x)
        y = self.audio_encoder(y)

        return x, y
    
class KidsGuardModel(nn.Module):
    def __init__(self, d_model, nhead, num_layers):
        super().__init__()
        self.encoder = EncoderModel(d_model=d_model, nhead=nhead, num_layers=num_layers)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

        # # Single modality / avg / elt-wise multiplication
        # self.linear1 = nn.Linear(d_model, d_model//2)
        # self.linear2 = nn.Linear(d_model//2, 4)

        # # Concat first
        # self.linear1 = nn.Linear(2*d_model, d_model)
        # self.linear2 = nn.Linear(d_model, 4)

        # # Concat later
        # self.linear1 = nn.Linear(d_model, d_model//2)
        # self.linear2 = nn.Linear(d_model, d_model//2)
        # self.linear3 = nn.Linear(d_model, 4)

        # Avg
        # self.linear = nn.Linear(d_model, 4)

        # Outer-product
        # # A
        # self.linear_video = nn.Linear(d_model, 120)
        # self.linear_audio = nn.Linear(d_model, 120)
        # self.linear1 = nn.Linear(120*120, 4096)
        # self.linear2 = nn.Linear(4096, 512)
        # self.linear3 = nn.Linear(512, 4)

        # # B
        # self.linear_video = nn.Linear(d_model, 120)
        # self.linear_audio = nn.Linear(d_model, 120)
        # self.linear1 = nn.Linear(120*120, 1024)
        # self.linear2 = nn.Linear(1024, 4)

    def forward(self, x, y):
        x, y = self.encoder(x, y)

        # print('x.shape:', x.shape)  # (b,d)
        # print('y.shape:', y.shape)  # (b,d)

        # # video only
        # input = x
        # output = self.relu(self.linear1(input))
        # output = self.linear2(output)

        # # audio only
        # input = y
        # output = self.relu(self.linear1(input))
        # output = self.linear2(output)

        # # Concat first
        # input = torch.cat([x, y], dim=1)
        # output = self.relu(self.linear1(input))
        # output = self.linear2(output)

        # # Concat later
        # x = self.relu(self.linear1(x))
        # y = self.relu(self.linear2(y))
        # input = torch.cat([x,y], dim = 1)
        # output = self.linear3(input)

        # For focal loss only
        # output = self.softmax(output)

        # # Avg
        # input = (x + y)/2
        # # Elt-wise multiplication
        # input = x * y
        # output = self.linear1(input)
        # output = self.linear2(output)

        # Outer-product
        # # A
        # x, y = self.linear_video(x), self.linear_audio(y)
        # x, y = x.unsqueeze(2), y.unsqueeze(1)
        # input = torch.matmul(x, y)
        # output = self.relu(self.linear1(input.view(input.size(0), -1)))
        # output = self.relu(self.linear2(output))
        # output = self.linear3(output)

        # B
        x, y = self.linear_video(x), self.linear_audio(y)
        x, y = x.unsqueeze(2), y.unsqueeze(1)
        input = torch.matmul(x, y)
        output = self.relu(self.linear1(input.view(input.size(0), -1)))
        output = self.linear2(output)
        # output = F.softmax()
        return output

# --------------------Hyperparameters-----------------------------
d_model = 768
nhead = 8
num_layers = 1
n_epochs = 200
train_batch_size = 15000
val_batch_size = 7500
test_batch_size = 7500
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
#---------------------Losses--------------------------------------
criterion = nn.CrossEntropyLoss()
# criterion = torch.hub.load(
# 	'adeelh/pytorch-multi-class-focal-loss',
# 	model='focal_loss',
# 	alpha=None,
# 	gamma=2,
# 	reduction='mean',
# 	device=device,
# 	dtype=torch.float32,
# 	force_reload=False
# )

#---------------------Optimizer and model-------------------------
model = KidsGuardModel(d_model, nhead, num_layers)
model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience =10,factor=0.5, threshold=1e-15, verbose=True)

def log_metrics_to_csv(metrics, filename):
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(metrics)

results_folder = 'KidsGuard/results/wo_pretrain/demo'
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

metrics_file = 'metrics.csv'
metrics_path = os.path.join(results_folder, metrics_file)
log_metrics_to_csv(['train_loss', 'val_loss', 'accuracy', 'precision', 'recall', 'F-1 score', 'AUC' ,'epoch'], metrics_path)

with h5py.File('KidsGuard/kidsguard-dataset/kidsguard_complete.h5', 'r') as f:
    labels = torch.from_numpy(f['labels'][...])

# Creating a custom dataset
dataset = KidsGuardDataset()

# Creating indices for stratified split
num_samples = len(dataset)
num_train = int(0.75 * num_samples)
num_val = int(0.05 * num_samples)
num_test = num_samples - num_train - num_val

# Prepare indices to perform stratified split
num_samples = len(dataset)
indices = list(range(num_samples))
targets = labels  # Replace 'targets' with the actual target labels

# Stratified split using sklearn's StratifiedShuffleSplit
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=random_seed)  # 75:5:20 split
train_idx, test_idx = next(sss.split(indices, targets))

train_val_dataset = Subset(dataset, train_idx)
test_dataset = Subset(dataset, test_idx)

# Further split the training set into training and validation sets
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.0625, random_state=random_seed)  # 75:5:20 split
train_idx, val_idx = next(sss.split(train_val_dataset.indices, [targets[i] for i in train_val_dataset.indices]))

train_dataset = Subset(train_val_dataset, train_idx)
val_dataset = Subset(train_val_dataset, val_idx)

# Creating DataLoader for train, validation, and test sets
train_loader = DataLoader(train_dataset, batch_size=train_batch_size, num_workers=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

# Save the split indices to a file
indices_file = 'split_indices.pkl'
indices_path = os.path.join(results_folder, indices_file)
with open(indices_path, 'wb') as f:
    split_indices = {
        'train': train_idx,
        'val': val_idx,
        'test': test_idx
    }
    pickle.dump(split_indices, f)

train_losses = []
val_losses = []

best_loss = float('inf')  
best_model_state = None

accuracy_score = Accuracy(task='multiclass', average='weighted', num_classes=4).to(device)
precision_score = Precision(task='multiclass', average='weighted', num_classes=4).to(device)
recall_score = Recall(task='multiclass', average='weighted', num_classes=4).to(device)
f1_score = F1Score(task='multiclass', average='weighted', num_classes=4).to(device)
auc_score = AUROC(task='multiclass', average='weighted', num_classes=4).to(device)

def compute_metrics(predicted_probs, true_labels):
    predicted_labels = torch.argmax(predicted_probs, dim=1)

    accuracy = accuracy_score(true_labels, predicted_labels).item()
    print("\nAccuracy:", accuracy)

    # Multiclass precision, recall, and F1 score
    precision = precision_score(predicted_labels, true_labels).item()
    recall = recall_score(predicted_labels, true_labels).item()
    f1 = f1_score(predicted_labels, true_labels).item()
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

    # AUC (Area Under the ROC Curve)
    auc = auc_score(predicted_probs, true_labels).item()
    print("AUC:", auc)
    return accuracy, precision, recall, f1, auc

print('Training started.....')
for epoch in range(n_epochs):
    print(f'Epoch [{epoch+1}/{n_epochs}]')
    start_time = time.time()
    # Training loop
    running_loss = 0.0
    num_batches = len(train_loader)
    model.train()
    for i, input in enumerate(train_loader):  # Iterate over training batches
        print(f'Step [{i+1}/{num_batches}]')

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        video_embeddings = input['video_embeddings'].to(device)
        audio_embeddings = input['audio_embeddings'].to(device)
        labels = input['labels'].to(dtype=torch.long, device=device)

        print('video_embeddings.shape:', video_embeddings.shape)
        print('audio_embeddings.shape:', audio_embeddings.shape)
        print('labels.shape:', labels.shape)
        # sys.exit()
        preds = model(video_embeddings, audio_embeddings).to(device)
        
        # Compute loss
        loss = criterion(preds, labels)               
        print(f'Train loss: {loss.item():.3f}')
        print()

        # Backpropagation and weight updation
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    # Calculate average training loss for this epoch
    average_train_loss = running_loss / num_batches
    print()
    print(f'Avg train loss: {average_train_loss:.3f}')

    running_loss = 0.0
    model.eval()
    # validation step
    with torch.no_grad():
        num_batches = len(val_loader)
        for input in val_loader:
            # Forward pass
            video_embeddings = input['video_embeddings'].to(device)
            audio_embeddings = input['audio_embeddings'].to(device)
            labels = input['labels'].to(dtype=torch.long, device=device)

            preds = model(video_embeddings, audio_embeddings).to(device)

            # Compute loss
            loss = criterion(preds, labels)
            # print(f'Avg contrastive loss for {npairs} pairs: {loss.item():.3f}')

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_model_state = model.state_dict()
                torch.save(model.state_dict(), f'{results_folder}/best_model.pth')  # Save the best model

            running_loss += loss.item()

        # Calculate average validation metrics for this epoch
        average_val_loss = running_loss / num_batches

        # print('labels.shape:', labels.shape)
        # print('preds.shape:', preds.shape)
        acc, prec, recall, f1, auc = compute_metrics(F.softmax(preds, dim = 1), labels)

        metrics = [average_train_loss, average_val_loss, acc, prec, recall, f1, auc, epoch]
        log_metrics_to_csv(metrics, metrics_path)

    train_losses.append(average_train_loss)
    val_losses.append(average_val_loss)
    scheduler.step(val_losses[-1])
    end_time = time.time()
    print(f'Avg val loss: {average_val_loss: .3f}')
    print('Time taken:{:.4f} minutes'.format((end_time - start_time)/60))
    print('-----------------------------')
    print()

torch.save(model.state_dict(), f'{results_folder}/last_model.pth')
np.save(f"{results_folder}/train_loss.npy", np.array(train_losses))
np.save(f"{results_folder}/val_loss.npy", np.array(val_losses))

print()