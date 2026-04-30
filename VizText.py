# ============================================================
# RoBERTa + ResNet101 + Early Fusion + PCNN — MVSA-Single
# ============================================================

import subprocess, sys  # import modules to run shell commands and access Python executable

subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "transformers", "torch", "torchvision", "Pillow", "scikit-learn", "tqdm"])
# install required libraries using pip; "-q" suppresses verbose output

import os, re, random  # os for file handling, re for regex, random for reproducibility
import numpy as np     # numerical operations
from PIL import Image  # image loading and processing
from tqdm import tqdm  # progress bar for loops

import torch  # main PyTorch library
import torch.nn as nn  # neural network module
import torch.nn.functional as F  # functional utilities like activation, pooling
from torch.utils.data import Dataset, DataLoader, Subset  # dataset utilities

from torchvision import transforms, models  # image preprocessing and pretrained models
from transformers import RobertaTokenizer, RobertaModel, get_linear_schedule_with_warmup
# tokenizer, model, and learning rate scheduler

from sklearn.metrics import accuracy_score, classification_report  # evaluation metrics


# ── Config ──────────────────────────────────────────────────
SEED        = 42  # random seed for reproducibility
DATA_DIR    = "/content/drive/MyDrive/MVSA_Single/data"  # dataset directory path
LABEL_FILE  = "/content/drive/MyDrive/MVSA_Single/labelResultAll.txt"  # label file path
MAX_LEN     = 128  # max token length for text input
IMG_SIZE    = 224  # image size expected by model
BATCH_SIZE  = 16   # batch size for training
EPOCHS      = 20   # number of training epochs
LR_TEXT     = 1e-5  # learning rate for text encoder
LR_IMAGE    = 5e-6  # learning rate for image encoder
LR_HEAD     = 5e-5  # learning rate for classifier layers
TRAIN_RATIO = 0.70  # percentage of data used for training
VAL_RATIO   = 0.15  # percentage of data used for validation
PATIENCE    = 5     # early stopping patience
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# choose GPU if available, else CPU

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
# set seeds for reproducibility across libraries

print(f"Device: {DEVICE}")  # print which device is being used


# ── Label Parsing ────────────────────────────────────────────
LABEL_MAP = {"negative": 0, "neutral": 1, "positive": 2}
# mapping string labels to numeric classes

def parse_labels(label_file):  # function to parse dataset labels
    samples = []  # initialize empty list to store valid samples

    with open(label_file, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()  # read all lines from label file

    for line in lines[1:]:  # skip header line and iterate over rest
        line = line.strip()  # remove leading/trailing whitespace
        if not line:  # skip empty lines
            continue  # move to next iteration

        parts = re.split(r"\s+", line, maxsplit=1)
        # split line into two parts: ID and label info

        if len(parts) < 2:  # if line doesn't have expected format
            continue  # skip invalid line

        sample_id  = parts[0].strip()  # extract sample ID
        labels     = parts[1].strip().split(",")  # split label string

        text_label = labels[0].strip().lower()  # extract first label (text sentiment)

        if text_label not in LABEL_MAP:  # check if label is valid
            continue  # skip unknown labels

        img_path = os.path.join(DATA_DIR, f"{sample_id}.jpg")  # construct image path
        txt_path = os.path.join(DATA_DIR, f"{sample_id}.txt")  # construct text path

        if os.path.exists(img_path) and os.path.exists(txt_path):
            # ensure both files exist
            samples.append((txt_path, img_path, LABEL_MAP[text_label]))
            # add tuple (text path, image path, numeric label)

    return samples  # return collected samples


samples = parse_labels(LABEL_FILE)  # call function to parse dataset
print(f"Total valid samples: {len(samples)}")  # print number of valid samples


# ── Dataset ──────────────────────────────────────────────────
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
# load pretrained RoBERTa tokenizer

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),  # resize image to 256x256
    transforms.RandomCrop(IMG_SIZE),  # randomly crop to 224x224
    transforms.RandomHorizontalFlip(),  # randomly flip horizontally
    transforms.RandomRotation(10),  # rotate image randomly up to 10 degrees
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    # randomly change brightness, contrast, saturation
    transforms.ToTensor(),  # convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
    # normalize image using ImageNet stats
    transforms.RandomErasing(p=0.1),  # randomly erase part of image
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),  # resize directly to 224x224
    transforms.ToTensor(),  # convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
    # normalize image
])


class MVSADataset(Dataset):  # custom dataset class
    def __init__(self, samples, transform):
        self.samples   = samples  # store samples list
        self.transform = transform  # store transform

    def __len__(self):
        return len(self.samples)  # return dataset size

    def __getitem__(self, idx):
        txt_path, img_path, label = self.samples[idx]  # unpack sample

        with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read().strip()  # read and clean text

        enc = tokenizer(text, max_length=MAX_LEN,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt")
        # tokenize text with padding and truncation

        input_ids      = enc["input_ids"].squeeze(0)
        # remove batch dimension from token IDs

        attention_mask = enc["attention_mask"].squeeze(0)
        # remove batch dimension from attention mask

        try:
            img = Image.open(img_path).convert("RGB")
            # load image and convert to RGB
        except Exception:
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE))
            # fallback blank image if loading fails

        img = self.transform(img)  # apply image transformations

        return input_ids, attention_mask, img, torch.tensor(label, dtype=torch.long)
        # return processed sample


# ── Dataset Split ────────────────────────────────────────────
n_total   = len(samples)  # total number of samples
n_train   = int(n_total * TRAIN_RATIO)  # number of training samples
n_val     = int(n_total * VAL_RATIO)  # number of validation samples
n_test    = n_total - n_train - n_val  # remaining samples for testing

rng       = torch.Generator().manual_seed(SEED)
# create random generator with fixed seed

indices   = torch.randperm(n_total, generator=rng).tolist()
# generate shuffled indices

train_idx = indices[:n_train]  # first part for training
val_idx   = indices[n_train:n_train + n_val]  # next part for validation
test_idx  = indices[n_train + n_val:]  # remaining for testing

full_train_ds = MVSADataset(samples, train_transform)  # full training dataset
full_val_ds   = MVSADataset(samples, val_transform)  # full validation dataset

train_ds = Subset(full_train_ds, train_idx)  # create training subset
val_ds   = Subset(full_val_ds, val_idx)  # create validation subset
test_ds  = Subset(full_val_ds, test_idx)  # create test subset

print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
# print dataset split sizes


train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=2, pin_memory=True)
# training data loader

val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=2, pin_memory=True)
# validation data loader

test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=2, pin_memory=True)
# test data loader


# ── Model ────────────────────────────────────────────────────

class TextEncoder(nn.Module):  # define text encoder
    def __init__(self):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained("roberta-base")
        # load pretrained RoBERTa model

        for name, param in self.roberta.named_parameters():
            if not any(f"layer.{i}" in name for i in [10, 11]):
                param.requires_grad = False
                # freeze all layers except last two

    def forward(self, input_ids, attention_mask):
        out = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        # forward pass through RoBERTa

        return out.last_hidden_state[:, 0, :]
        # take CLS token representation


class ImageEncoder(nn.Module):  # define image encoder
    def __init__(self):
        super().__init__()

        base = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
        # load pretrained ResNet101

        for name, param in base.named_parameters():
            if not name.startswith("layer4"):
                param.requires_grad = False
                # freeze all layers except last block

        self.backbone = nn.Sequential(*list(base.children())[:-1])
        # remove final classification layer

    def forward(self, x):
        return self.backbone(x).flatten(1)
        # forward pass and flatten output


class PCNN(nn.Module):  # define PCNN module
    def __init__(self, in_dim, num_filters=256, kernel_sizes=(2, 3, 4)):
        super().__init__()

        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(1, num_filters, kernel_size=k, padding=k // 2),
                nn.ReLU()
            ) for k in kernel_sizes
        ])
        # create multiple convolution layers with different kernel sizes

        self.out_dim = num_filters * len(kernel_sizes)
        # total output dimension

    def forward(self, x):
        x = x.unsqueeze(1)
        # add channel dimension

        return torch.cat([
            F.max_pool1d(conv(x), conv(x).size(2)).squeeze(2)
            for conv in self.convs
        ], dim=1)
        # apply conv + max pooling and concatenate outputs


class EarlyFusionPCNN(nn.Module):  # define full model
    def __init__(self, num_classes=3, dropout=0.5):
        super().__init__()

        self.text_enc = TextEncoder()
        # initialize text encoder

        self.img_enc  = ImageEncoder()
        # initialize image encoder

        fused_dim     = 768 + 2048
        # combined feature size

        self.pcnn     = PCNN(in_dim=fused_dim)
        # PCNN module

        self.classifier = nn.Sequential(
            nn.Linear(self.pcnn.out_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        # classification layers

    def forward(self, input_ids, attention_mask, images):
        text_feat = self.text_enc(input_ids, attention_mask)
        # extract text features

        img_feat  = self.img_enc(images)
        # extract image features

        fused     = torch.cat([text_feat, img_feat], dim=1)
        # concatenate features

        pcnn_out  = self.pcnn(fused)
        # pass through PCNN

        return self.classifier(pcnn_out)
        # output class logits


model = EarlyFusionPCNN(num_classes=3).to(DEVICE)
# create model and move to device

print("Model built successfully.")
# confirm model creation


# ── Optimizer ────────────────────────────────────────────────
optimizer = torch.optim.AdamW([
    {"params": model.text_enc.parameters(), "lr": LR_TEXT},
    {"params": model.img_enc.parameters(),  "lr": LR_IMAGE},
    {"params": list(model.pcnn.parameters()) +
               list(model.classifier.parameters()), "lr": LR_HEAD},
], weight_decay=0.01)
# define optimizer with different learning rates


total_steps  = len(train_loader) * EPOCHS
# total training steps

warmup_steps = int(total_steps * 0.1)
# warmup steps (10%)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)
# learning rate scheduler

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
# loss function with label smoothing


# ── Training Loop ────────────────────────────────────────────
def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    # set model mode

    total_loss, all_preds, all_labels = 0.0, [], []
    # initialize tracking variables

    ctx = torch.enable_grad() if train else torch.no_grad()
    # enable gradients only during training

    with ctx:
        for input_ids, attention_mask, images, labels in tqdm(loader, leave=False):

            input_ids      = input_ids.to(DEVICE)
            attention_mask = attention_mask.to(DEVICE)
            images         = images.to(DEVICE)
            labels         = labels.to(DEVICE)
            # move batch to device

            logits = model(input_ids, attention_mask, images)
            # forward pass

            loss   = criterion(logits, labels)
            # compute loss

            if train:
                optimizer.zero_grad()
                # clear gradients

                loss.backward()
                # backpropagation

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                # gradient clipping

                optimizer.step()
                # update weights

                scheduler.step()
                # update learning rate

            total_loss += loss.item()
            # accumulate loss

            all_preds.extend(logits.argmax(1).cpu().numpy())
            # store predictions

            all_labels.extend(labels.cpu().numpy())
            # store true labels

    return total_loss / len(loader), accuracy_score(all_labels, all_preds)
    # return average loss and accuracy


best_val_acc = 0.0
# best validation accuracy

no_improve   = 0
# early stopping counter

print("\n" + "="*60)
# print separator

print(" RoBERTa + ResNet101 + Early Fusion + PCNN")
# print title

print("="*60)
# print separator


for epoch in range(1, EPOCHS + 1):
    # iterate over epochs

    tr_loss, tr_acc = run_epoch(train_loader, train=True)
    # train for one epoch

    vl_loss, vl_acc = run_epoch(val_loader, train=False)
    # validate for one epoch

    print(f"Epoch [{epoch:02d}/{EPOCHS}] "
          f"Train Loss: {tr_loss:.4f} Acc: {tr_acc*100:.2f}% | "
          f"Val Loss: {vl_loss:.4f} Acc: {vl_acc*100:.2f}%")
    # print results

    if vl_acc > best_val_acc:
        best_val_acc = vl_acc
        no_improve   = 0
        torch.save(model.state_dict(), "best_early_fusion.pth")
        # save best model

    else:
        no_improve += 1

        if no_improve >= PATIENCE:
            print(f"Early stopping at epoch {epoch}.")
            break


# ── Test ─────────────────────────────────────────────────────

model.load_state_dict(torch.load("best_early_fusion.pth", map_location=DEVICE))
# load best model weights

model.eval()
# set model to evaluation mode

all_preds, all_labels = [], []
# initialize lists

with torch.no_grad():
    # disable gradients

    for input_ids, attention_mask, images, labels in test_loader:

        logits = model(input_ids.to(DEVICE),
                       attention_mask.to(DEVICE),
                       images.to(DEVICE))
        # forward pass

        all_preds.extend(logits.argmax(1).cpu().numpy())
        # collect predictions

        all_labels.extend(labels.numpy())
        # collect ground truth labels


test_acc = accuracy_score(all_labels, all_preds)
# compute test accuracy

print("\n" + "="*60)
# print separator

print(f"  ✅  FUSION TEST ACCURACY: {test_acc * 100:.2f}%")
# print test accuracy

print("="*60)
# print separator

print(classification_report(all_labels, all_preds,
      target_names=["negative", "neutral", "positive"]))
# print precision, recall, F1-score report
