import os
_MLEVOLVE_BATCH_SIZE_OVERRIDE = os.environ.get('MLEVOLVE_BATCH_SIZE_OVERRIDE')
_MLEVOLVE_PROBE_MODE = os.environ.get('MLEVOLVE_PROBE_MODE') == '1'
_MLEVOLVE_PROBE_MAX_EPOCHS = int(os.environ['MLEVOLVE_PROBE_MAX_EPOCHS']) if os.environ.get('MLEVOLVE_PROBE_MAX_EPOCHS') else None
_MLEVOLVE_PROBE_MAX_TRAIN_BATCHES = int(os.environ['MLEVOLVE_PROBE_MAX_TRAIN_BATCHES']) if os.environ.get('MLEVOLVE_PROBE_MAX_TRAIN_BATCHES') else None

def _mlevolve_apply_probe_limits():
    if not _MLEVOLVE_PROBE_MODE or _MLEVOLVE_PROBE_MAX_TRAIN_BATCHES is None:
        return
    try:
        from torch.utils.data import DataLoader
    except Exception:
        return
    _original_iter = DataLoader.__iter__

    def _limited_iter(self):
        iterator = _original_iter(self)
        for _idx, item in enumerate(iterator):
            if _idx >= _MLEVOLVE_PROBE_MAX_TRAIN_BATCHES:
                break
            yield item
    DataLoader.__iter__ = _limited_iter
_mlevolve_apply_probe_limits()
import os
import gc
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import timm
from sklearn.metrics import roc_auc_score as _mlevolve_original_roc_auc_score


def roc_auc_score(y_true, y_score, *args, **kwargs):
    try:
        return _mlevolve_original_roc_auc_score(y_true, y_score, *args, **kwargs)
    except ValueError as exc:
        if 'Only one class present' in str(exc):
            return 0.5
        raise

MODEL_FAMILY = 'custom_cnn'
NUM_EPOCHS = 3
BATCH_SIZE = 512
NUM_WORKERS = 2
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 1e-05
CHECKPOINT_DIR = './working'
SUBMISSION_DIR = './submission'
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, 'best_model_28041cb6f1ba409b949c6ba85dff301c.pth')
SUBMISSION_PATH = os.path.join(SUBMISSION_DIR, 'submission_28041cb6f1ba409b949c6ba85dff301c.csv')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
AMP_ENABLED = False
AMP_DTYPE = None
if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = False
SCALER = None

def log_precision_settings():
    """Prints the current device and precision configuration."""
    print('--- Datatype & Precision Settings ---')
    print(f'Target device: {DEVICE}')
    if DEVICE.type == 'cuda':
        print(f'CUDA device name: {torch.cuda.get_device_name(DEVICE)}')
        print(f"Automatic Mixed Precision (AMP): {('Enabled' if AMP_ENABLED else 'Disabled')}")
        if AMP_ENABLED and AMP_DTYPE:
            print(f'AMP Dtype: {AMP_DTYPE}')
        tf32_enabled = torch.backends.cuda.matmul.allow_tf32
        print(f"TF32 on matmul: {('Enabled' if tf32_enabled else 'Disabled')}")
    else:
        print('Running on CPU. AMP and TF32 are not applicable.')
    print('-' * 35)

class HistologyDataset(Dataset):
    """
    Custom PyTorch Dataset for the PatchCamelyon dataset.
    Loads images on-the-fly and applies specified transformations.
    """

    def __init__(self, df, data_dir, transform=None, is_test=False):
        self.df = df
        self.data_dir = data_dir
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_id = self.df.iloc[idx, 0]
        img_path = os.path.join(self.data_dir, f'{img_id}.tif')
        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            print(f'Error: Image file not found at {img_path}')
            image = Image.new('RGB', (96, 96), color='black')
            if self.is_test:
                return (image, img_id)
            else:
                return (image, torch.tensor(-1, dtype=torch.float32))
        if self.transform:
            image = self.transform(image)
        if self.is_test:
            return (image, img_id)
        else:
            label = int(self.df.iloc[idx, 1])
            return (image, torch.tensor(label, dtype=torch.float32))

def get_dataloaders(batch_size, num_workers, valid_size=0.1, random_state=42):
    """
    Creates and returns the training, validation, and test DataLoaders.
    """
    BASE_PATH = './input'
    TRAIN_DIR = os.path.join(BASE_PATH, 'train')
    TEST_DIR = os.path.join(BASE_PATH, 'test')
    LABELS_PATH = os.path.join(BASE_PATH, 'train_labels.csv')
    data_transforms = {'train': transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), transforms.RandomRotation(90), transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05), transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]), 'valid': transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])}
    full_train_df = pd.read_csv(LABELS_PATH)
    train_df, valid_df = train_test_split(full_train_df, test_size=valid_size, random_state=random_state, stratify=full_train_df['label'])
    train_df = train_df.reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)
    train_dataset = HistologyDataset(df=train_df, data_dir=TRAIN_DIR, transform=data_transforms['train'])
    valid_dataset = HistologyDataset(df=valid_df, data_dir=TRAIN_DIR, transform=data_transforms['valid'])
    submission_df = pd.read_csv(os.path.join(BASE_PATH, 'sample_submission.csv'))
    test_dataset = HistologyDataset(df=submission_df, data_dir=TEST_DIR, transform=data_transforms['valid'], is_test=True)
    print(f'Train samples: {len(train_dataset)}, Validation samples: {len(valid_dataset)}, Test samples: {len(test_dataset)}')
    train_loader = DataLoader(train_dataset, batch_size=int(_MLEVOLVE_BATCH_SIZE_OVERRIDE) if _MLEVOLVE_BATCH_SIZE_OVERRIDE is not None else batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=int(_MLEVOLVE_BATCH_SIZE_OVERRIDE) if _MLEVOLVE_BATCH_SIZE_OVERRIDE is not None else batch_size * 2, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=int(_MLEVOLVE_BATCH_SIZE_OVERRIDE) if _MLEVOLVE_BATCH_SIZE_OVERRIDE is not None else batch_size * 2, shuffle=False, num_workers=num_workers, pin_memory=True)
    return (train_loader, valid_loader, test_loader)

class CustomCNN(nn.Module):
    """A minimal custom CNN to ensure a baseline model runs."""

    def __init__(self):
        super(CustomCNN, self).__init__()
        self.conv_block1 = nn.Sequential(nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.conv_block2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.conv_block3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(nn.Linear(128 * 12 * 12, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, 1))

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x

def get_model():
    """
    Factory function to create and return an instance of the model.
    Uses a minimal custom CNN to debug scheduler failures.
    """
    model = CustomCNN()
    return model

def get_loss_function():
    """Returns the loss function for the task."""
    return nn.BCEWithLogitsLoss()

def train_one_epoch(model, loader, optimizer, criterion, device, scaler, amp_enabled):
    """Trains the model for one epoch."""
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, criterion, device, amp_enabled):
    """Evaluates the model on the validation set."""
    model.eval()
    all_labels = []
    all_preds = []
    total_loss = 0.0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            all_labels.extend(labels.float().cpu().numpy())
            all_preds.extend(torch.sigmoid(outputs).float().cpu().numpy())
    val_loss = total_loss / len(loader)
    val_auc = roc_auc_score(all_labels, all_preds)
    return (val_loss, val_auc)

def generate_submission(model, loader, device, amp_enabled, submission_path):
    """Generates predictions for the test set and saves them to a submission file."""
    model.eval()
    all_ids = []
    all_preds = []
    with torch.no_grad():
        for images, ids in loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.sigmoid(outputs).float().cpu().numpy().flatten()
            all_preds.extend(preds)
            all_ids.extend(ids)
    submission_df = pd.DataFrame({'id': all_ids, 'label': all_preds})
    os.makedirs(os.path.dirname(submission_path), exist_ok=True)
    submission_df.to_csv(submission_path, index=False)
    print(f'Submission file created at: {submission_path}')

def main():
    """Main function to orchestrate the training and evaluation pipeline."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    log_precision_settings()
    print('Loading data...')
    train_loader, valid_loader, test_loader = get_dataloaders(batch_size=int(_MLEVOLVE_BATCH_SIZE_OVERRIDE) if _MLEVOLVE_BATCH_SIZE_OVERRIDE is not None else BATCH_SIZE, num_workers=NUM_WORKERS)
    print('Initializing model...')
    model = get_model().to(DEVICE)
    criterion = get_loss_function()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    print(f'Model Family: {MODEL_FAMILY}')
    print(f'Device: {DEVICE}')
    best_val_auc = 0.0
    print('\nStarting training...')
    for epoch in range(min(int(_MLEVOLVE_PROBE_MAX_EPOCHS), NUM_EPOCHS) if _MLEVOLVE_PROBE_MODE and _MLEVOLVE_PROBE_MAX_EPOCHS is not None else NUM_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE, SCALER, AMP_ENABLED)
        val_loss, val_auc = evaluate(model, valid_loader, criterion, DEVICE, AMP_ENABLED)
        scheduler.step()
        print(f'Epoch {epoch + 1}/{NUM_EPOCHS} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}')
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f'  -> New best model saved with Val AUC: {best_val_auc:.4f}')
    print('Training finished.')
    del train_loader, valid_loader
    gc.collect()
    torch.cuda.empty_cache()
    print('\nLoading best model for inference...')
    model.load_state_dict(torch.load(CHECKPOINT_PATH))
    print('Generating submission file...')
    generate_submission(model, test_loader, DEVICE, AMP_ENABLED, SUBMISSION_PATH)
    print(f'Final Validation Score: {best_val_auc}')
if __name__ == '__main__':
    main()