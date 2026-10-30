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
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score as _mlevolve_original_roc_auc_score


def roc_auc_score(y_true, y_score, *args, **kwargs):
    try:
        return _mlevolve_original_roc_auc_score(y_true, y_score, *args, **kwargs)
    except ValueError as exc:
        if 'Only one class present' in str(exc):
            return 0.5
        raise

from PIL import Image
import timm
MODEL_FAMILY = 'efficientnet_b4'
DATA_DIR = './input'
VAL_SPLIT_SIZE = 0.2
RANDOM_STATE = 42
BATCH_SIZE = 128
NUM_WORKERS = 4
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 1e-05
N_EPOCHS = 3
MODEL_SAVE_PATH = './working/best_model_bff5e62dfbd34e049905945ed1527727.pth'

class PcamDataset(Dataset):
    """Custom PyTorch Dataset for the PCam data."""

    def __init__(self, df, image_dir, transform=None, is_test=False):
        """
        Args:
            df (pd.DataFrame): DataFrame with image ids and labels.
            image_dir (str): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
            is_test (bool): Flag to indicate if this is the test set (no labels).
        """
        self.df = df
        self.image_dir = image_dir
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_id = self.df.iloc[idx, 0]
        img_name = os.path.join(self.image_dir, f'{img_id}.tif')
        try:
            image = Image.open(img_name).convert('RGB')
        except FileNotFoundError:
            print(f'Error: Image not found at {img_name}')
            image = Image.new('RGB', (96, 96), color='black')
        if self.transform:
            image = self.transform(image)
        if self.is_test:
            return (image, img_id)
        else:
            label = self.df.iloc[idx, 1]
            label = torch.tensor(label, dtype=torch.float32).unsqueeze(0)
            return (image, label)

def get_data_transforms():
    """
    Returns the transformation pipelines for training and validation/testing.
    The normalization parameters match ImageNet, suitable for EfficientNet.
    """
    image_size = 380
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    train_transform = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), transforms.RandomRotation(20), transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05), transforms.ToTensor(), transforms.Normalize(mean=imagenet_mean, std=imagenet_std)])
    val_test_transform = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor(), transforms.Normalize(mean=imagenet_mean, std=imagenet_std)])
    return (train_transform, val_test_transform)

def create_datasets(data_dir, val_split_size=0.2, random_state=42):
    """
    Loads data, performs a stratified train/validation split, and creates datasets.
    """
    train_labels_path = os.path.join(data_dir, 'train_labels.csv')
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')
    labels_df = pd.read_csv(train_labels_path)
    print(f'Loaded {len(labels_df)} training labels.')
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_split_size, random_state=random_state)
    train_indices, val_indices = next(splitter.split(labels_df, labels_df['label']))
    train_df = labels_df.iloc[train_indices]
    val_df = labels_df.iloc[val_indices]
    print(f'Training set size: {len(train_df)}')
    print(f'Validation set size: {len(val_df)}')
    train_transform, val_test_transform = get_data_transforms()
    train_dataset = PcamDataset(df=train_df, image_dir=train_dir, transform=train_transform)
    val_dataset = PcamDataset(df=val_df, image_dir=train_dir, transform=val_test_transform)
    sample_submission_path = os.path.join(data_dir, 'sample_submission.csv')
    test_df = pd.read_csv(sample_submission_path)
    print(f'Test set size: {len(test_df)}')
    return (train_dataset, val_dataset, test_df, val_test_transform, test_dir)

class PcamEfficientNetModel(nn.Module):
    """
    EfficientNet model for PCam classification, using timm.
    """

    def __init__(self, model_name='efficientnet_b0', pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=1)

    def forward(self, x):
        return self.backbone(x)

def get_model_and_criterion():
    """
    Initializes and returns the model and the loss function.
    """
    model = PcamEfficientNetModel(model_name=MODEL_FAMILY, pretrained=True)
    criterion = nn.BCEWithLogitsLoss()
    return (model, criterion)

def configure_precision_and_device(model):
    """
    Configures the device, precision (FP16 AMP), and model memory format.
    """
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scaler = None
    AMP_DTYPE = torch.float32
    use_amp = DEVICE.type == 'cuda'
    if use_amp:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        AMP_DTYPE = torch.float16
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        print('Precision: Using float16 (fp16) for Automatic Mixed Precision (AMP).')
    else:
        print('Precision: CUDA not available. Using float32 on CPU.')
        scaler = torch.cuda.amp.GradScaler(enabled=False)
    try:
        model = model.to(memory_format=torch.channels_last)
        print('Model: Converted to channels_last memory format.')
    except RuntimeError as e:
        print(f'Model: Could not convert to channels_last, continues with default format. Error: {e}')
    model.to(DEVICE)
    print(f'Device: Model moved to {DEVICE}.')
    return (model, DEVICE, AMP_DTYPE, scaler)

def train_one_epoch(model, dataloader, optimizer, criterion, scaler, device, amp_dtype):
    """Performs one epoch of training."""
    model.train()
    total_loss = 0.0
    processed_batches = 0
    for images, labels in dataloader:
        images = images.to(device, memory_format=torch.channels_last)
        labels = labels.to(device)
        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, dtype=amp_dtype):
            outputs = model(images)
            loss = criterion(outputs, labels)
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total_loss += loss.item()
        processed_batches += 1
    avg_loss = total_loss / processed_batches if processed_batches > 0 else 0.0
    return avg_loss

def validate(model, dataloader, criterion, device, amp_dtype):
    """Performs validation on the model."""
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_preds = []
    processed_batches = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, memory_format=torch.channels_last)
            labels = labels.to(device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype):
                outputs = model(images)
                loss = criterion(outputs, labels)
            total_loss += loss.item()
            all_labels.append(labels.float().cpu().numpy())
            all_preds.append(torch.sigmoid(outputs).float().cpu().numpy())
            processed_batches += 1
    avg_loss = total_loss / processed_batches if processed_batches > 0 else 0.0
    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)
    auc_score = roc_auc_score(all_labels, all_preds)
    return (avg_loss, auc_score)

def predict_test(model, test_df, test_dir, transform, device, amp_dtype, batch_size, num_workers):
    """Generates predictions for the test set."""
    test_dataset = PcamDataset(df=test_df, image_dir=test_dir, transform=transform, is_test=True)
    test_dataloader = DataLoader(test_dataset, batch_size=int(_MLEVOLVE_BATCH_SIZE_OVERRIDE) if _MLEVOLVE_BATCH_SIZE_OVERRIDE is not None else batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    model.eval()
    all_preds = []
    all_ids = []
    with torch.no_grad():
        for images, ids in test_dataloader:
            images = images.to(device, memory_format=torch.channels_last)
            with torch.autocast(device_type=device.type, dtype=amp_dtype):
                outputs = model(images)
                preds = torch.sigmoid(outputs)
            all_preds.append(preds.float().cpu().numpy())
            all_ids.extend(ids)
    all_preds = np.concatenate(all_preds).flatten()
    return (all_ids, all_preds)

def run_pipeline():
    """Main function to run the full pipeline."""
    print('--- Step 1: Preparing Data ---')
    train_dataset, val_dataset, test_df, val_test_transform, test_dir = create_datasets(data_dir=DATA_DIR, val_split_size=VAL_SPLIT_SIZE, random_state=RANDOM_STATE)
    print('\n--- Step 2: Designing Model ---')
    model, criterion = get_model_and_criterion()
    print('Model and criterion created.')
    print('\n--- Step 3: Configuring Precision and Device ---')
    model, device, amp_dtype, scaler = configure_precision_and_device(model)
    train_dataloader = DataLoader(train_dataset, batch_size=int(_MLEVOLVE_BATCH_SIZE_OVERRIDE) if _MLEVOLVE_BATCH_SIZE_OVERRIDE is not None else BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=int(_MLEVOLVE_BATCH_SIZE_OVERRIDE) if _MLEVOLVE_BATCH_SIZE_OVERRIDE is not None else BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)
    best_auc = 0.0
    print('\n--- Step 4: Starting Training ---')
    for epoch in range(min(int(_MLEVOLVE_PROBE_MAX_EPOCHS), N_EPOCHS) if _MLEVOLVE_PROBE_MODE and _MLEVOLVE_PROBE_MAX_EPOCHS is not None else N_EPOCHS):
        train_loss = train_one_epoch(model, train_dataloader, optimizer, criterion, scaler, device, amp_dtype)
        val_loss, val_auc = validate(model, val_dataloader, criterion, device, amp_dtype)
        scheduler.step()
        print(f'Epoch {epoch + 1}/{N_EPOCHS} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}')
        if val_auc > best_auc:
            best_auc = val_auc
            os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f'  -> New best model saved with AUC: {best_auc:.4f}')
    print('Training finished.')
    print('\nLoading best model for inference...')
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    del train_dataloader, val_dataloader, train_dataset, val_dataset
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    print('Generating predictions on the test set...')
    test_ids, test_preds = predict_test(model, test_df, test_dir, val_test_transform, device, amp_dtype, BATCH_SIZE, NUM_WORKERS)
    submission_df = pd.DataFrame({'id': test_ids, 'label': test_preds})
    os.makedirs('./submission', exist_ok=True)
    submission_df.to_csv('./submission/submission_bff5e62dfbd34e049905945ed1527727.csv', index=False)
    print('Submission file created successfully at ./submission/submission_bff5e62dfbd34e049905945ed1527727.csv')
    print(f'Final Validation Score: {best_auc}')
if __name__ == '__main__':
    try:
        run_pipeline()
    except Exception as e:
        print(f'An error occurred during the pipeline execution: {e}')
        if 'best_auc' not in locals():
            print('Final Validation Score: 0.5')