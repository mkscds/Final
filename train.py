import os
import sys
from datetime import datetime
import psutil
import torch

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="timm.models.layers")
from sklearn.metrics import balanced_accuracy_score, accuracy_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader
import open_clip
from PIL import Image
import mobileclip
from mobileclip.modules.common.mobileone import reparameterize_model
from dataset_preparation import FedIsic2019
from loss_function import BaselineLoss, LocalBalFocalCon, prototype_loss
from utils_single import train_vlm, gen_peft_model, get_scheduler, compute_epoch_metrics
from Config import parse_arguments, print_config


import os
import sys
from datetime import datetime
import psutil
import torch
import pandas as pd
import numpy as np

# Create timestamped output directory at the start
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"./training_runs/run_{timestamp}"
os.makedirs(output_dir, exist_ok=True)

# Set up logging to both file and console
log_file_path = os.path.join(output_dir, "training.log")
resource_log_path = os.path.join(output_dir, "resource_usage.log")

class Logger:
    def __init__(self, log_file):
        self.terminal = sys.stdout
        self.log = open(log_file, "a")
   
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()

# Redirect stdout to logger
sys.stdout = Logger(log_file_path)
sys.stderr = Logger(log_file_path)

print(f"Training run started at: {timestamp}")
print(f"Output directory: {output_dir}")
print("="*60)

# Resource monitoring utility functions
def get_ram_usage():
    """Get current RAM usage in GB"""
    process = psutil.Process()
    ram_gb = process.memory_info().rss / (1024 ** 3)
    return ram_gb

def get_system_ram():
    """Get total system RAM usage"""
    ram = psutil.virtual_memory()
    return {
        'total_gb': ram.total / (1024 ** 3),
        'used_gb': ram.used / (1024 ** 3),
        'percent': ram.percent
    }

def get_gpu_usage():
    """Get GPU memory usage"""
    if torch.cuda.is_available():
        gpu_stats = {}
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
            reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
            total = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
            gpu_stats[f'gpu_{i}'] = {
                'allocated_gb': allocated,
                'reserved_gb': reserved,
                'total_gb': total,
                'percent': (allocated / total) * 100 if total > 0 else 0
            }
        return gpu_stats
    return {}

def log_resource_usage(message="", file_path=None):
    """Log current resource usage"""
    ram_usage = get_ram_usage()
    system_ram = get_system_ram()
    gpu_usage = get_gpu_usage()
    
    log_msg = f"\n{'='*60}\n"
    log_msg += f"Resource Usage - {message}\n"
    log_msg += f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    log_msg += f"{'='*60}\n"
    log_msg += f"Process RAM: {ram_usage:.2f} GB\n"
    log_msg += f"System RAM: {system_ram['used_gb']:.2f} / {system_ram['total_gb']:.2f} GB ({system_ram['percent']:.1f}%)\n"
    
    if gpu_usage:
        for gpu_name, stats in gpu_usage.items():
            log_msg += f"{gpu_name.upper()}: {stats['allocated_gb']:.2f} / {stats['total_gb']:.2f} GB ({stats['percent']:.1f}%)\n"
            log_msg += f"  Reserved: {stats['reserved_gb']:.2f} GB\n"
    else:
        log_msg += "GPU: Not available\n"
    
    log_msg += f"{'='*60}\n"
    
    print(log_msg)
    
    # Also write to resource log file
    if file_path:
        with open(file_path, 'a') as f:
            f.write(log_msg)
    
    return ram_usage, gpu_usage

# Initial resource usage
log_resource_usage("Initial", resource_log_path)

# Parse arguments
args = parse_arguments()

# Update model_save_dir to use the timestamped folder
args.model_save_dir = output_dir

print_config(args)
device = args.device
data_path = './data'
model_name = args.model_name

print("="*60)
print(f"Training Configuration:")
print(f"  Model: {model_name}")
print(f"  Loss Function: {args.loss_function}")
print(f"  Optimizer: {args.optimizer}")
print(f"  Learning Rate: {args.learning_rate}")
print(f"  Batch Size: {args.batch_size}")
print(f"  Epochs: {args.epochs}")
print(f"  Device: {device}")
print(f"  LoRA r_param: {args.r_param}")
print(f"  Output Directory: {output_dir}")
print("="*60)

# model_name = "CLIP"  # Change to "S2" to load the S2 model

if model_name == "S0":
    path_s0 = '/mnt/data/home/shashank/.cache/huggingface/hub/models--apple--MobileCLIP2-S0/snapshots/3136ea51c8ed56b9f9abfab04cb816735aaad6cb/mobileclip2_s0.pt'
    model, _, preprocess = open_clip.create_model_and_transforms('MobileCLIP2-S0', pretrained=path_s0)
    tokenizer = open_clip.get_tokenizer('MobileCLIP2-S0')

elif model_name == "S2":
    path_s2 = '/mnt/data/home/shashank/.cache/huggingface/hub/models--apple--MobileCLIP2-S2/snapshots/72424e7025436db18f15c3eff6ee8c7c15ad4481/mobileclip2_s2.pt'
    model, _, preprocess = open_clip.create_model_and_transforms('MobileCLIP2-S2', pretrained=path_s2)
    tokenizer = open_clip.get_tokenizer('MobileCLIP2-S2')

elif model_name == "CLIP":
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    tokenizer = open_clip.get_tokenizer('ViT-B-32')

elif model_name == "CNN":
    pass


# Load datasets
print("\nLoading datasets...")
train_dataset = FedIsic2019(train=True, data_path=data_path, pooled=False, cid=0, sz=224, clip_tokenizer=tokenizer)
test_dataset = FedIsic2019(train=False, data_path=data_path, pooled=False, cid=0, sz=224, clip_tokenizer=tokenizer)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

print(f"  Train samples: {len(train_dataset)}")
print(f"  Test samples: {len(test_dataset)}")

# Compute class frequencies (moved before text tokens to fix order)
all_labels = train_dataset.df.target.values
local_class_freq = torch.bincount(torch.tensor(all_labels), minlength=8).float()
local_class_freq = local_class_freq / local_class_freq.sum()  # Normalize to probabilities
local_class_freq = local_class_freq.to(device)

print("\nClass distribution:")
for i in range(8):
    print(f"  Class {i}: {local_class_freq[i].item():.4f}")

# Get text tokens (BEFORE creating LoRA model)
text_tokens = train_dataset.text_tokens
text_tokens = text_tokens.to(device)

# Create LoRA model
print(f"\nCreating LoRA model with r={args.r_param}...")
model = gen_peft_model(model, args)
model = model.to(device)

# Log resource usage after loading model
log_resource_usage("After Loading Model", resource_log_path)

# Count trainable parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Total parameters: {total_params:,}")
print(f"  Trainable parameters: {trainable_params:,}")
print(f"  Trainable %: {100 * trainable_params / total_params:.2f}%")

# Train the model
print("\n" + "="*60)
print("Starting training...")
print("="*60 + "\n")

history = train_vlm(
    model=model,
    train_loader=train_loader,
    val_loader=test_loader,
    text_tokens=text_tokens,
    args=args
)

# Log resource usage after training
log_resource_usage("After Training", resource_log_path)

# Plot training curves
try:
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Loss
    axes[0].plot(history['train_loss'], label='Train', marker='o')
    axes[0].plot(history['val_loss'], label='Val', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy
    axes[1].plot(history['train_acc'], label='Train', marker='o')
    axes[1].plot(history['val_acc'], label='Val', marker='s')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    # Balanced Accuracy
    axes[2].plot(history['train_bacc'], label='Train', marker='o')
    axes[2].plot(history['val_bacc'], label='Val', marker='s')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Balanced Accuracy')
    axes[2].set_title('Training and Validation Balanced Accuracy')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nTraining curves saved to: {plot_path}")
    plt.close()
except ImportError:
    print("\nMatplotlib not available, skipping plot generation")

# ========================================================================
# FIXED: Load best model and evaluate
# ========================================================================
print("\n" + "="*60)
print("Loading best model for final evaluation...")
print("="*60)

best_model_path = os.path.join(output_dir, "best_model.pth")
if os.path.exists(best_model_path):
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"\nBest model loaded from epoch {checkpoint['epoch']}")
    print(f"Best validation balanced accuracy: {checkpoint['best_val_bacc']:.4f}")
    
    # ===== FIXED: Recompute prototypes for final evaluation =====
    print("\nRecomputing text prototypes for final evaluation...")
    with torch.no_grad():
        local_prototypes = model.encode_text(text_tokens)
        local_prototypes = F.normalize(local_prototypes, dim=-1)
    
    # ===== FIXED: Reinitialize loss function if needed =====
    loss_fn = None
    if args.loss_function == 'hybrid':
        loss_fn = LocalBalFocalCon(
            num_classes=8, 
            temperature=getattr(args, 'temperature', 0.07),
            gamma=getattr(args, 'gamma', 2.0),
            eps=1e-6,
            use_multi_positives=getattr(args, 'use_multi_positives', True),
            use_balanced_denom=getattr(args, 'use_balanced_denom', False),
            use_focal=getattr(args, 'use_focal', True),
            use_dynamic_tau=getattr(args, 'use_dynamic_tau', False)
        )
    # For 'clip' and 'prototype', loss_fn stays None (functions are used directly)
    
    # Final evaluation on test set
    model.eval()
    final_metrics = compute_epoch_metrics(
        model=model,
        data_loader=test_loader,
        local_prototypes=local_prototypes,
        args=args,
        loss_fn=loss_fn,
        local_class_freq=local_class_freq,
        optimizer=None,  # Not needed for evaluation
        is_training=False
    )
    
    print("\nFinal Test Set Performance:")
    print(f"  Loss: {final_metrics['loss']:.4f}")
    print(f"  Accuracy: {final_metrics['accuracy']:.4f}")
    print(f"  Balanced Accuracy: {final_metrics['balanced_accuracy']:.4f}")
    
    # ===== OPTIONAL: Per-class metrics =====
    print("\nComputing per-class metrics...")
    from sklearn.metrics import classification_report, confusion_matrix
    
    # Collect all predictions, labels, and image IDs
    all_preds = []
    all_labels = []
    all_image_ids = []
    all_probs = []  # Store prediction probabilities
    
    with torch.no_grad():
        for images, labels, batch_text_tokens in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Get predictions
            image_features = model.encode_image(images)
            image_features = F.normalize(image_features, dim=-1)
            logits = image_features @ local_prototypes.T
            
            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            all_probs.append(probs.cpu())
    
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_probs = torch.cat(all_probs).numpy()
    
    # Get image IDs from test dataset
    test_image_ids = [os.path.basename(path).replace('.jpg', '') for path in test_dataset.image_paths]
    
    # Create detailed predictions DataFrame
    print("\nCreating predictions CSV...")
    
    # Map class indices to names
    idx2label = test_dataset.idx2label
    
    predictions_data = {
        'image_id': test_image_ids,
        'true_label': all_labels,
        'true_label_name': [idx2label[label] for label in all_labels],
        'predicted_label': all_preds,
        'predicted_label_name': [idx2label[pred] for pred in all_preds],
        'correct': (all_preds == all_labels).astype(int),
        'center': test_dataset.centers
    }
    
    # Add probability columns for each class
    for i in range(8):
        predictions_data[f'prob_class_{i}_{idx2label[i]}'] = all_probs[:, i]
    
    # Add max probability (confidence)
    predictions_data['confidence'] = all_probs.max(axis=1)
    
    predictions_df = pd.DataFrame(predictions_data)
    
    # Save full predictions
    predictions_path = os.path.join(output_dir, 'predictions_full.csv')
    predictions_df.to_csv(predictions_path, index=False)
    print(f"Full predictions saved to: {predictions_path}")
    
    # Create and save correct predictions only
    correct_df = predictions_df[predictions_df['correct'] == 1]
    correct_path = os.path.join(output_dir, 'predictions_correct.csv')
    correct_df.to_csv(correct_path, index=False)
    print(f"Correct predictions saved to: {correct_path} ({len(correct_df)} samples)")
    
    # Create and save incorrect predictions only
    incorrect_df = predictions_df[predictions_df['correct'] == 0]
    incorrect_path = os.path.join(output_dir, 'predictions_incorrect.csv')
    incorrect_df.to_csv(incorrect_path, index=False)
    print(f"Incorrect predictions saved to: {incorrect_path} ({len(incorrect_df)} samples)")
    
    # Create summary statistics
    print("\nPrediction Summary:")
    print(f"  Total samples: {len(predictions_df)}")
    print(f"  Correct: {len(correct_df)} ({len(correct_df)/len(predictions_df)*100:.2f}%)")
    print(f"  Incorrect: {len(incorrect_df)} ({len(incorrect_df)/len(predictions_df)*100:.2f}%)")
    print(f"  Average confidence (correct): {correct_df['confidence'].mean():.4f}")
    print(f"  Average confidence (incorrect): {incorrect_df['confidence'].mean():.4f}")
    
    # Save per-class accuracy
    class_accuracy = []
    for i in range(8):
        class_mask = predictions_df['true_label'] == i
        class_correct = predictions_df[class_mask]['correct'].sum()
        class_total = class_mask.sum()
        class_acc = class_correct / class_total if class_total > 0 else 0
        class_accuracy.append({
            'class_id': i,
            'class_name': idx2label[i],
            'correct': class_correct,
            'total': class_total,
            'accuracy': class_acc
        })
    
    class_acc_df = pd.DataFrame(class_accuracy)
    class_acc_path = os.path.join(output_dir, 'class_accuracy.csv')
    class_acc_df.to_csv(class_acc_path, index=False)
    print(f"\nPer-class accuracy saved to: {class_acc_path}")
    
    # Print per-class accuracy
    print("\nPer-Class Accuracy:")
    for row in class_accuracy:
        print(f"  {row['class_name']:30s}: {row['correct']:4d}/{row['total']:4d} ({row['accuracy']*100:5.2f}%)")
    
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    
    # Print and save classification report
    print("\nClassification Report:")
    report = classification_report(all_labels, all_preds, 
                               target_names=[f'Class {i}' for i in range(8)],
                               digits=4)
    print(report)
    
    # Save classification report to file
    report_path = os.path.join(output_dir, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Classification report saved to: {report_path}")
    
    # Print confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)
    
    # Save confusion matrix as text
    cm_text_path = os.path.join(output_dir, 'confusion_matrix.txt')
    with open(cm_text_path, 'w') as f:
        f.write("Confusion Matrix:\n")
        f.write(str(cm))
    
    # Save confusion matrix plot
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=[f'C{i}' for i in range(8)],
                   yticklabels=[f'C{i}' for i in range(8)])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Confusion Matrix')
        
        cm_path = os.path.join(output_dir, 'confusion_matrix.png')
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {cm_path}")
        plt.close()
    except ImportError:
        print("\nSeaborn not available, skipping confusion matrix plot")
    
else:
    print(f"\nWarning: Best model not found at {best_model_path}")

# Final resource usage summary
print("\n" + "="*60)
print("FINAL RESOURCE USAGE SUMMARY")
print("="*60)
final_ram, final_gpu = log_resource_usage("Final", resource_log_path)

# Calculate peak usage (if available)
try:
    if torch.cuda.is_available():
        print("\nPeak GPU Memory Usage:")
        for i in range(torch.cuda.device_count()):
            peak_allocated = torch.cuda.max_memory_allocated(i) / (1024 ** 3)
            peak_reserved = torch.cuda.max_memory_reserved(i) / (1024 ** 3)
            print(f"  GPU {i}:")
            print(f"    Peak Allocated: {peak_allocated:.2f} GB")
            print(f"    Peak Reserved: {peak_reserved:.2f} GB")
            
            # Log to resource file
            with open(resource_log_path, 'a') as f:
                f.write(f"\nPeak GPU {i} Memory:\n")
                f.write(f"  Allocated: {peak_allocated:.2f} GB\n")
                f.write(f"  Reserved: {peak_reserved:.2f} GB\n")
except Exception as e:
    print(f"Could not retrieve peak GPU memory: {e}")

print("\n" + "="*60)
print("Training completed successfully!")
print(f"All outputs saved to: {output_dir}")
print("="*60)

# Close the log file
sys.stdout.log.close()
sys.stderr.log.close()