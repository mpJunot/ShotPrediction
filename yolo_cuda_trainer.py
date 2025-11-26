from ultralytics import YOLO
import argparse
from sys import platform
import torch

class YOLOTrainer:
    def __init__(self, data, epochs, size, batch_size=16, workers=8):
        self.epoch = epochs
        self.size = size
        self.batch_size = batch_size
        self.workers = workers

        if platform in ['linux', 'win32'] and torch.cuda.is_available():
            self.device = 'cuda'
            self._optimize_cuda()
        elif platform == 'darwin':
            self.device = 'mps'
        else:
            self.device = 'cpu'

        print(f'Using Device: {self.device}')
        if self.device == 'cuda':
            print(f'   GPU: {torch.cuda.get_device_name(0)}')
            print(f'   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB')

        self.model = "yolov8m.pt"
        self.data = data

    def _optimize_cuda(self):
        """Enable CUDA optimizations for faster training"""
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.set_float32_matmul_precision('high')

        print('CUDA optimizations enabled')

    def train(self):
        """Start training with optimized parameters"""
        print(f'Training {self.epoch} epochs with optimized settings')

        model = YOLO(self.model)

        training_args = {
            # Core parameters
            'data': self.data,                    # Dataset YAML file path
            'epochs': self.epoch,                 # Number of training epochs
            'imgsz': self.size,                   # Input image size (640, 800, 1280)
            'device': self.device,                # Device: 'cuda', 'mps', or 'cpu'
            'batch': self.batch_size,             # Batch size (images per iteration)
            'workers': self.workers,               # Data loading workers

            # Performance optimizations
            'amp': True,                          # Automatic Mixed Precision (FP16) for faster training
            'cache': True,                        # Cache images in RAM for faster loading

            # Optimizer settings
            'optimizer': 'AdamW',                 # Optimizer: 'SGD', 'Adam', or 'AdamW'
            'cos_lr': True,                       # Cosine learning rate scheduler

            # Data augmentation
            'hsv_h': 0.015,                       # HSV hue augmentation (±1.5%)
            'hsv_s': 0.7,                         # HSV saturation augmentation (±70%)
            'hsv_v': 0.4,                         # HSV value/brightness augmentation (±40%)
            'degrees': 0.0,                       # Rotation angle (degrees)
            'translate': 0.1,                     # Translation factor (±10%)
            'scale': 0.5,                         # Scaling factor (±50%)
            'shear': 0.0,                         # Shear angle (degrees)
            'flipud': 0.0,                       # Vertical flip probability
            'fliplr': 0.5,                       # Horizontal flip probability (50%)
            'mosaic': 1.0,                        # Mosaic augmentation probability (100%)
            'close_mosaic': 10,                   # Disable mosaic in last 10 epochs

            # Training control
            'patience': 50,                       # Early stopping patience (epochs)
            'save': True,                         # Save checkpoints
            'save_period': 10,                    # Save checkpoint every N epochs
            'plots': True,                        # Generate training plots
            'verbose': True,                      # Verbose output
        }

        model.train(**training_args)

        print('Training completed!')
        print(f'Results saved in: runs/detect/train')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='YOLOv8 Trainer with CUDA Optimizations')
    parser.add_argument('-d', '--data', type=str, default='data.yaml',
                        help='Path to dataset YAML file')
    parser.add_argument('-e', '--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('-i', '--size', type=int, default=640,
                        help='Image size (640, 800, 1280, etc.)')
    parser.add_argument('-b', '--batch', type=int, default=-1,
                        help='Batch size (-1 for auto, 16-32 recommended for RTX GPUs)')
    parser.add_argument('-w', '--workers', type=int, default=8,
                        help='Number of dataloader workers')

    args = parser.parse_args()

    batch_size = args.batch
    if batch_size == -1:
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            if vram_gb >= 24:
                batch_size = 32
            elif vram_gb >= 12:
                batch_size = 16
            elif vram_gb >= 8:
                batch_size = 12
            else:
                batch_size = 8
            print(f'Auto-detected batch size: {batch_size} (VRAM: {vram_gb:.1f}GB)')
        else:
            batch_size = 8

    trainer = YOLOTrainer(
        data=args.data,
        epochs=args.epochs,
        size=args.size,
        batch_size=batch_size,
        workers=args.workers
    )
    trainer.train()
