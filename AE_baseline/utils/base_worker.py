"""
Base Worker for CQ500 Anomaly Detection
"""

import random
import os
import torch
import numpy as np
from torch.utils.data import DataLoader

from networks.ae import AE
from networks.aeu import AEU
from dataloaders import CQ500DataModule
from utils.losses import AELoss, AEULoss, L1Loss


class BaseWorker:
    """
    Base worker class for training and evaluation
    
    Handles:
        - GPU setup
        - Seed setting
        - Network initialization
        - DataLoader setup
        - Optimizer setup
        - Checkpointing
    """
    
    def __init__(self, opt):
        self.opt = opt
        self.seed = None
        self.data_module = None
        self.train_loader = None
        self.test_loader = None
        self.scheduler = None
        self.optimizer = None

        self.net = None
        self.criterion = None

    def set_gpu_device(self):
        """Set GPU device"""
        if torch.cuda.is_available():
            torch.cuda.set_device(self.opt.gpu)
            print(f"=> Set GPU device: {self.opt.gpu}")
        else:
            print("=> CUDA not available, using CPU")

    def set_seed(self):
        """Set random seed for reproducibility"""
        self.seed = self.opt.train['seed']
        if self.seed is None:
            self.seed = np.random.randint(1, 999999)
        
        random.seed(self.seed)
        os.environ['PYTHONHASHSEED'] = str(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
        
        print(f"=> Set random seed: {self.seed}")

    def set_network_loss(self):
        """Initialize network and loss function"""
        
        # Network
        if self.opt.model['name'] in ['ae', 'ae-grad', 'ae-l1']:
            self.net = AE(
                input_size=self.opt.model['input_size'],
                in_planes=self.opt.model['in_c'],
                base_width=self.opt.model['base_width'],
                expansion=self.opt.model['expansion'],
                mid_num=self.opt.model['hidden_num'],
                latent_size=self.opt.model['ls'],
                en_num_layers=self.opt.model["en_depth"],
                de_num_layers=self.opt.model["de_depth"]
            )
            
            # Loss function
            if self.opt.model['name'] == 'ae-l1':
                self.criterion = L1Loss()
            elif self.opt.model['name'] == 'ae-grad':
                self.criterion = AELoss(grad_score=True)
            else:
                self.criterion = AELoss()
                
        elif self.opt.model['name'] == 'aeu':
            self.net = AEU(
                input_size=self.opt.model['input_size'],
                in_planes=self.opt.model['in_c'],
                base_width=self.opt.model['base_width'],
                expansion=self.opt.model['expansion'],
                mid_num=self.opt.model['hidden_num'],
                latent_size=self.opt.model['ls'],
                en_num_layers=self.opt.model["en_depth"],
                de_num_layers=self.opt.model["de_depth"]
            )
            self.criterion = AEULoss()
            
        elif self.opt.model['name'] == 'vae':
            from networks.vae import VAE
            from utils.losses import VAELoss
            self.net = VAE(
                input_size=self.opt.model['input_size'],
                in_planes=self.opt.model['in_c'],
                base_width=self.opt.model['base_width'],
                expansion=self.opt.model['expansion'],
                mid_num=self.opt.model['hidden_num'],
                latent_size=self.opt.model['ls'],
                en_num_layers=self.opt.model["en_depth"],
                de_num_layers=self.opt.model["de_depth"]
            )
            kl_weight = self.opt.model.get('kl_weight', 0.005)
            self.criterion = VAELoss(kl_weight=kl_weight)
            
        elif self.opt.model['name'] == 'unet':
            from networks.unet import UNet
            self.net = UNet(
                in_channels=self.opt.model['in_c'],
                n_classes=self.opt.model['in_c'],  # reconstruction task
                depth=self.opt.model.get('unet_depth', 5),
                wf=self.opt.model.get('unet_wf', 6),
                padding=True,
                norm=self.opt.model.get('unet_norm', 'group'),
                up_mode=self.opt.model.get('unet_up_mode', 'upconv')
            )
            self.criterion = AELoss()  # Use same reconstruction loss as AE
            
        else:
            raise NotImplementedError(f"Model {self.opt.model['name']} not implemented")
        
        if torch.cuda.is_available():
            self.net = self.net.cuda()
        
        print(f"=> Network: {self.opt.model['name']}")
        print(f"=> Loss function: {self.criterion.__class__.__name__}")

    def set_optimizer(self):
        """Initialize optimizer"""
        self.optimizer = torch.optim.Adam(
            self.net.parameters(),
            lr=self.opt.train['lr'],
            weight_decay=self.opt.train['weight_decay']
        )
        print(f"=> Optimizer: Adam (lr={self.opt.train['lr']}, wd={self.opt.train['weight_decay']})")

    def set_dataloader(self, test=False):
        """
        Setup DataLoader using CQ500DataModule
        
        Args:
            test: If True, only setup test loader
        """
        print(f"=> Setting up CQ500 DataLoader...")
        
        # Initialize DataModule
        self.data_module = CQ500DataModule(
            data_root="./CQ500",
            label_csv="./CQ500/cq500_labels.csv",
            batch_size=self.opt.train['batch_size'],
            img_size=self.opt.model['input_size'],
            num_workers=4 if torch.cuda.is_available() else 0,
            train_ratio=0.7,
            val_ratio=0.15,
            use_augmentation=True,
            window_center=40,
            window_width=80,
            slice_range=None,
            use_3d_volume=False,  # 2D direct load
            seed=self.seed
        )
        
        if not test:
            # Setup train and val
            self.data_module.setup(stage='fit')
            self.train_loader = self.data_module.train_dataloader()
            print(f"=> Train: {len(self.data_module.train_dataset)} slices")
        
        # Setup test
        self.data_module.setup(stage='test')
        self.test_loader = self.data_module.test_dataloader()
        print(f"=> Test: {len(self.data_module.test_dataset)} slices")
        print(f"=> Test batch size: {self.opt.train['batch_size']}")

    def save_checkpoint(self):
        """Save model checkpoint"""
        checkpoint_path = os.path.join(self.opt.train['save_dir'], "checkpoints", "model.pt")
        torch.save(self.net.state_dict(), checkpoint_path)
        print(f"=> Saved checkpoint to {checkpoint_path}")

    def load_checkpoint(self):
        """Load model checkpoint"""
        checkpoint_path = os.path.join(self.opt.train['save_dir'], "checkpoints", "model.pt")
        
        if torch.cuda.is_available():
            self.net.load_state_dict(torch.load(checkpoint_path, map_location=f"cuda:{self.opt.gpu}"))
        else:
            self.net.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        
        print(f"=> Loaded checkpoint from {checkpoint_path}")

    def close_network_grad(self):
        """Disable gradient computation"""
        for param in self.net.parameters():
            param.requires_grad = False

    def enable_network_grad(self):
        """Enable gradient computation"""
        for param in self.net.parameters():
            param.requires_grad = True

    def run_eval(self):
        """Run evaluation and save metrics"""
        results = self.evaluate()
        metrics_save_path = os.path.join(self.opt.train['save_dir'], "metrics.txt")
        
        with open(metrics_save_path, "w") as f:
            for key, value in results.items():
                f.write(str(key) + ": " + str(value) + "\n")
                print(f"{key}: {value:.4f}")
        
        print(f"=> Saved metrics to {metrics_save_path}")

    def evaluate(self) -> dict:
        """Override in subclass"""
        raise NotImplementedError("Implement in subclass")