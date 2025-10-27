"""
AE-U Worker for training and evaluation with uncertainty
"""

import time
import numpy as np
import torch
from sklearn import metrics

from utils.ae_worker import AEWorker
from utils.util import AverageMeter


class AEUWorker(AEWorker):
    """
    Worker class for Autoencoder with Uncertainty
    
    Extends AEWorker to handle uncertainty estimation (log_var)
    """
    
    def __init__(self, opt):
        super(AEUWorker, self).__init__(opt)

    def train_epoch(self):
        """
        Train for one epoch (with uncertainty loss)
        
        Returns:
            Tuple of (total_loss, recon_loss, log_var_loss) averages
        """
        self.net.train()
        losses = AverageMeter()
        recon_losses = AverageMeter()
        log_var_losses = AverageMeter()
        
        for idx_batch, data_batch in enumerate(self.train_loader):
            img = data_batch['img']
            img = img.cuda()

            net_out = self.net(img)
            
            # AEU loss returns (total_loss, recon_loss, log_var_mean)
            loss, recon_loss, log_var_mean = self.criterion(img, net_out)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            losses.update(loss.item(), img.size(0))
            recon_losses.update(recon_loss, img.size(0))
            log_var_losses.update(log_var_mean, img.size(0))
        
        return losses.avg, recon_losses.avg, log_var_losses.avg
    
    def run_train(self):
        """
        Main training loop for AE-U
        
        Handles logging of multiple loss components and checkpointing
        """
        import os
        
        num_epochs = self.opt.train['epochs']
        print("=> Initial learning rate: {:g}".format(self.opt.train['lr']))
        
        best_val_loss = float('inf')
        log_interval = 10
        epoch_losses = []
        epoch_recon_losses = []
        epoch_logvar_losses = []
        epoch_times = []
        
        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()
            
            # Train for one epoch
            train_loss, recon_loss, logvar_loss = self.train_epoch()
            epoch_time = time.time() - epoch_start
            
            # Accumulate metrics
            epoch_losses.append(train_loss)
            epoch_recon_losses.append(recon_loss)
            epoch_logvar_losses.append(logvar_loss)
            epoch_times.append(epoch_time)
            
            # Log every 10 epochs or at the end
            if epoch % log_interval == 0 or epoch == num_epochs:
                avg_loss = sum(epoch_losses) / len(epoch_losses)
                avg_recon = sum(epoch_recon_losses) / len(epoch_recon_losses)
                avg_logvar = sum(epoch_logvar_losses) / len(epoch_logvar_losses)
                total_time = sum(epoch_times)
                avg_time = total_time / len(epoch_times)
                
                print(f"Epoch [{epoch:3d}/{num_epochs}] "
                      f"Loss: {avg_loss:.4f} (Recon: {avg_recon:.4f}, LogVar: {avg_logvar:.4f}) "
                      f"Time: {avg_time:.2f}s")
                
                # Reset accumulators
                epoch_losses = []
                epoch_recon_losses = []
                epoch_logvar_losses = []
                epoch_times = []
            
            # Checkpoint and evaluate
            if epoch % self.opt.train['eval_freq'] == 0 or epoch == num_epochs:
                self.save_checkpoint()
                
                # Evaluate on test set
                if self.opt.train.get('run_eval', True):
                    print("\nEvaluating...")
                    try:
                        results = self.evaluate()
                        print(f"AUROC: {results['AUROC']:.4f}, AUPRC: {results['AUPRC']:.4f}\n")
                    except Exception as e:
                        print(f"Evaluation failed: {e}")
                        print("Continuing training...\n")
                
                # Save best model
                if train_loss < best_val_loss:
                    best_val_loss = train_loss
                    best_checkpoint = os.path.join(self.opt.train['save_dir'], "checkpoints", "best_model.pt")
                    torch.save(self.net.state_dict(), best_checkpoint)
                    print(f"=> Saved best model (loss: {best_val_loss:.4f})")

        self.save_checkpoint()
        print(f"\n{'='*70}")
        print(f"Training complete! Best loss: {best_val_loss:.4f}")
        print(f"{'='*70}")