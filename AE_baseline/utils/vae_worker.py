"""
VAE Worker for training and evaluation
"""

import time
import os
import torch

from utils.ae_worker import AEWorker
from utils.util import AverageMeter


class VAEWorker(AEWorker):
    """
    Worker class for Variational Autoencoder training and evaluation
    
    Extends AEWorker to handle VAE-specific training loop with
    multiple loss components (reconstruction + KL divergence)
    """
    
    def __init__(self, opt):
        super(VAEWorker, self).__init__(opt)

    def train_epoch(self):
        """
        Train for one epoch
        
        Returns:
            Tuple of (total_loss, recon_loss, kl_loss) averages
        """
        self.net.train()
        losses, recon_losses, kl_losses = AverageMeter(), AverageMeter(), AverageMeter()
        
        for idx_batch, data_batch in enumerate(self.train_loader):
            img = data_batch['img']
            img = img.cuda()

            # Forward pass
            net_out = self.net(img)

            # Compute loss (returns 3 values for VAE)
            loss, recon_loss, kl_loss = self.criterion(img, net_out)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Update metrics
            bs = img.size(0)
            losses.update(loss.item(), bs)
            recon_losses.update(recon_loss, bs)
            kl_losses.update(kl_loss, bs)
            
        return losses.avg, recon_losses.avg, kl_losses.avg

    def run_train(self):
        """
        Main training loop for VAE
        
        Handles logging of multiple loss components and checkpointing
        """
        num_epochs = self.opt.train['epochs']
        print("=> Initial learning rate: {:g}".format(self.opt.train['lr']))
        
        best_val_loss = float('inf')
        log_interval = 10
        epoch_losses = []
        epoch_recon_losses = []
        epoch_kl_losses = []
        epoch_times = []
        
        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()
            
            # Train for one epoch
            train_loss, recon_loss, kl_loss = self.train_epoch()
            epoch_time = time.time() - epoch_start
            
            # Accumulate metrics
            epoch_losses.append(train_loss)
            epoch_recon_losses.append(recon_loss)
            epoch_kl_losses.append(kl_loss)
            epoch_times.append(epoch_time)
            
            # Log every 10 epochs or at the end
            if epoch % log_interval == 0 or epoch == num_epochs:
                avg_loss = sum(epoch_losses) / len(epoch_losses)
                avg_recon = sum(epoch_recon_losses) / len(epoch_recon_losses)
                avg_kl = sum(epoch_kl_losses) / len(epoch_kl_losses)
                total_time = sum(epoch_times)
                avg_time = total_time / len(epoch_times)
                
                print(f"Epoch [{epoch:3d}/{num_epochs}] "
                      f"Loss: {avg_loss:.4f} (Recon: {avg_recon:.4f}, KL: {avg_kl:.4f}) "
                      f"Time: {avg_time:.2f}s")
                
                # Reset accumulators
                epoch_losses = []
                epoch_recon_losses = []
                epoch_kl_losses = []
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

