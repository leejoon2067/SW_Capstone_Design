"""
AE Worker for training and evaluation
"""

import time
import numpy as np
import torch
from sklearn import metrics
import os

from utils.base_worker import BaseWorker
from utils.util import AverageMeter


class AEWorker(BaseWorker):
    """
    Worker class for Autoencoder training and evaluation
    
    Handles:
        - Training loop
        - Evaluation metrics (AUROC, AUPRC)
        - Checkpointing
    """
    
    def __init__(self, opt):
        super(AEWorker, self).__init__(opt)
        self.grad_flag = True if self.opt.model['name'] in ['ae-grad'] else False

    def train_epoch(self):
        """Train for one epoch"""
        self.net.train()
        losses = AverageMeter()
        
        for idx_batch, data_batch in enumerate(self.train_loader):
            img = data_batch['img']
            img = img.cuda()

            net_out = self.net(img)
            loss = self.criterion(img, net_out)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            losses.update(loss.item(), img.size(0))
        
        return losses.avg

    def evaluate(self):
        """
        Evaluate model on test set
        
        Returns:
            dict with AUROC and AUPRC metrics
        """
        self.net.eval()

        test_scores, test_labels = [], []
        
        print("Evaluating on test set...")
        
        with torch.no_grad():  # Gradient 끄기 (안전한 방법)
            for idx_batch, data_batch in enumerate(self.test_loader):
                img, label = data_batch['img'], data_batch['label']
                img = img.cuda()

                net_out = self.net(img)

                # Compute anomaly score (per sample)
                anomaly_score = self.criterion(img, net_out, anomaly_score=True, keepdim=False)
                anomaly_score = anomaly_score.detach().cpu().numpy()  # (B,)
                
                # Handle batch
                if anomaly_score.ndim == 0:
                    # Single sample
                    test_scores.append(anomaly_score.item())
                    test_labels.append(label.item())
                else:
                    # Batch
                    test_scores.extend(anomaly_score.tolist())
                    test_labels.extend(label.cpu().numpy().tolist())

            test_scores = np.array(test_scores)
            test_labels = np.array(test_labels)

        print(f"Total samples evaluated: {len(test_scores)}")

        # Compute metrics
        auroc = metrics.roc_auc_score(test_labels, test_scores)
        precision, recall, _ = metrics.precision_recall_curve(test_labels, test_scores)
        auprc = metrics.auc(recall, precision)

        results = {
            'AUROC': auroc,
            'AUPRC': auprc,
        }

        return results