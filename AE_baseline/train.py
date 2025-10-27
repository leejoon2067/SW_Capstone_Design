"""
Train 코드 for AE/AE-U with CQ500
"""

import os
import sys
import time
import torch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from options import Options
from utils.ae_worker import AEWorker
from utils.aeu_worker import AEUWorker


def main():
    """Main training function"""
    
    print("=" * 70)
    print("CQ500 Anomaly Detection - Training")
    print("=" * 70)
    
    # Parse options
    opt = Options(isTrain=True)
    opt.parse()
    opt.save_options()
    
    print(f"\nDataset: {opt.dataset}")
    print(f"Model: {opt.model['name']}")
    print(f"Input size: {opt.model['input_size']}")
    print(f"Batch size: {opt.train['batch_size']}")
    print(f"Epochs: {opt.train['epochs']}")
    print(f"Learning rate: {opt.train['lr']}")
    print()
    
    # Select worker based on model
    if opt.model['name'] in ['ae', 'ae-grad', 'ae-l1', 'unet']:
        worker = AEWorker(opt)
    elif opt.model['name'] == 'aeu':
        worker = AEUWorker(opt)
    elif opt.model['name'] == 'vae':
        from utils.vae_worker import VAEWorker
        worker = VAEWorker(opt)
    else:
        raise NotImplementedError(f"Model {opt.model['name']} not implemented")
    
    # Setup
    worker.set_seed()
    worker.set_gpu_device()
    worker.set_network_loss()
    worker.set_optimizer()
    worker.set_dataloader(test=False)
    
    print("\n" + "=" * 70)
    print("Start Training")
    print("=" * 70)
    
    # VAE/AEU는 자체 run_train() 메서드 사용 (여러 loss 출력 처리)
    if opt.model['name'] in ['vae', 'aeu']:
        worker.run_train()
    else:
        # AE, UNet 등 일반 모델 학습 루프
        best_val_loss = float('inf')
        log_interval = 10

        epoch_losses = []
        epoch_times = []
        
        for epoch in range(opt.train['epochs']):
            epoch_start = time.time()
            
            # Train
            train_loss = worker.train_epoch()
            epoch_time = time.time() - epoch_start

            # 손실과 시간 누적
            epoch_losses.append(train_loss)
            epoch_times.append(epoch_time)
            
            # 10 에포크마다 또는 마지막 에포크일 때 로그 출력
            if (epoch + 1) % log_interval == 0 or (epoch + 1) == opt.train['epochs']:
                avg_loss = sum(epoch_losses) / len(epoch_losses)
                total_time = sum(epoch_times)
                avg_time = total_time / len(epoch_times)
                
                print(f"Epoch [{epoch+1}/{opt.train['epochs']}] "
                      f"Avg Loss: {avg_loss:.4f} "
                      f"Avg Time: {avg_time:.2f}s "
                      f"Total: {total_time:.2f}s")
                
                # 누적 변수 초기화
                epoch_losses = []
                epoch_times = []

            # Save checkpoint
            if (epoch + 1) % opt.train['eval_freq'] == 0 or (epoch + 1) == opt.train['epochs']:
                worker.save_checkpoint()
                
                # Evaluate (선택적 - 시간이 오래 걸릴 수 있음)
                if opt.train.get('run_eval', True):  # 기본값: True
                    print("\nEvaluating...")
                    try:
                        results = worker.evaluate()
                        print(f"AUROC: {results['AUROC']:.4f}, AUPRC: {results['AUPRC']:.4f}\n")
                    except Exception as e:
                        print(f"Evaluation failed: {e}")
                        print("Continuing training...\n")
                
                # Save best model
                if train_loss < best_val_loss:
                    best_val_loss = train_loss
                    best_checkpoint = os.path.join(opt.train['save_dir'], "checkpoints", "best_model.pt")
                    torch.save(worker.net.state_dict(), best_checkpoint)
                    print(f"=> Saved best model (loss: {best_val_loss:.4f})")
        
        print("\n" + "=" * 70)
        print("Training Complete!")
        print("=" * 70)
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Model saved to: {opt.train['save_dir']}")


if __name__ == "__main__":
    main()