"""
Test 코드 for AE/AE-U with CQ500
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from options import Options
from utils.ae_worker import AEWorker
from utils.aeu_worker import AEUWorker


def main():
    """Main testing function"""
    
    print("=" * 70)
    print("CQ500 Anomaly Detection - Testing")
    print("=" * 70)
    
    # Parse options
    opt = Options(isTrain=False)
    opt.parse()
    
    print(f"\nDataset: {opt.dataset}")
    print(f"Model: {opt.model['name']}")
    print(f"Model path: {opt.test['model_path']}")
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
    worker.set_dataloader(test=True)
    worker.load_checkpoint()
    
    print("\n" + "=" * 70)
    print("Start Testing")
    print("=" * 70)
    
    # Evaluate
    worker.run_eval()
    
    print("\n" + "=" * 70)
    print("Testing Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()