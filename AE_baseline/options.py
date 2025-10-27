# 학습 옵션 및 설정

import os
import argparse
import re
from datetime import datetime
import glob


class Options:
    def __init__(self, isTrain):
        self.project_name = None
        self.dataset = None
        self.fold = None
        self.result_dir = None
        self.isTrain = isTrain
        self.model = dict()
        self.train = dict()
        self.test = dict()
        self.transform = dict()
        self.post = dict()
        self.gpu = None
        # self.tags = None
        # self.notes = None

        self.data_name = {'cq500': 'CQ500'}
        self.epochs = {'cq500': 250}
        self.in_c = {'cq500': 1}

    def parse(self):
        """ Parse the options, replace the default value if there is a new input """
        parser = argparse.ArgumentParser(description='')
        
        parser.add_argument("-d", '--dataset', type=str, default='cq500',)                    
        parser.add_argument("-g", '--gpu', type=int, default=0)
        parser.add_argument("-p", '--project-name', type=str, default="sw_capstone", required=False,)
        parser.add_argument("-n", '--notes', type=str, default="default", required=False,
                            help='Notes of the current experiment. e.g., ae-architecture')
        parser.add_argument("-f", '--fold', type=str, default='0', help='0-4, five fold cross validation')
        parser.add_argument("-m", '--model-name', type=str, default='ae', help='ae, aeu, memae')
        parser.add_argument('--input-size', type=int, default=64, help='input size of the image')

        # Parameters only for reconstruction model
        parser.add_argument('--base-width', type=int, default=16,
                            help='Base channels of CNN layers. Please do not modify this value, and adjust the '
                                 'expansion instead.')
        parser.add_argument('--expansion', type=int, default=1, help='expansion of the base channels.')
        parser.add_argument('--hidden-num', type=int, default=1024, help='Hidden size of the bottleneck')
        parser.add_argument("-ls", '--latent-size', type=int, default=16,
                            help='latent size of the reconstruction model')
        parser.add_argument('--en-depth', type=int, default=1, help='Depth of each encoder block')
        parser.add_argument('--de-depth', type=int, default=1, help='Depth of each decoder block')

        parser.add_argument('--train-epochs', type=int, default=250, help='number of training epochs')
        parser.add_argument('--train-eval-freq', type=int, default=999, help='epoch to evaluate')
        parser.add_argument('-bs', '--train-batch-size', type=int, default=64, help='batch size')
        parser.add_argument('--train-lr', type=float, default=1e-3, help='initial learning rate')
        parser.add_argument('--train-weight-decay', type=float, default=0, help='weight decay')
        parser.add_argument('--train-seed', type=int, default=None, help='random seed')

        parser.add_argument("-save", '--test-save-flag', action='store_true')
        parser.add_argument('--test-model-path', type=str, default=None, help='model path to test')

        args = parser.parse_args()

        self.gpu = args.gpu
        self.dataset = args.dataset
        self.project_name = args.project_name
        self.fold = args.fold
        self.result_dir = os.path.join(f"C:/SW_Capstone_Design/AE_baseline/Experiment/{self.dataset}")

        self.model['name'] = args.model_name
        self.model['in_c'] = self.in_c.setdefault(self.dataset, 1)
        self.model['input_size'] = args.input_size

        # Parameters only for reconstruction model
        self.model['base_width'] = args.base_width
        self.model['expansion'] = args.expansion
        self.model['hidden_num'] = args.hidden_num
        self.model['ls'] = args.latent_size
        self.model['en_depth'] = args.en_depth
        self.model['de_depth'] = args.de_depth

        # --- training params --- #
        if self.isTrain:
            # 날짜별 + 순서별 폴더 생성
            self.train['save_dir'] = self._get_experiment_dir()
        else:
            # 테스트 모드: fold 방식 유지 (기존 호환성)
            self.train['save_dir'] = '{}/{}/fold_{}'.format(self.result_dir, self.model['name'], self.fold)
        
        self.train['epochs'] = self.epochs.setdefault(self.dataset, 250)
        self.train['eval_freq'] = args.train_eval_freq
        self.train['batch_size'] = args.train_batch_size
        self.train['lr'] = args.train_lr
        self.train['weight_decay'] = args.train_weight_decay
        self.train['seed'] = args.train_seed

        # --- test parameters --- #
        self.test['save_flag'] = args.test_save_flag
        self.test['save_dir'] = '{:s}/test_results'.format(self.train['save_dir'])
        if not args.test_model_path:
            self.test['model_path'] = f'{self.train["save_dir"]}/checkpoints/model.pt'
        
        # Load model parameters from train_options.txt for testing
        if not self.isTrain:
            self.load_train_options()

    def save_options(self):
        if not os.path.exists(self.train['save_dir']):
            os.makedirs(self.train['save_dir'], exist_ok=True)
            os.makedirs(os.path.join(self.train['save_dir'], 'test_results'), exist_ok=True)
            os.makedirs(os.path.join(self.train['save_dir'], 'checkpoints'), exist_ok=True)

        filename = '{:s}/train_options.txt'.format(self.train['save_dir'])
        file = open(filename, 'w')
        groups = ['model', 'train', 'test', 'transform']

        file.write("# ---------- Options ---------- #")
        file.write('\ndataset: {:s}\n'.format(self.dataset))
        file.write('isTrain: {}\n'.format(self.isTrain))
        for group, options in self.__dict__.items():
            if group not in groups:
                continue
            file.write('\n\n-------- {:s} --------\n'.format(group))
            if group == 'transform':
                for name, val in options.items():
                    if (self.isTrain and name != 'test') or (not self.isTrain and name == 'test'):
                        file.write("{:s}:\n".format(name))
                        for t_val in val.transforms:
                            file.write("\t{:s}\n".format(t_val.__class__.__name__))
            else:
                for name, val in options.items():
                    file.write("{:s} = {:s}\n".format(name, repr(val)))
        file.close()
    
    def load_train_options(self):
        """Load model parameters from train_options.txt for testing"""
        train_options_path = os.path.join(self.train['save_dir'], 'train_options.txt')
        
        if not os.path.exists(train_options_path):
            print(f"Warning: train_options.txt not found at {train_options_path}")
            print("Using default/command-line parameters for model architecture.")
            return
        
        print(f"=> Loading model parameters from {train_options_path}")
        
        with open(train_options_path, 'r') as f:
            content = f.read()
        
        # Parse model parameters
        model_section = re.search(r'-------- model --------\n(.*?)\n\n', content, re.DOTALL)
        if model_section:
            model_params = model_section.group(1)
            
            # Extract parameters using regex
            param_patterns = {
                'input_size': r"input_size = (\d+)",
                'base_width': r"base_width = (\d+)",
                'expansion': r"expansion = (\d+)",
                'hidden_num': r"hidden_num = (\d+)",
                'ls': r"ls = (\d+)",
                'en_depth': r"en_depth = (\d+)",
                'de_depth': r"de_depth = (\d+)",
            }
            
            for key, pattern in param_patterns.items():
                match = re.search(pattern, model_params)
                if match:
                    value = int(match.group(1))
                    self.model[key] = value
                    print(f"  Loaded {key}: {value}")
        
        print("=> Model parameters loaded successfully")
    
    def _get_experiment_dir(self):
        """
        날짜별 + 실험 순서별 디렉토리 생성
        
        Format: Experiment/{dataset}/{model}/{YYYYMMDD_NNN}/
        Example: Experiment/cq500/vae/20250127_001/
        
        Returns:
            str: 실험 디렉토리 경로
        """
        # 날짜 폴더명 생성 (YYYYMMDD)
        today = datetime.now().strftime("%Y%m%d")
        
        # 모델별 베이스 디렉토리
        base_dir = os.path.join(self.result_dir, self.model['name'])
        
        # 오늘 날짜로 시작하는 폴더 찾기
        pattern = os.path.join(base_dir, f"{today}_*")
        existing_dirs = glob.glob(pattern)
        
        if not existing_dirs:
            # 오늘 첫 번째 실험
            exp_num = 1
        else:
            # 기존 실험 번호 찾기
            exp_numbers = []
            for dir_path in existing_dirs:
                dir_name = os.path.basename(dir_path)
                try:
                    # "20250127_001" -> 1
                    num = int(dir_name.split('_')[1])
                    exp_numbers.append(num)
                except (IndexError, ValueError):
                    continue
            
            # 다음 번호 할당
            exp_num = max(exp_numbers) + 1 if exp_numbers else 1
        
        # 최종 디렉토리 경로
        exp_dir = os.path.join(base_dir, f"{today}_{exp_num:03d}")
        
        print(f"=> Experiment directory: {exp_dir}")
        return exp_dir
