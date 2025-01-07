import os
import re
import pickle
import datetime
from sklearn.metrics import r2_score
import numpy as np

class Config:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if key == 'desc':
                continue
            if isinstance(value, dict):
                if key == 'value':
                    if isinstance(value, dict):
                        for key_l, value_l in value.items():
                            if isinstance(value_l, dict):
                                value_l = Config(value_l)
                            self.__dict__[key_l] = value_l
                else:
                    if isinstance(value, dict):
                        if 'value' in value and not isinstance(value['value'], dict):
                            value = value['value']
                        else:
                            value = Config(value)
                    self.__dict__[key] = value
            elif key != 'value':
                self.__dict__[key] = value
                
    def to_dict(self):
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result
    
def get_highest_checkpoint_file(model_path, prefix):
    # Check if 'model.pkl' exists
    model_file = os.path.join(model_path, f'{prefix}_CellFlow.pkl')
    if os.path.exists(model_file):
        print(f"'{prefix}_CellFlow.pkl' file exists: {model_file}")
        return model_file
    else:
        # Search for files starting with 'checkpoint'
        checkpoint_files = [
            f for f in os.listdir(model_path)
            if f.startswith('checkpoint') and f.endswith('_CellFlow.pkl')
        ]
        if not checkpoint_files:
            print("No checkpoint files found.")
            return None
        # Extract the iteration number using a regular expression
        pattern = r'checkpoint-(\d+)_CellFlow.pkl'
        checkpoint_iters = {}
        for file in checkpoint_files:
            match = re.search(pattern, file)
            if match:
                iter_num = int(match.group(1))
                checkpoint_iters[iter_num] = file
        if not checkpoint_iters:
            print("No valid checkpoint files found.")
            return None
        # Get the file with the highest iteration number
        highest_iter = max(checkpoint_iters.keys())
        highest_checkpoint = checkpoint_iters[highest_iter]
        print(f"Selected checkpoint file: {highest_checkpoint}")
        return os.path.join(model_path, highest_checkpoint)
    
def compute_r_squared(x, y) -> float:
    """Compute the R squared between true (x) and predicted (y)"""
    # check if x or y is empty
    if len(x) == 0 or len(y) == 0:
        debugging_dir = '/home/icb/lea.zimmermann/projects/cell_flow_perturbation/debugging'
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(os.path.join(debugging_dir,f'valid_data_{timestamp}.pkl'), "wb") as f:
            pickle.dump(x, f)
        with open(os.path.join(debugging_dir,f'pred_data_{timestamp}.pkl'), "wb") as f:
            pickle.dump(y, f)
        return 0.0
    try:
        return r2_score(np.mean(x, axis=0), np.mean(y, axis=0))
    except Exception as e:
        print(f"Error caught: {e}")
        debugging_dir = '/home/icb/lea.zimmermann/projects/cell_flow_perturbation/debugging'
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(os.path.join(debugging_dir,f'valid_data_{timestamp}.pkl'), "wb") as f:
            pickle.dump(x, f)
        with open(os.path.join(debugging_dir,f'pred_data_{timestamp}.pkl'), "wb") as f:
            pickle.dump(y, f)
        return
    
def reconstruct_data_fn(x, pcs, means):
    return x @ np.transpose(pcs) + np.transpose(means)

def calc_zscores(sample1, sample2):
    mean1 = sample1.mean(axis=0)
    std1 = sample1.std(axis=0)
    mean2 = sample2.mean(axis=0)
    std2 = sample2.std(axis=0)
    n1 = sample1.shape[0]
    n2 = sample2.shape[0]
    se = np.sqrt((std1**2 / n1) + (std2**2 / n2))
    z_scores = (mean1 - mean2) / se
    return z_scores