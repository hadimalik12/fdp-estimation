import os
import sys

# Navigate to the parent directory of the project structure
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
src_dir = os.path.join(project_dir, 'src')
fig_dir = os.path.join(project_dir, 'fig')
data_dir = os.path.join(project_dir, 'data')
log_dir = os.path.join(project_dir, 'log')
os.makedirs(fig_dir, exist_ok=True)

# Add the src directory to sys.path
if src_dir not in sys.path:
    sys.path.append(src_dir)

import mech.full_DPSGD as DPSGDModule

data_args = {
    "method": "default",
    "data_dir": data_dir,
    "internal_result_path": "/scratch/bell/wei402/fdp-estimation/results"
}

args = DPSGDModule.generate_params(data_args=data_args, log_dir=log_dir, model_type="CNN")
sampler = DPSGDModule.CNN_DPSGDSampler(args)

score0, score1 = sampler.preprocess(num_samples=6)
print(score0)
print(score1)