RATIOS = [0.001, 0.005]

SLIM_LR = [1e-2, 5e-3, 1e-3]
SLIM_REG = [30000, 10000, 5000, 1000]

HC_LR = [1e-2, 5e-3, 1e-3]
HC_REG = [2000, 1000, 500]
HC_P = [0.9, 0.5, 0.2]
HC_BETA = [0.67, 0.33]

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
