Hi Stefan





# Authors : 
- SENNEVILLE Adhemar (MVA)

# Work overview
As part of the class of G. RICHARD and R. BADEAU, I studied the paper **[Style Transfer of Audio Effects with
Differentiable Signal Processing](https://arxiv.org/abs/2207.08759)** from Adobe Research and Queen Mary University.

The main contibrution of that repository is the adaptation of the WaveShaper Pluging from FL-Studio in a Pytorch differenciable version with high expressivity.

## Wave Shaper
![avering](https://github.com/b-ptiste/dtw-soft/assets/75781257/b1373a3a-f1b7-4ea3-8701-912d511f7c72)

# Simple utilisation 

Our code is compatible with any native **Pytorch** implementation. We over-write the backward for efficiency purposes.

```python
import torch
from tslearn.datasets import UCR_UEA_datasets
from DTWLoss_CUDA import DTWLoss

# load data
ucr = UCR_UEA_datasets()
X_train, y_train, X_test, y_test = ucr.load_dataset("SonyAIBORobotSurface2")
from DTWLoss_CUDA import DTWLoss

# convert to torch
X_train = torch.from_numpy(X_train).float().requires_grad_(True)
loss = DTWLoss(gamma=0.1)
optimizer = # your optimizer

##############
# your code ##
##############

value = loss(X_train[0].unsqueeze(0), X_train[1].unsqueeze(0))
optimizer.zero_grad()
value.backward()
optimizer.step()
```

# Experiments

## Training on using Waveshaper
![avering](https://github.com/b-ptiste/dtw-soft/assets/75781257/b1373a3a-f1b7-4ea3-8701-912d511f7c72)

## Training on PEQ
![Capture d'écran 2024-01-09 114025](https://github.com/b-ptiste/dtw-soft/assets/75781257/02cdacde-e02b-42f1-afaa-8954730e1fe9)

## Training on PEQ and Waveshaper
![Capture d'écran 2024-01-09 114258](https://github.com/b-ptiste/dtw-soft/assets/75781257/e1c1702a-8952-4fc7-a2e1-af74c60e94de)

# Credit

[Style Transfer of Audio Effects with Differentiable Signal Processing by Christian J. Steinmetz and Nicholas J. Bryan and Joshua D. Reiss, 2022](https://arxiv.org/abs/2207.08759)
