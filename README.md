# Authors : 
- SENNEVILLE Adhemar (MVA)

# Work overview
As part of the class of G. RICHARD and R. BADEAU, I studied the paper **[Style Transfer of Audio Effects with
Differentiable Signal Processing](https://arxiv.org/abs/2207.08759)** from Adobe Research and Queen Mary University.

The main contibrution of that repository is the adaptation of the WaveShaper Pluging from FL-Studio in a Pytorch differenciable version with high expressivity. This plugin could be very useful in the style tranfere a certain gendra relying extensivly on saturation and distortion of sounds.

## Wave Shaper
The WaveShaper is a plugin that apply a function $f: [-1,1] \rightarrow [-1,1]$ to all sample of an audio signal. This function is usuly shaped by the utilisator using besier curves or all sortes of interpolations.
![avering](https://github.com/AdhemarDeSenneville/DDSP_WaveShaper/blob/main/fig/waveshaper.jpg?raw=true)


I also included all the images used for the generation of the WaveShaper dataset.

![avering](https://raw.githubusercontent.com/AdhemarDeSenneville/DDSP_WaveShaper/main/fig/WaveShaper_dataset.png)

To my knoledge, the WaveShaper, wile beeing simple, has never been model as a DDSP, this is du to it expresivity that is in theory infinit (one parameter for each value between 0 and 1). 
This are 4 examples of possible utilisation of the WaveShapes in a production Pipeline

- **Gain**: $f(x) = ax$
- **Bit Cruncher**: $f(x) = \left\lfloor \frac{|ax|}{a} \right\rfloor$
- **Saturation**: $f(x) = \max(x, a)$
- **Overdrive**: $f(x) = \tanh(x)$


I tried implementing the waveshaper as a MLP, whever seeing the poor results I opted for 
Using parametric interpolation on 2D points, it was possible to create a Differentiable WaveShaper with high expressivity 

- $MLP$ - $f$ is modeled as a Multi Layer Perceptron
- $LIX_n$ - 
- $DIX_n$ - 
- $DIYX_n$ - 

![avering](https://raw.githubusercontent.com/AdhemarDeSenneville/DDSP_WaveShaper/main/fig/results_3.png)

![avering](https://raw.githubusercontent.com/AdhemarDeSenneville/DDSP_WaveShaper/main/fig/results_1.png)

![avering](https://raw.githubusercontent.com/AdhemarDeSenneville/DDSP_WaveShaper/main/fig/results_2.png)

# Simple experiment

The code is made using **Pytorch**. Here is the example of a differentiable processing pipeline using (in serie) a Parametric Equilizer From the Transfere style paper and The WaveShaper.

![avering](https://raw.githubusercontent.com/AdhemarDeSenneville/DDSP_WaveShaper/main/fig/Training_Architecture_2.png)

```python
import torch
from WaveShaper import WaveShaper
from PEQ import ParametricEQ
# load a mp3 file into pytorch tenso format
input_audio = 
output_audio = 

waveshaper = WaveShaper(3) # Use 3 points for the interpolation
peq = ParametricEQ(sample_rate = 44100)

# Train
for _ in range(100):
    output_audio_estimated = waveshaper(peq(input_audio))

    loss = torch.nn.MSELoss()(output_audio_estimated,output_audio)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Show Estimated Parametres

# for the WaveShaper


```

# Experiments

## Training on using Waveshaper
![avering](https://github.com/b-ptiste/dtw-soft/assets/75781257/b1373a3a-f1b7-4ea3-8701-912d511f7c72)


# Credit

[Style Transfer of Audio Effects with Differentiable Signal Processing by Christian J. Steinmetz and Nicholas J. Bryan and Joshua D. Reiss, 2022](https://arxiv.org/abs/2207.08759)
