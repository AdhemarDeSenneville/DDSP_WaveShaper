# WaveShaper DDSP

[**Code**](./code/Experiments_WaveShaper.ipynb)
| [**Tutorial**](#Tutorial)
| [**Presentation**](DDSP_Presentation.pptx)
| [**Original Papers**](https://arxiv.org/abs/2207.08759)

# Work overview
As part of the class of G. RICHARD and R. BADEAU [Audio signal Analysis, Indexing and Transformations](https://www.master-mva.com/cours/audio-signal-analysis-indexing-and-transformations/), I studied the paper **[Style Transfer of Audio Effects with
Differentiable Signal Processing](https://arxiv.org/abs/2207.08759)** from Adobe Research and Queen Mary University.

The main contibrution of that repository is the adaptation of the WaveShaper Pluging from FL-Studio in a Pytorch differenciable version with high expressivity. This plugin could be very useful in the style tranfere a certain gendra relying extensivly on saturation and distortion of sounds.

## Delivrables

# Wave Shaper
The WaveShaper is a plugin that apply a function $f: [-1,1] \rightarrow [-1,1]$ to all sample of an audio signal. This function is usuly shaped by the utilisator using besier curves or all sortes of interpolations. Here we will consider that f is antisymetric, so we model f only on $[0,1]$
![](./fig/waveshaper.jpg?raw=true)


I also included all the images used for the generation of the *WaveShaper dataset*.

![](./fig/WaveShaper_dataset.png)

To my knoledge, the WaveShaper, wile beeing simple, has never been model as a DDSP, this is du to it expresivity that is in theory infinit (one parameter for each value between -1 and 1). 
This are 4 examples of possible utilisation of the WaveShapes in a production Pipeline

- **Gain**: $f(x) = ax$
- **Bit Cruncher**: $f(x) = \left\lfloor \frac{|ax|}{a} \right\rfloor$
- **Saturation**: $f(x) = \max(x, a)$
- **Overdrive**: $f(x) = \tanh(x)$

## Model Approaches

I tried implementing the WaveShaper as a MLP, whever seeing the poor results I opted for Using parametric interpolation on 2D points, it was possible to create a Differentiable WaveShaper with high expressivity for a low parameter count

### $MLP$ 
$f$ is modeled as a Multi Layer Perceptron wich architecture is a sparce auto encoder with one input and one output. I tried experimenting on having mixed activation function for on layer (Relu and sigmoide) to enhence the expresivity. The parameter efficiency of those models is bad.
![](./fig/results_1_MLP.png)
### $LIX_n$
It has $n$ parameters linearly spaced between 0 and 1
![](./fig/results_1_LIX.png)
### $DIX_n$
It has $2n$ parameters. Compared to $LIX_n$, $n$ more parameters are used to control the interpolation 'rate' of each point
![](./fig/results_1_DIX.png)
### $DIYX_n$
It has $3n$ parameters. Compared to $DIX_n$, $n-2$ more parameters are used to control the x position of each points. For stability reason first and second points are clampt to 0 and 1 respectively.

![](./fig/results_1.png)

![](./fig/results_2.png)

## Expesivity

Here is the Loss VS Parameter plot of those different approches
![](./fig/results_3.png)



# Tutorial

The code is made using **Pytorch**. Here is the example of a differentiable training pipeline using the WaveShaper.

![](./fig/Training_Architecture.png)

```python
import torch
import torchaudio
import matplotlib.pyplot as plt
from WaveShaper import WaveShaper

# Load input and output audio files
input_audio, _ = torchaudio.load("input.mp3")
output_audio, _ = torchaudio.load("output.mp3") # Same length

waveshaper = WaveShaper(3) # WaveShaper with 3 points for interpolation
optimizer = torch.optim.Adam(waveshaper.parameters(), lr=0.01)

# Train the WaveShaper
for _ in range(100):
    output_audio_estimated = waveshaper(input_audio.clone())

    loss = torch.nn.MSELoss()(output_audio_estimated, output_audio)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Plotting the results
fig, axe = plt.subplots()
waveshaper.plot(axe)
axe.set_title("WaveShaper Output"); axe.legend(); plt.show()

# Print the parameters of the WaveShaper
print("Params:", waveshaper.params.data, waveshaper.params_var.data, waveshaper.params_X.data)
```

# Experiment

## [Experiment 1: Training PEQ - IIR model](./code/Experiments_PEQ_IIR.ipynb)

## [Experiment 2: Training PEQ - Paper model (Better)](./code/Experiments_PEQ_Paper.ipynb)

## [Experiment 3: Training WaveShaper](./code/Experiments_WaveShaper.ipynb)

## [Experiment 4: Training WaveShaper and PEQ](./code/Experiments_WaveShaper_PEQ.ipynb)
Here is the example of a differentiable processing pipeline using (in serie) a Parametric Equilizer From the Transfere style paper and The WaveShaper.
![](./fig/Training_Architecture_2.png)

# Authors :
- de SENNEVILLE Adhémar (MVA)

# To do :

- Compatible with DDSP

# Credit

[Style Transfer of Audio Effects with Differentiable Signal Processing by Christian J. Steinmetz and Nicholas J. Bryan and Joshua D. Reiss, 2022](https://arxiv.org/abs/2207.08759)
