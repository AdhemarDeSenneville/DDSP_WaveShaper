# from https://github.com/adobe-research/DeepAFx-ST

import math
import torch
from typing import List

from Processor import Processor

def butter(fc, fs: float = 2.0):
    """

    Recall Butterworth polynomials
    N = 1   s + 1
    N = 2   s^2 + sqrt(2s) + 1
    N = 3   (s^2 + s + 1)(s + 1)
    N = 4   (s^2 + 0.76536s + 1)(s^2 + 1.84776s + 1)

    Scaling
    LP to LP:   s -> s/w_c
    LP to HP:   s -> w_c/s

    Bilinear transform:
    s = 2/T_d * (1 - z^-1)/(1 + z^-1)

    For 1-pole butterworth lowpass

    1 / (s + 1)     1-pole prototype
    1 / (s/w_c + 1)  LP to LP
    1 / (2/T_d * (1 - z^-1)/(1 + z^-1))/w_c + 1)  Bilinear transform

    """

    # apply pre-warping to the cutoff
    T_d = 1 / fs
    w_d = (2 * math.pi * fc) / fs
    #    sys.exit()
    w_c = (2 / T_d) * torch.tan(w_d / 2)

    a0 = 2 + (T_d * w_c)
    a1 = (T_d * w_c) - 2
    b0 = T_d * w_c
    b1 = T_d * w_c

    b = torch.stack([b0, b1], dim=0).view(-1)
    a = torch.stack([a0, a1], dim=0).view(-1)

    # normalize
    b = b.type_as(fc) / a0
    a = a.type_as(fc) / a0

    return b, a


def biqaud(
    gain_dB: torch.Tensor,
    cutoff_freq: torch.Tensor,
    q_factor: torch.Tensor,
    sample_rate: float,
    filter_type: str = "peaking",
):

    # convert inputs to Tensors if needed
    # gain_dB = torch.tensor([gain_dB])
    # cutoff_freq = torch.tensor([cutoff_freq])
    # q_factor = torch.tensor([q_factor])

    A = 10 ** (gain_dB / 40.0)
    w0 = 2 * math.pi * (cutoff_freq / sample_rate)
    alpha = torch.sin(w0) / (2 * q_factor)
    cos_w0 = torch.cos(w0)
    sqrt_A = torch.sqrt(A)

    if filter_type == "high_shelf":
        b0 = A * ((A + 1) + (A - 1) * cos_w0 + 2 * sqrt_A * alpha)
        b1 = -2 * A * ((A - 1) + (A + 1) * cos_w0)
        b2 = A * ((A + 1) + (A - 1) * cos_w0 - 2 * sqrt_A * alpha)
        a0 = (A + 1) - (A - 1) * cos_w0 + 2 * sqrt_A * alpha
        a1 = 2 * ((A - 1) - (A + 1) * cos_w0)
        a2 = (A + 1) - (A - 1) * cos_w0 - 2 * sqrt_A * alpha
    elif filter_type == "low_shelf":
        b0 = A * ((A + 1) - (A - 1) * cos_w0 + 2 * sqrt_A * alpha)
        b1 = 2 * A * ((A - 1) - (A + 1) * cos_w0)
        b2 = A * ((A + 1) - (A - 1) * cos_w0 - 2 * sqrt_A * alpha)
        a0 = (A + 1) + (A - 1) * cos_w0 + 2 * sqrt_A * alpha
        a1 = -2 * ((A - 1) + (A + 1) * cos_w0)
        a2 = (A + 1) + (A - 1) * cos_w0 - 2 * sqrt_A * alpha
    elif filter_type == "peaking":
        b0 = 1 + alpha * A
        b1 = -2 * cos_w0
        b2 = 1 - alpha * A
        a0 = 1 + (alpha / A)
        a1 = -2 * cos_w0
        a2 = 1 - (alpha / A)
    else:
        raise ValueError(f"Invalid filter_type: {filter_type}.")

    b = torch.stack([b0, b1, b2], dim=0).view(-1)
    a = torch.stack([a0, a1, a2], dim=0).view(-1)

    # normalize
    b = b.type_as(gain_dB) / a0
    a = a.type_as(gain_dB) / a0

    return b, a


def freqz(b, a, n_fft: int = 512):

    B = torch.fft.rfft(b, n_fft)
    A = torch.fft.rfft(a, n_fft)

    H = B / A

    return H


def freq_domain_filter(x, H, n_fft):

    X = torch.fft.rfft(x, n_fft)

    # move H to same device as input x
    H = H.type_as(X)

    Y = X * H

    y = torch.fft.irfft(Y, n_fft)

    return y


def approx_iir_filter(b, a, x):
    """Approimxate the application of an IIR filter.

    Args:
        b (Tensor): The numerator coefficients.

    """

    # round up to nearest power of 2 for FFT
    # n_fft = 2 ** math.ceil(math.log2(x.shape[-1] + x.shape[-1] - 1))

    n_fft = 2 ** torch.ceil(torch.log2(torch.tensor(x.shape[-1] + x.shape[-1] - 1)))
    n_fft = n_fft.int()

    # move coefficients to same device as x
    b = b.type_as(x).view(-1)
    a = a.type_as(x).view(-1)

    # compute complex response
    H = freqz(b, a, n_fft=n_fft).view(-1)

    # apply filter
    y = freq_domain_filter(x, H, n_fft)

    # crop
    y = y[: x.shape[-1]]

    return y


def approx_iir_filter_cascade(
    b_s: List[torch.Tensor],
    a_s: List[torch.Tensor],
    x: torch.Tensor,
):
    """Apply a cascade of IIR filters.

    Args:
        b (list[Tensor]): List of tensors of shape (3)
        a (list[Tensor]): List of tensors of (3)
        x (torch.Tensor): 1d Tensor.
    """

    if len(b_s) != len(a_s):
        raise RuntimeError(
            f"Must have same number of coefficients. Got b: {len(b_s)} and a: {len(a_s)}."
        )

    # round up to nearest power of 2 for FFT
    # n_fft = 2 ** math.ceil(math.log2(x.shape[-1] + x.shape[-1] - 1))
    n_fft = 2 ** torch.ceil(torch.log2(torch.tensor(x.shape[-1] + x.shape[-1] - 1)))
    n_fft = n_fft.int()

    # this could be done in parallel
    b = torch.stack(b_s, dim=0).type_as(x)
    a = torch.stack(a_s, dim=0).type_as(x)

    H = freqz(b, a, n_fft=n_fft)
    H = torch.prod(H, dim=0).view(-1)

    # apply filter
    y = freq_domain_filter(x, H, n_fft)

    # crop
    y = y[: x.shape[-1]]

    return y



@torch.jit.script
def parametric_eq(
    x: torch.Tensor,
    sample_rate: float,
    low_shelf_gain_dB: torch.Tensor,
    low_shelf_cutoff_freq: torch.Tensor,
    low_shelf_q_factor: torch.Tensor,
    first_band_gain_dB: torch.Tensor,
    first_band_cutoff_freq: torch.Tensor,
    first_band_q_factor: torch.Tensor,
    second_band_gain_dB: torch.Tensor,
    second_band_cutoff_freq: torch.Tensor,
    second_band_q_factor: torch.Tensor,
    third_band_gain_dB: torch.Tensor,
    third_band_cutoff_freq: torch.Tensor,
    third_band_q_factor: torch.Tensor,
    fourth_band_gain_dB: torch.Tensor,
    fourth_band_cutoff_freq: torch.Tensor,
    fourth_band_q_factor: torch.Tensor,
    high_shelf_gain_dB: torch.Tensor,
    high_shelf_cutoff_freq: torch.Tensor,
    high_shelf_q_factor: torch.Tensor,
):
    """Six-band parametric EQ.

    Low-shelf -> Band 1 -> Band 2 -> Band 3 -> Band 4 -> High-shelf

    Args:
        x (torch.Tensor): 1d signal.


    """
    a_s, b_s = [], []
    #print(f"autodiff peq fs = {sample_rate}")

    # -------- apply low-shelf filter --------
    b, a = biqaud(
        low_shelf_gain_dB,
        low_shelf_cutoff_freq,
        low_shelf_q_factor,
        sample_rate,
        "low_shelf",
    )
    b_s.append(b)
    a_s.append(a)

    # -------- apply first-band peaking filter --------
    b, a = biqaud(
        first_band_gain_dB,
        first_band_cutoff_freq,
        first_band_q_factor,
        sample_rate,
        "peaking",
    )
    b_s.append(b)
    a_s.append(a)

    # -------- apply second-band peaking filter --------
    b, a = biqaud(
        second_band_gain_dB,
        second_band_cutoff_freq,
        second_band_q_factor,
        sample_rate,
        "peaking",
    )
    b_s.append(b)
    a_s.append(a)

    # -------- apply third-band peaking filter --------
    b, a = biqaud(
        third_band_gain_dB,
        third_band_cutoff_freq,
        third_band_q_factor,
        sample_rate,
        "peaking",
    )
    b_s.append(b)
    a_s.append(a)

    # -------- apply fourth-band peaking filter --------
    b, a = biqaud(
        fourth_band_gain_dB,
        fourth_band_cutoff_freq,
        fourth_band_q_factor,
        sample_rate,
        "peaking",
    )
    b_s.append(b)
    a_s.append(a)

    # -------- apply high-shelf filter --------
    b, a = biqaud(
        high_shelf_gain_dB,
        high_shelf_cutoff_freq,
        high_shelf_q_factor,
        sample_rate,
        "high_shelf",
    )
    b_s.append(b)
    a_s.append(a)
    
    x = approx_iir_filter_cascade(b_s, a_s, x.view(-1))
        
    return x


class ParametricEQ(Processor):
    def __init__(
        self,
        sample_rate,
        min_gain_dB=-24.0,
        default_gain_dB=0.0,
        max_gain_dB=24.0,
        min_q_factor=0.1,
        default_q_factor=0.707,
        max_q_factor=10,
        eps=1e-8,
    ):
        """ """
        super().__init__()
        self.sample_rate = sample_rate
        self.eps = eps
        self.ports = [
            {
                "name": "Lowshelf gain",
                "min": min_gain_dB,
                "max": max_gain_dB,
                "default": default_gain_dB,
                "units": "dB",
            },
            {
                "name": "Lowshelf cutoff",
                "min": 20.0,
                "max": 200.0,
                "default": 100.0,
                "units": "Hz",
            },
            {
                "name": "Lowshelf Q",
                "min": min_q_factor,
                "max": max_q_factor,
                "default": default_q_factor,
                "units": "",
            },
            {
                "name": "First band gain",
                "min": min_gain_dB,
                "max": max_gain_dB,
                "default": default_gain_dB,
                "units": "dB",
            },
            {
                "name": "First band cutoff",
                "min": 200.0,
                "max": 2000.0,
                "default": 400.0,
                "units": "Hz",
            },
            {
                "name": "First band Q",
                "min": min_q_factor,
                "max": max_q_factor,
                "default": 0.707,
                "units": "",
            },
            {
                "name": "Second band gain",
                "min": min_gain_dB,
                "max": max_gain_dB,
                "default": default_gain_dB,
                "units": "dB",
            },
            {
                "name": "Second band cutoff",
                "min": 200.0,
                "max": 4000.0,
                "default": 1000.0,
                "units": "Hz",
            },
            {
                "name": "Second band Q",
                "min": min_q_factor,
                "max": max_q_factor,
                "default": default_q_factor,
                "units": "",
            },
            {
                "name": "Third band gain",
                "min": min_gain_dB,
                "max": max_gain_dB,
                "default": default_gain_dB,
                "units": "dB",
            },
            {
                "name": "Third band cutoff",
                "min": 2000.0,
                "max": 8000.0,
                "default": 4000.0,
                "units": "Hz",
            },
            {
                "name": "Third band Q",
                "min": min_q_factor,
                "max": max_q_factor,
                "default": default_q_factor,
                "units": "",
            },
            {
                "name": "Fourth band gain",
                "min": min_gain_dB,
                "max": max_gain_dB,
                "default": default_gain_dB,
                "units": "dB",
            },
            {
                "name": "Fourth band cutoff",
                "min": 4000.0,
                "max": (24000 // 2) * 0.9,
                "default": 8000.0,
                "units": "Hz",
            },
            {
                "name": "Fourth band Q",
                "min": min_q_factor,
                "max": max_q_factor,
                "default": default_q_factor,
                "units": "",
            },
            {
                "name": "Highshelf gain",
                "min": min_gain_dB,
                "max": max_gain_dB,
                "default": default_gain_dB,
                "units": "dB",
            },
            {
                "name": "Highshelf cutoff",
                "min": 4000.0,
                "max": (24000 // 2) * 0.9,
                "default": 8000.0,
                "units": "Hz",
            },
            {
                "name": "Highshelf Q",
                "min": min_q_factor,
                "max": max_q_factor,
                "default": default_q_factor,
                "units": "",
            },
        ]

        self.num_control_params = len(self.ports)

    def forward(self, x, p, sample_rate=24000, **kwargs):

        bs, chs, s = x.size()

        inputs = torch.split(x, 1, 0)
        params = torch.split(torch.sigmoid(p), 1, 0)

        y = []  # loop over batch dimension
        for input, param in zip(inputs, params):
            denorm_param = self.denormalize_params(param.view(-1))
            y.append(parametric_eq(input.view(-1), sample_rate, *denorm_param))

        return torch.stack(y, dim=0).view(bs, 1, -1)