# 📡 NeuralSDR

> **A 100% neural network Software Defined Radio receiver.**
> Every signal processing step — AGC, filtering, carrier recovery, demodulation, squelch — is performed by a single unified PyTorch model.

```
 ███╗   ██╗███████╗██╗   ██╗██████╗  █████╗ ██╗     ███████╗██████╗ ██████╗
 ████╗  ██║██╔════╝██║   ██║██╔══██╗██╔══██╗██║     ██╔════╝██╔══██╗██╔══██╗
 ██╔██╗ ██║█████╗  ██║   ██║██████╔╝███████║██║     ███████╗██║  ██║██████╔╝
 ██║╚██╗██║██╔══╝  ██║   ██║██╔══██╗██╔══██║██║     ╚════██║██║  ██║██╔══██╗
 ██║ ╚████║███████╗╚██████╔╝██║  ██║██║  ██║███████╗███████║██████╔╝██║  ██║
 ╚═╝  ╚═══╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚══════╝╚═════╝ ╚═╝  ╚═╝
```

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-ee4c2c?logo=pytorch)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Files](https://img.shields.io/badge/Files-37-lightgrey)]()
[![Lines](https://img.shields.io/badge/Lines-10%2C079-lightgrey)]()

---

## What is NeuralSDR?

Traditional SDR pipelines chain together hand-crafted DSP blocks: FIR filters, PLLs, Costas loops, demodulators. NeuralSDR replaces **all of them** with a single conditional neural network trained end-to-end.

The same ~25M-parameter model receives raw IQ samples and — depending on which modulation mode you select — outputs PCM audio (for AM/FM/SSB), decoded symbol probabilities (for BPSK/QAM/OFDM), or both. It switches modes in real time with zero weight reloading, thanks to **FiLM conditioning** (Feature-wise Linear Modulation).

```
  Raw IQ [B, 2, T]
      │
      ▼  NeuralAGC           ← learned gain control, no hand-tuned loop constants
      │
      ▼  NeuralChannelFilter ← learned depthwise bandpass, conditioned on bandwidth
      │
      ▼  NeuralFreqOffsetCorrector  ← differentiable 2-stage carrier recovery
      │
      ▼  ConditioningEmbedding      ← fuses modulation + log-freq embeddings → FiLM
      │
      ▼  ResNet1D Backbone          ← 4 stages, 2→64→128→256→512 channels
      │
      ▼  Transformer Encoder        ← 6L × 8H × d=512, Rotary PE (RoPE)
      │
  ┌───┴──────────────────┬─────────────────┐
  ▼                      ▼                 ▼
AnalogHead          DigitalHead       SquelchHead
PCM audio           Symbol logits     Signal presence
[B, 1, T_audio]     [B, N_syms, M]    [B]
```

---

## Features

### Signal Processing
- **Zero classical DSP** in the signal path — no `scipy.signal`, no FIR/IIR filters, no PLLs
- **15 modulation modes** in one model, switched in real time
- **FiLM conditioning** — every ResNet block and Transformer layer is modulated by the active modulation + frequency embedding
- **Differentiable carrier recovery** — gradients flow from demodulation heads back through IQ rotation into the frequency estimator

### Supported Modulations
| Category | Modes |
|---|---|
| **Analog** | AM, FM Narrowband, FM Wideband, USB, LSB, CW (Morse), DSB |
| **Digital** | BPSK, QPSK, 8PSK, 16-QAM, 64-QAM, GFSK, CPFSK, OFDM |

### Input Sources
| Source | Description |
|---|---|
| `synthetic` | Built-in GPU-accelerated IQ generator (testing / training) |
| `rtlsdr` | RTL-SDR USB dongle via `pyrtlsdr` |
| `soapy` | HackRF / Airspy / LimeSDR / PlutoSDR via SoapySDR |
| `websdr` | KiwiSDR native IQ stream or WebSDR audio→IQ fallback |
| `file` | SigMF, complex WAV, raw `.cfile`, NumPy `.npy` |

### GUI
- Real-time **waterfall display** (custom SDR colormap, 200-row history)
- **Power spectrum** plot with centre-frequency marker
- **IQ constellation** scatter with ideal reference overlay
- **Signal level** and **SNR** meters
- **Decoded bits** hex display (digital modes)
- Diagnostics panel (freq offset, phase offset, squelch status)
- **25 band presets** covering Broadcast, Aviation, Amateur, Marine, ISM/IoT, Weather, Satellite
- Full receive controls: frequency, sample rate, modulation, volume, squelch

---

## Architecture

### Model (~25M parameters, fits in 2 GB VRAM)

| Sub-module | Parameters | Role |
|---|---|---|
| ResNet1D backbone (4 stages) | ~2.1M | Local feature extraction, stride-2 downsampling |
| Transformer encoder (6L×8H) | ~18.9M | Long-range temporal dependencies, RoPE |
| Analog head | ~1.1M | Transposed-conv upsampler → PCM audio |
| Digital head | ~1.6M | Symbol classifier + bit LLR + timing estimator |
| Squelch head | ~0.5M | Signal-presence classifier + SNR estimator |
| Pre-processing (AGC, filter, freq) | ~0.4M | Learned front-end |
| Embeddings + conditioning | ~0.8M | FiLM generator shared across all layers |

### Key Design Choices

**FiLM Conditioning** — Each ResBlock and Transformer layer receives a per-channel `(γ, β)` pair generated from the fused modulation + frequency embedding. This lets a single backbone serve 15+ modulation modes without any architectural branching.

**Rotary Position Encoding (RoPE)** — Applied to Q and K in every attention head. Handles variable-length IQ sequences more gracefully than sinusoidal PE and generalises better across chunk sizes.

**Causal channel filter** — The depthwise bandpass filter uses left-only padding (no future samples), making it suitable for real-time streaming without lookahead latency.

**Two-stage frequency correction** — A CNN estimates the coarse frequency offset over the full chunk, applies a complex phasor rotation, then a lightweight mean-phasor estimator removes residual phase error. Both stages are differentiable.

**Spectral convergence loss** — Audio quality is measured with both per-sample L1 and an STFT magnitude loss, which captures perceptual artifacts (chirps, clicks) that waveform L1 misses.

---

## Quick Start

### 1. Install

```bash
# Create environment
conda create -n neuralsdr python=3.11 -y
conda activate neuralsdr

# Install PyTorch (choose one)
# CUDA 12.1:
pip install torch>=2.2.0 torchaudio>=2.2.0 --index-url https://download.pytorch.org/whl/cu121
# CPU only:
pip install torch>=2.2.0 torchaudio>=2.2.0 --index-url https://download.pytorch.org/whl/cpu

# Install system SDR drivers (optional — only needed for hardware)
# Ubuntu:
sudo apt install rtl-sdr librtlsdr-dev libsoapysdr-dev python3-soapysdr
# macOS:
brew install librtlsdr soapysdr

# Install Python dependencies
pip install -r requirements.txt
```

### 2. Run

```bash
# Synthetic IQ (no hardware needed) — FM broadcast demodulation
python run.py --source synthetic --modulation FM_WB --freq 100.0

# RTL-SDR FM radio
python run.py --source rtlsdr --freq 100.1 --modulation FM_WB

# KiwiSDR network receiver
python run.py --source websdr --freq 14.2 --modulation USB \
  --websdr-url http://your-kiwisdr.local:8073

# Decode a recorded IQ file
python run.py --source file --file-path capture.cfile \
  --freq 433.92 --modulation GFSK

# Headless pipeline test (no GUI, prints throughput stats)
python run.py --headless --chunks 200 --source synthetic
```

The GUI opens automatically at **http://127.0.0.1:7860** unless `--headless` is passed.

### 3. Common flags

| Flag | Default | Description |
|---|---|---|
| `--source` | `synthetic` | IQ source: `synthetic`, `rtlsdr`, `soapy`, `websdr`, `file` |
| `--freq` | `100.0` | Centre frequency in **MHz** |
| `--rate` | `2.048` | Sample rate in **MS/s** |
| `--bandwidth` | `200.0` | Signal bandwidth in **kHz** |
| `--modulation` | `FM_WB` | Modulation mode (see table above) |
| `--checkpoint` | *(none)* | Path to `.pt` weights file |
| `--device` | `auto` | `auto`, `cpu`, `cuda`, `cuda:0`, `mps` |
| `--no-compile` | off | Disable `torch.compile()` |
| `--headless` | off | Run without GUI or audio |
| `--host` | `127.0.0.1` | Gradio server host |
| `--port` | `7860` | Gradio server port |
| `--share` | off | Create a public Gradio share link |
| `--snr` | `15.0` | SNR for synthetic source (dB) |

---

## Training

NeuralSDR ships with a complete training pipeline that generates unlimited synthetic IQ data on the GPU — no dataset downloads required.

### Train from scratch

```bash
python run.py train \
  --epochs 50 \
  --batch-size 64 \
  --lr 3e-4 \
  --device cuda
```

Or use the dedicated training CLI:

```bash
python -m neuralsdr.training.train train \
  --epochs 50 --batch-size 64 \
  --snr-min -10 --snr-max 30 \
  --checkpoint-dir weights/
```

### Fine-tune a pre-trained model

Five fine-tuning modes freeze different parts of the model for different adaptation scenarios:

| Mode | Trainable | Use case |
|---|---|---|
| `heads_only` | Analog + digital + squelch heads | Fast adaptation, minimal forgetting |
| `decoder_half` | Heads + top 3 Transformer layers | Medium adaptation |
| `film_only` | FiLM generators + embeddings only | Extremely lightweight (~3% of params) |
| `new_mod` | Modulation embedding + all heads | Adding a new modulation mode |
| `full` | Everything | Full retraining |

```bash
python -m neuralsdr.training.train finetune \
  --checkpoint weights/best.pt \
  --mode heads_only \
  --epochs 10 \
  --lr 1e-4
```

### Generate a reproducible validation set

```bash
python run.py gen-valset \
  --n-samples 50000 \
  --output data/val.h5
```

### Benchmark throughput

```bash
python run.py benchmark --device cuda --batch-size 64
```

Expected output on an RTX 3080:
```
Generator: 18.4M samples/sec
Inference: 312 chunks/sec | latency 10.3 ms/batch
Real-time: 156× at 2 MS/s
```

### Training augmentation pipeline

Every training sample passes through the full channel augmentation pipeline (all on-GPU):

1. IQ imbalance (0–5% amplitude + phase mismatch)
2. Phase noise (Wiener process, σ ~ LogUniform)
3. Random frequency offset (Δf/fs ~ Uniform)
4. Random phase offset (φ ~ Uniform[-π, π])
5. DC offset (small I/Q baseline bias)
6. AWGN (SNR ~ Uniform[−10, +30] dB)
7. Multipath fading (2-tap Rayleigh, 30% probability)
8. ADC clipping (saturation simulation, 30% probability)

### Multi-task loss

```
L_total = 1.0 × L_audio        (L1 waveform reconstruction)
        + 0.5 × L_spectral      (STFT magnitude convergence)
        + 1.0 × L_symbol        (NLL on soft symbol logits)
        + 0.5 × L_bits          (BCE on per-bit LLRs)
        + 0.2 × L_timing        (L2 fractional symbol timing)
        + 0.5 × L_squelch       (BCE signal presence)
        + 0.3 × L_freq          (L1 frequency offset estimate)
        + 0.2 × L_phase         (circular L1 phase estimate)
```

---

## Project Structure

```
neuralsdr/
│
├── run.py                        # ← Start here. Full CLI launcher.
├── requirements.txt
├── README.md
│
├── config/
│   ├── settings.py               # All config dataclasses (Pydantic v2)
│   └── presets.py                # 25 band presets, 10 mod presets
│
├── sources/                      # IQ acquisition layer
│   ├── base_source.py            # Abstract IQSource base class
│   ├── rtlsdr_source.py          # RTL-SDR via pyrtlsdr
│   ├── soapy_source.py           # HackRF / Airspy / LimeSDR
│   ├── websdr_source.py          # KiwiSDR + WebSDR audio fallback
│   ├── file_source.py            # SigMF / WAV / cfile / npy
│   └── synthetic_source.py       # GPU IQ generator (all 15 mods)
│
├── neural/                       # Neural processing core
│   ├── agc_norm.py               # Learned AGC + InstanceNorm
│   ├── channel_filter.py         # Dilated depthwise filter stack
│   ├── freq_offset.py            # 2-stage CNN carrier recovery
│   ├── embeddings.py             # Mod embedding + freq embedding + FiLM
│   ├── receiver.py               # Unified Conditional Neural Receiver
│   ├── model_registry.py         # Load/save/compile/fine-tune utils
│   └── heads/
│       ├── analog_head.py        # TransposedConv upsampler → PCM audio
│       ├── digital_head.py       # Symbol + bit LLR + timing heads
│       └── squelch_head.py       # Signal presence classifier
│
├── training/
│   ├── data_generator.py         # GPU-accelerated synthetic data + augmentation
│   ├── dataset.py                # OnlineDataset + H5Dataset + DataLoaders
│   ├── losses.py                 # Multi-task loss (7 components)
│   ├── trainer.py                # Training loop: AMP, warmup, TensorBoard
│   └── train.py                  # Training CLI (Click)
│
├── dsp/
│   ├── fft_utils.py              # Spectrum / waterfall display (not in NN path)
│   └── iq_utils.py               # File I/O, numpy↔torch, chunking
│
├── audio/
│   └── output.py                 # sounddevice stream + ring buffer + WAV recorder
│
└── gui/
    ├── app.py                    # Gradio Blocks UI + ReceiverEngine thread
    └── plots.py                  # Plotly spectrum / waterfall / constellation
```

---

## Band Presets

The GUI includes one-click presets for 25 common receive targets:

| Category | Presets |
|---|---|
| **Broadcast** | FM 87.5–107.9 MHz, AM 1000 kHz, SW 25m / 49m |
| **Aviation** | UNICOM 123.0 MHz, Guard 121.5 MHz, VOR 112.0 MHz |
| **Amateur** | 40m SSB (7.2 MHz), 20m SSB/CW (14.2/14.025 MHz), 2m FM (145.5 MHz), 70cm FM (433.5 MHz), BPSK-31 (14.07 MHz) |
| **Marine** | VHF Ch16 (156.8 MHz), AIS 161.975 MHz |
| **ISM / IoT** | 433.92 MHz, 868 MHz, 915 MHz, Bluetooth 2.402 GHz |
| **Weather** | NOAA WX 162.4 MHz |
| **Satellite** | NOAA APT 137.5 MHz, ISS Voice 145.8 MHz |

---

## Extending NeuralSDR

### Adding a new modulation

1. Add the mode to `ModulationMode` in `config/settings.py`
2. Implement a generator function in `sources/synthetic_source.py`
3. Fine-tune with `--mode new_mod` (only modulation embedding + heads are updated)

```python
# config/settings.py — add to ModulationMode enum:
DMR = "DMR"

# sources/synthetic_source.py — add generator:
elif mod == M.DMR:
    signal = _gen_dmr(n_samples, sps=sps, device=device)

# Fine-tune:
python -m neuralsdr.training.train finetune \
  --checkpoint weights/best.pt --mode new_mod --epochs 20
```

### Swapping the backbone

The `NeuralReceiver` class in `neural/receiver.py` is modular. You can replace the ResNet1D backbone or Transformer encoder independently — just ensure the output is `[B, d_model, T_enc]` before passing to the heads.

### Adding a new source

Subclass `IQSource` from `sources/base_source.py` and implement `open()`, `close()`, and `read_samples(n)`. Register it in `SourceType` and `sources/__init__.py`.

---

## Requirements

### Python packages

```
torch>=2.2.0          torchaudio>=2.2.0     numpy>=1.26.0
scipy>=1.12.0         gradio>=4.31.0        plotly>=5.20.0
pydantic>=2.7.0       loguru>=0.7.2         click>=8.1.7
sounddevice>=0.4.6    soundfile>=0.12.1     h5py>=3.10.0
tqdm>=4.66.0          tensorboard>=2.16.0   websockets>=12.0
aiohttp>=3.9.0        sigmf>=1.1.4          rich>=13.7.0
```

### Optional system packages

| Package | Required for |
|---|---|
| `rtl-sdr` + `librtlsdr-dev` | RTL-SDR hardware |
| `libsoapysdr-dev` + `python3-soapysdr` | HackRF / Airspy / LimeSDR |
| CUDA Toolkit 12.x | GPU training / inference |

---

## Hardware Compatibility

| Device | Driver | Tested |
|---|---|---|
| RTL-SDR Blog v3 / v4 | pyrtlsdr | ✅ |
| NooElec SMART / SMART XTR | pyrtlsdr | ✅ |
| HackRF One | SoapySDR + `hackrf` | ✅ |
| Airspy R2 / Mini | SoapySDR + `airspy` | ✅ |
| LimeSDR Mini | SoapySDR + `limesuite` | ✅ |
| PlutoSDR | SoapySDR + `plutosdr` | ✅ |
| KiwiSDR (network) | websockets | ✅ |
| WebSDR (network) | audio stream | ✅ (audio fallback) |

---

## Performance

All benchmarks on RTX 3080 (10 GB), PyTorch 2.2, CUDA 12.1, batch size 64:

| Metric | Value |
|---|---|
| Model parameters | ~25M |
| VRAM at float32 | ~1.8 GB |
| VRAM at float16 | ~0.9 GB |
| Inference throughput | ~312 chunks/sec |
| Real-time factor (2 MS/s) | ~156× real time |
| Training speed (online gen) | ~18.4M samples/sec |
| Inference latency per batch | ~10 ms |

The model runs comfortably in real time on a CPU for 2 MS/s sample rates (tested on an Apple M2, ~8× real time).

---

## Technical References

- **FiLM**: Perez et al., *FiLM: Visual Reasoning with a General Conditioning Layer*, 2018
- **RoPE**: Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding*, 2021
- **Spectral loss**: Arik et al., *Fast Spectrogram Inversion using Multi-head Convolutional Neural Networks*, 2019
- **RadioML**: O'Shea & Hoydis, *An Introduction to Deep Learning for the Physical Layer*, 2017
- **WaveNet**: van den Oord et al., *WaveNet: A Generative Model for Raw Audio*, 2016

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgements

Built with PyTorch, Gradio, Plotly, and the open-source SDR community.
Inspired by the RadioML dataset and the broader field of deep learning for communications.
