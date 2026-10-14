# rapido-whisper

rapido-whisper is a Whisper-only inference engine for NVIDIA GPUs, derived from
[CTranslate2](https://github.com/OpenNMT/CTranslate2) 4.8.3 (MIT). It keeps the CTranslate2
Python API for Whisper (`import ctranslate2`), so [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
and [CrisperWhisper](https://github.com/nyrahealth/CrisperWhisper) run on it unchanged.

Its rule is **bit-identical outputs**. Every change keeps tokens, scores, no-speech
probabilities and word timestamps equal to the original CTranslate2 on the same GPU and
cuBLAS version. Speedups that change floating point rounding are opt-in or left out.

## What it contains

- Whisper only: encoder, decoder, `generate`, `align`, `detect_language`, and the
  Transformers Whisper converter (`ct2-transformers-converter`).
- CUDA only, native code for Ada (sm_89: RTX 40xx, L4, L40) and Blackwell (sm_120: RTX 50xx).
- Compute types `float16`, `bfloat16` and `float32`. `auto` resolves to `float16`.
- **CUDA graphs for decoder steps** (`CT2_CUDA_GRAPHS=1`). Each step is captured once and
  replayed in later calls, for example in every window of streaming ASR. The graphs run
  the same kernels as the regular path. Measured on an RTX 5080 with large-v3: `generate`
  is about 20% faster (5.9 → 4.6 ms/token greedy, 719 → 611 ms beam 5 on a 15 s window).
- The CrisperWhisper decoding APIs from
  [nyrahealth/CTranslate2](https://github.com/nyrahealth/CTranslate2): `prefill`,
  `forward_step`, `forward_batch`, `*_with_attention`, `generate_greedy_with_attention`,
  `generate_dual_greedy` and `generate_speculative`.

Removed from CTranslate2: every model except Whisper (translators, generators, encoders,
Wav2Vec2), the CLI, CPU backends (MKL, oneDNN, Ruy, OpenBLAS, Accelerate, OpenMP), AMD
HIP, tensor parallelism, FlashAttention, AWQ, int8/int16 quantization, and LLM-only layer
features (rotary, ALiBi, RMSNorm, GQA/MQA, sliding windows, T5 relative positions, Gemma
norms). A model that needs any of them is rejected at load time with an explicit error.

## Build

Requirements: CUDA 12.8+ (sm_120 needs it), CMake ≥ 3.18 and GCC. The release flags below
match the wheels; `-DWITH_CUDNN=ON` runs the encoder's conv1d through cuDNN instead, which
changes its rounding.

```bash
git submodule update --init --recursive
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DWITH_CUDA=ON -DWITH_CUDNN=OFF -DCUDA_DYNAMIC_LOADING=ON \
  -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.8 \
  "-DCUDA_ARCH_FLAGS_OVERRIDE=-gencode;arch=compute_89,code=sm_89;-gencode;arch=compute_120,code=sm_120;-gencode;arch=compute_120,code=compute_120" \
  -DCMAKE_INSTALL_PREFIX=$PWD/../install
make -j$(nproc) install

cd ../python
CTRANSLATE2_ROOT=$PWD/../install pip install --no-build-isolation .
```

`CUDA_ARCH_FLAGS_OVERRIDE` passes the `-gencode` flags directly, because older CMake
FindCUDA cannot parse two-digit architectures such as 12.0. Set `LD_LIBRARY_PATH` to
`install/lib` at runtime.

## Install

```bash
pip install "CTranslate2-rapid-whisper[cuda]"
```

The wheels are built like the upstream ctranslate2 wheels: without cuDNN and with CUDA
loaded at runtime, so they produce the same outputs as the upstream wheel with the same
cuBLAS. The cuBLAS version decides the exact floating point results; `import ctranslate2`
loads cuBLAS 12 from, in this order: a `libcublas.so.12` in `LD_LIBRARY_PATH`, the
`nvidia-cublas-cu12` wheel (the `cuda` extra, also pulled by torch's CUDA 12 builds), the
system CUDA. Pin `nvidia-cublas-cu12` (e.g. `==12.8.4.1`) to get the same results everywhere.

The distribution installs the `ctranslate2` module, so it replaces the upstream
`ctranslate2` package and cannot be installed next to it. faster-whisper declares a
`ctranslate2` dependency; install it with `--no-deps`, otherwise pip installs the upstream
package over this one:

```bash
pip uninstall -y ctranslate2
pip install "CTranslate2-rapid-whisper[cuda]"
pip install --no-deps faster-whisper
pip install "av>=11,<16" tokenizers onnxruntime huggingface_hub tqdm
```

Versions: `ctranslate2.__version__` stays at the API-compatible `4.8.3`, and
`ctranslate2.rapido_version` gives the engine version.

## Runtime options

| Variable | Default | Effect |
|---|---|---|
| `CT2_CUDA_GRAPHS` | `0` | Run decoder steps as CUDA graphs. Outputs are bit-identical. |
| `CT2_CUDA_GRAPHS_MAX_STEPS` | `128` | Last decoding step run as a graph, and capacity of the persistent self-attention caches. Longer decodes continue on the regular path. |
| `CT2_CUDA_GRAPHS_MAX_BUFFER_SETS` | `2` | Buffer sets kept, one per decoding shape (batch × beams, encoder output). |
| `CT2_CUDA_GRAPHS_MEMLOG` | `0` | Log the graphs' device memory (info level, e.g. with `CT2_VERBOSE=1`). |
| `CT2_CUDA_GRAPHS_DEVICE_GREEDY` | `1` | Greedy token selection inside the graph (no-timestamp greedy only). |
| `CT2_CUDA_GRAPHS_QUEUE` | `2` | Greedy steps queued before reading their results. |
| `CT2_WHISPER_FOLD_PREFIX` | `0` | Forward a text prefix in one pass. **Not bit-identical**: scores differ in the last fp16 bits. |

Memory: every captured step graph holds ~8 MB of device memory (the launch parameters of
its ~1000 kernels), and each decoding shape keeps persistent buffers: the self-attention
caches at full capacity (twice, ping-pong) and the cross-attention keys/values. Random
sampling (the temperature fallback) and batches that shrink during a call are decoded
without graphs. Measured on an RTX 4090 with large-v3 float16, 82 interview windows of
30 s, beam 3 with temperature fallback:

| | peak device memory | time |
|---|---|---|
| upstream ctranslate2 4.8.1 wheel | 5.5 GB | 92 s |
| rapido, `CT2_CUDA_GRAPHS_MAX_STEPS=128` (default) | 6.7 GB | 83 s |
| rapido, `CT2_CUDA_GRAPHS_MAX_STEPS=448` | 7.9 GB | 83 s |
| rapido, `CT2_CUDA_GRAPHS_MAX_STEPS=96` | 6.4 GB | 87 s |

## Verification and speed

Every release is checked against the upstream `ctranslate2` 4.8.1 wheel from PyPI, the one
production used, with faster-whisper 1.2.1 and Systran/faster-whisper-large-v3 in float16,
the same cuBLAS 12.8.4.1 and a fixed random seed (the temperature fallback samples). The
workload is three long-form interviews (5.6 h): every third 30 s window decoded three ways
(beam 3 with temperature fallback, beam 3 at temperature 0, greedy), and streaming with
LocalAgreement on nine 5-minute segments (the window grows by 0.5 s per round, up to
15 s). Tokens, scores, no-speech probabilities, word timestamps and every streaming round
are compared exactly. rapido runs with `CT2_CUDA_GRAPHS=1`.

| | RTX 5080 (sm_120) | RTX 4090 (sm_89) |
|---|---|---|
| differences in 672 windows and 5400 streaming rounds | 0 | 0 |
| 30 s windows: upstream → rapido | 796 s → 709 s (−11%) | 689 s → 609 s (−12%) |
| streaming with LocalAgreement: upstream → rapido | 2134 s → 1798 s (−16%) | 1681 s → 1544 s (−8%) |

On the RTX 4090 the GPU was shared with other services, so its times are noisier. The
release wheel (graphs up to step 128 by default) measured 751 s and 1923 s on the RTX 5080
in a later run.

Live streaming server (wfs2, agent mode, four recorded sessions replayed in real
time, RTX 5080), ASR time per call:

| | p50 | p95 | ASR calls in the same time |
|---|---|---|---|
| upstream ctranslate2 | 184 ms | 385 ms | 3302 |
| rapido | 163 ms | 316 ms | 3797 |

The faster engine lets LocalAgreement run more, shorter-spaced rounds. The turn texts of
the two runs differ by 8.0%, within the noise of real-time replay: two runs of the
upstream engine differ by 8.9%, because the rounds land at different moments.

The release checks also cover CUDA graph capture next to onnxruntime CUDA sessions on the
same GPU (Silero VAD and SmartTurn run next to Whisper in a streaming server), peak device
memory, and the CrisperWhisper decoding APIs (their tokens match the nyrahealth CTranslate2
fork they come from).

## License

MIT, as CTranslate2. The CrisperWhisper APIs come from nyrahealth/CTranslate2 (MIT). The
CrisperWhisper model weights have their own non-commercial license.
