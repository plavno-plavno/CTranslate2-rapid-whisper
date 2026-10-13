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
- CUDA only, built for Blackwell (sm_120). Ada (sm_89) is planned and is a build flag.
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

Requirements: CUDA 12.8+ (sm_120 needs it), cuDNN 9, CMake ≥ 3.18 and GCC.

```bash
git submodule update --init --recursive
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DWITH_CUDA=ON -DWITH_CUDNN=ON \
  -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-12.8 \
  "-DCUDA_ARCH_FLAGS_OVERRIDE=-gencode;arch=compute_120,code=sm_120;-gencode;arch=compute_120,code=compute_120" \
  -DCMAKE_INSTALL_PREFIX=$PWD/../install
make -j$(nproc) install

cd ../python
CTRANSLATE2_ROOT=$PWD/../install pip install --no-build-isolation .
```

`CUDA_ARCH_FLAGS_OVERRIDE` passes the `-gencode` flags directly, because older CMake
FindCUDA cannot parse two-digit architectures such as 12.0. Set `LD_LIBRARY_PATH` to
`install/lib` at runtime.

The pip distribution is named `rapido-whisper` and installs the `ctranslate2` module.
faster-whisper declares a `ctranslate2` dependency, so install it with `--no-deps`.
Otherwise pip installs the upstream package over this one:

```bash
pip install --no-deps faster-whisper
pip install "av>=11,<16" tokenizers onnxruntime huggingface_hub tqdm
```

Versions: `ctranslate2.__version__` stays at the API-compatible `4.8.3`, and
`ctranslate2.rapido_version` gives the engine version.

## Runtime options

| Variable | Default | Effect |
|---|---|---|
| `CT2_CUDA_GRAPHS` | `0` | Run decoder steps as CUDA graphs. Outputs are bit-identical. |
| `CT2_CUDA_GRAPHS_MAX_STEPS` | `448` | Capacity of the persistent self-attention caches. |
| `CT2_CUDA_GRAPHS_DEVICE_GREEDY` | `1` | Greedy token selection inside the graph (no-timestamp greedy only). |
| `CT2_CUDA_GRAPHS_QUEUE` | `2` | Greedy steps queued before reading their results. |
| `CT2_WHISPER_FOLD_PREFIX` | `0` | Forward a text prefix in one pass. **Not bit-identical**: scores differ in the last fp16 bits. |

## Verification

The equivalence tooling lives outside the library: `compare_ct2.py` (1340 outputs covering
greedy, beam, n-best, batches, prefixes, timestamps and faster-whisper `transcribe` with
word timestamps), `fork_check.py` (long-form interviews and streaming with
LocalAgreement) and `stream_la.py`. Against CTranslate2 4.8.3, with and without CUDA
graphs, they report 0 differing tokens and 0 differing scores, no-speech probabilities and
word times, and identical streaming rounds.

## License

MIT, as CTranslate2. The CrisperWhisper APIs come from nyrahealth/CTranslate2 (MIT). The
CrisperWhisper model weights have their own non-commercial license.
