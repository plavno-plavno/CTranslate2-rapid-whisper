import abc
import argparse
import gc
import itertools
import os

from typing import List, Optional

import numpy as np

try:
    import huggingface_hub
    import torch
    import transformers
except ImportError:
    pass

from ctranslate2.converters import utils
from ctranslate2.converters.converter import Converter
from ctranslate2.specs import (
    attention_spec,
    common_spec,
    model_spec,
    transformer_spec,
    whisper_spec,
)

_SUPPORTED_ACTIVATIONS = {
    "gelu": common_spec.Activation.GELU,
    "gelu_fast": common_spec.Activation.GELUTanh,
    "gelu_new": common_spec.Activation.GELUTanh,
    "gelu_python": common_spec.Activation.GELU,
    "gelu_pytorch_tanh": common_spec.Activation.GELUTanh,
    "quick_gelu": common_spec.Activation.GELUSigmoid,
    "relu": common_spec.Activation.RELU,
    "silu": common_spec.Activation.SWISH,
    "swish": common_spec.Activation.SWISH,
}

_SUPPORTED_ROPE_SCALING = {
    "linear": attention_spec.RotaryScalingType.Linear,
    "su": attention_spec.RotaryScalingType.Su,
    "llama3": attention_spec.RotaryScalingType.Llama3,
    "longrope": attention_spec.RotaryScalingType.Su,
}

_SUPPORTED_QUANTIZATION = {
    "gemm": common_spec.Quantization.AWQ_GEMM,
    "gemv": common_spec.Quantization.AWQ_GEMV,
}

_MODEL_LOADERS = {}


def register_loader(config_name):
    """Registers a model loader for this configuration name."""

    def decorator(cls):
        _MODEL_LOADERS[config_name] = cls()
        return cls

    return decorator


class TransformersConverter(Converter):
    """Converts models from Hugging Face Transformers."""

    def __init__(
        self,
        model_name_or_path: str,
        activation_scales: Optional[str] = None,
        copy_files: Optional[List[str]] = None,
        load_as_float16: bool = False,
        revision: Optional[str] = None,
        low_cpu_mem_usage: bool = False,
        trust_remote_code: bool = False,
    ):
        """Initializes the converter.

        Arguments:
          model_name_or_path: Name of the pretrained model to download, or path to the
            directory containing the pretrained model.
          activation_scales: Path to the pre-computed activation scales. Models may
            use them to rescale some weights to smooth the intermediate activations
            and improve the quantization accuracy. See
            https://github.com/mit-han-lab/smoothquant.
          copy_files: List of filenames to copy from the Hugging Face model to the
            converted model directory.
          load_as_float16: Load the model weights as float16. More precisely, the model
            will be loaded with ``from_pretrained(..., dtype=torch.float16)``.
          revision: Revision of the model to download from the Hugging Face Hub.
          low_cpu_mem_usage: Enable the flag ``low_cpu_mem_usage`` when loading the model
            with ``from_pretrained``.
          trust_remote_code: Allow converting models using custom code.
        """
        self._model_name_or_path = model_name_or_path
        self._activation_scales = activation_scales
        self._copy_files = copy_files
        self._load_as_float16 = load_as_float16
        self._revision = revision
        self._low_cpu_mem_usage = low_cpu_mem_usage
        self._trust_remote_code = trust_remote_code

    def _load(self):
        with torch.no_grad():
            config = transformers.AutoConfig.from_pretrained(
                self._model_name_or_path, trust_remote_code=self._trust_remote_code
            )

            config_name = config.__class__.__name__
            loader = _MODEL_LOADERS.get(config_name)

            if loader is None:
                raise ValueError(
                    "No conversion is registered for the model configuration %s "
                    "(supported configurations are: %s)"
                    % (config_name, ", ".join(sorted(_MODEL_LOADERS.keys())))
                )

            model_class = getattr(transformers, loader.architecture_name)
            if hasattr(loader, "get_model_class"):
                model_class = loader.get_model_class(config, model_class)
            tokenizer_class = transformers.AutoTokenizer

            extra_kwargs = {"config": config}
            if hasattr(loader, "get_model_kwargs"):
                extra_kwargs.update(loader.get_model_kwargs(config))

            kwargs = {
                "dtype": (
                    torch.float16
                    if self._load_as_float16
                    else getattr(config, "dtype", None)
                    or getattr(config, "torch_dtype", None)
                )
            }

            if self._revision:
                kwargs["revision"] = self._revision
            if self._low_cpu_mem_usage:
                kwargs["low_cpu_mem_usage"] = self._low_cpu_mem_usage
            if self._trust_remote_code:
                kwargs["trust_remote_code"] = self._trust_remote_code

            model = self.load_model(
                model_class, self._model_name_or_path, **kwargs, **extra_kwargs
            )

            tokenizer_kwargs = {}
            if self._trust_remote_code:
                tokenizer_kwargs["trust_remote_code"] = self._trust_remote_code

            tokenizer = self.load_tokenizer(
                tokenizer_class, self._model_name_or_path, **tokenizer_kwargs
            )

            spec = loader(model, tokenizer)

            if self._activation_scales:
                activation_scales = torch.load(
                    self._activation_scales, map_location="cpu"
                )
                loader.smooth_activation(spec, activation_scales)

            if self._copy_files:
                for filename in self._copy_files:
                    spec.register_file(self.get_model_file(filename))

            return spec

    def load_model(self, model_class, model_name_or_path, **kwargs):
        return model_class.from_pretrained(model_name_or_path, **kwargs)

    def load_tokenizer(self, tokenizer_class, model_name_or_path, **kwargs):
        return tokenizer_class.from_pretrained(model_name_or_path, **kwargs)

    def get_model_file(self, filename):
        if os.path.isdir(self._model_name_or_path):
            path = os.path.join(self._model_name_or_path, filename)
        else:
            try:
                path = huggingface_hub.hf_hub_download(
                    repo_id=self._model_name_or_path, filename=filename
                )
            except huggingface_hub.utils.EntryNotFoundError:
                path = None

        if path is None or not os.path.isfile(path):
            raise ValueError(
                "File %s does not exist in model %s"
                % (filename, self._model_name_or_path)
            )

        return path


class ModelLoader(abc.ABC):
    """Base class for loading Transformers models into a CTranslate2 model specification."""

    @property
    def architecture_name(self):
        return None

    @abc.abstractmethod
    def get_model_spec(self, model):
        raise NotImplementedError()

    def __call__(self, model, tokenizer):
        spec = self.get_model_spec(model)
        self.set_config(spec.config, model, tokenizer)

        tokens = self.get_vocabulary(model, tokenizer)
        self.set_vocabulary(spec, tokens)

        return spec

    def get_vocabulary(self, model, tokenizer):
        return [
            token
            for token, _ in sorted(
                tokenizer.get_vocab().items(), key=lambda item: item[1]
            )
        ]

    def set_vocabulary(self, spec, tokens):
        pass

    def set_config(self, config, model, tokenizer):
        pass

    def set_layer_norm(self, spec, module):
        spec.gamma = module.weight
        spec.beta = module.bias

    def set_linear(self, spec, module, quant_type=common_spec.Quantization.CT2):
        if quant_type == common_spec.Quantization.CT2:
            spec.weight = module.weight
        else:
            spec.weight = module.qweight
            spec.weight_scale = module.scales
            spec.weight_zero = module.qzeros

        if isinstance(module, transformers.Conv1D):
            spec.weight = spec.weight.transpose(0, 1)
        if hasattr(module, "bias") and module.bias is not None:
            spec.bias = module.bias

    def set_embeddings(self, spec, module):
        spec.weight = module.weight

    def set_position_encodings(self, spec, module):
        spec.encodings = module.weight
        offset = getattr(module, "offset", 0)
        if offset > 0:
            spec.encodings = spec.encodings[offset:]

    def smooth_activation(self, spec, activation_scales):
        raise NotImplementedError(
            "No activation smoothing logic is defined for this model"
        )

    def get_rotary_params(self, config, default_rope_theta):
        rope_scaling = getattr(config, "rope_scaling", None)
        if rope_scaling:
            rope_type = rope_scaling.get("type") or rope_scaling.get("rope_type")

            if rope_type == "default":
                rotary_scaling_type = None
            else:
                rotary_scaling_type = _SUPPORTED_ROPE_SCALING.get(rope_type)
                if rotary_scaling_type is None:
                    raise NotImplementedError(
                        "RoPE scaling type '%s' is not yet implemented. "
                        "The following RoPE scaling types are currently supported: %s"
                        % (rope_type, ", ".join(_SUPPORTED_ROPE_SCALING.keys()))
                    )
            rotary_scaling_factor = rope_scaling.get("factor", 1)
            rope_theta = rope_scaling.get("rope_theta", default_rope_theta)
        else:
            rotary_scaling_type = None
            rotary_scaling_factor = 1
            rope_theta = getattr(config, "rope_theta", default_rope_theta)

        return rotary_scaling_type, rotary_scaling_factor, rope_theta


# Encoder-decoder weights layout shared with BART; base of WhisperLoader only.
class BartLoader(ModelLoader):
    @property
    def architecture_name(self):
        return "BartForConditionalGeneration"

    def get_model_spec(self, model):
        spec = transformer_spec.TransformerSpec.from_config(
            (model.config.encoder_layers, model.config.decoder_layers),
            model.config.encoder_attention_heads,
            pre_norm=model.config.normalize_before,
            activation=_SUPPORTED_ACTIVATIONS[model.config.activation_function],
            layernorm_embedding=getattr(model.config, "normalize_embedding", True),
        )

        self.set_encoder(spec.encoder, model.model.encoder)
        self.set_decoder(spec.decoder, model.model.decoder)
        self.set_linear(spec.decoder.projection, model.lm_head)

        final_logits_bias = getattr(model, "final_logits_bias", None)
        if final_logits_bias is not None and final_logits_bias.nonzero().numel() != 0:
            spec.decoder.projection.bias = final_logits_bias.squeeze()

        return spec

    def get_vocabulary(self, model, tokenizer):
        tokens = super().get_vocabulary(model, tokenizer)
        if model.config.vocab_size < len(tokens):
            tokens = tokens[: model.config.vocab_size]
        return tokens

    def set_vocabulary(self, spec, tokens):
        spec.register_source_vocabulary(tokens)
        spec.register_target_vocabulary(tokens)

    def set_config(self, config, model, tokenizer):
        config.bos_token = tokenizer.bos_token
        config.eos_token = tokenizer.eos_token
        config.unk_token = tokenizer.unk_token
        config.decoder_start_token = tokenizer.convert_ids_to_tokens(
            model.config.decoder_start_token_id
        )

    def set_encoder(self, spec, encoder):
        self.set_common_layers(spec, encoder)

        for layer_spec, layer in zip(spec.layer, encoder.layers):
            self.set_attention(
                layer_spec.self_attention,
                layer.self_attn,
                self_attention=True,
            )
            self.set_layer_norm(
                layer_spec.self_attention.layer_norm,
                layer.self_attn_layer_norm,
            )

            self.set_linear(layer_spec.ffn.linear_0, layer.fc1)
            self.set_linear(layer_spec.ffn.linear_1, layer.fc2)
            self.set_layer_norm(layer_spec.ffn.layer_norm, layer.final_layer_norm)

    def set_decoder(self, spec, decoder):
        self.set_common_layers(spec, decoder)

        for layer_spec, layer in zip(spec.layer, decoder.layers):
            self.set_attention(
                layer_spec.self_attention,
                layer.self_attn,
                self_attention=True,
            )
            self.set_layer_norm(
                layer_spec.self_attention.layer_norm,
                layer.self_attn_layer_norm,
            )

            if hasattr(layer, "encoder_attn"):
                self.set_attention(
                    layer_spec.attention,
                    layer.encoder_attn,
                    self_attention=False,
                )
                self.set_layer_norm(
                    layer_spec.attention.layer_norm,
                    layer.encoder_attn_layer_norm,
                )

            self.set_linear(layer_spec.ffn.linear_0, layer.fc1)
            self.set_linear(layer_spec.ffn.linear_1, layer.fc2)
            self.set_layer_norm(layer_spec.ffn.layer_norm, layer.final_layer_norm)

    def set_attention(self, spec, attention, self_attention=False):
        split_layers = [common_spec.LinearSpec() for _ in range(3)]
        self.set_linear(split_layers[0], attention.q_proj)
        self.set_linear(split_layers[1], attention.k_proj)
        self.set_linear(split_layers[2], attention.v_proj)

        if self_attention:
            utils.fuse_linear(spec.linear[0], split_layers)
        else:
            utils.fuse_linear(spec.linear[0], split_layers[:1])
            utils.fuse_linear(spec.linear[1], split_layers[1:])

        self.set_linear(spec.linear[-1], attention.out_proj)

    def set_common_layers(self, spec, module):
        import math

        if not hasattr(module, "embed_scale"):
            embed_scale = (
                math.sqrt(module.config.d_model)
                if module.config.scale_embedding
                else 1.0
            )
        else:
            embed_scale = module.embed_scale
        spec.scale_embeddings = embed_scale
        self.set_position_encodings(spec.position_encodings, module.embed_positions)
        self.set_embeddings(
            (
                spec.embeddings[0]
                if isinstance(spec.embeddings, list)
                else spec.embeddings
            ),
            module.embed_tokens,
        )

        if hasattr(module, "layer_norm"):
            self.set_layer_norm(spec.layer_norm, module.layer_norm)
        if hasattr(module, "layernorm_embedding"):
            self.set_layer_norm(spec.layernorm_embedding, module.layernorm_embedding)


@register_loader("WhisperConfig")
class WhisperLoader(BartLoader):
    @property
    def architecture_name(self):
        return "WhisperForConditionalGeneration"

    def get_model_spec(self, model):
        spec = whisper_spec.WhisperSpec(
            model.config.encoder_layers,
            model.config.encoder_attention_heads,
            model.config.decoder_layers,
            model.config.decoder_attention_heads,
        )

        self.set_encoder(spec.encoder, model.model.encoder)
        self.set_decoder(spec.decoder, model.model.decoder)
        self.set_linear(spec.decoder.projection, model.proj_out)

        return spec

    def _get_lang_ids_from_tokenizer(self, tokenizer):
        non_lang_special_tokens = [
            "<|endoftext|>",
            "<|startoftranscript|>",
            "<|translate|>",
            "<|transcribe|>",
            "<|startoflm|>",
            "<|startofprev|>",
            "<|nocaptions|>",
            "<|notimestamps|>",
        ]

        additional_tokens = getattr(tokenizer, "additional_special_tokens", [])
        if not additional_tokens:
            return []

        return [
            tokenizer.convert_tokens_to_ids(token)
            for token in additional_tokens
            if token not in non_lang_special_tokens
        ]

    def set_config(self, config, model, tokenizer):
        gen_config = getattr(model, "generation_config", None)

        if gen_config is not None:
            config.suppress_ids = gen_config.suppress_tokens
            config.suppress_ids_begin = gen_config.begin_suppress_tokens
            if hasattr(gen_config, "alignment_heads"):
                config.alignment_heads = gen_config.alignment_heads
            if hasattr(gen_config, "lang_to_id"):
                config.lang_ids = sorted(gen_config.lang_to_id.values())
        else:
            config.suppress_ids = model.config.suppress_tokens
            config.suppress_ids_begin = model.config.begin_suppress_tokens
            config.alignment_heads = _WHISPER_ALIGNMENT_HEADS.get(model.name_or_path)

        if getattr(config, "lang_ids", None) is None:
            config.lang_ids = self._get_lang_ids_from_tokenizer(tokenizer)

        if config.alignment_heads is None:
            # Use the last half layers for alignment by default.
            num_layers = model.config.decoder_layers
            num_heads = model.config.decoder_attention_heads
            config.alignment_heads = list(
                itertools.product(
                    range(num_layers // 2, num_layers),
                    range(num_heads),
                )
            )

    def get_vocabulary(self, model, tokenizer):
        tokens = super().get_vocabulary(model, tokenizer)

        # Add timestamp tokens.
        tokens.extend(
            "<|%.2f|>" % (i * 0.02)
            for i in range(model.config.vocab_size - len(tokens))
        )

        return tokens

    def set_vocabulary(self, spec, tokens):
        spec.register_vocabulary(tokens)

    def set_encoder(self, spec, encoder):
        self.set_conv1d(spec.conv1, encoder.conv1)
        self.set_conv1d(spec.conv2, encoder.conv2)
        super().set_encoder(spec, encoder)

    def set_decoder(self, spec, decoder):
        self.set_embeddings(spec.embeddings, decoder.embed_tokens)
        super().set_decoder(spec, decoder)

    def set_common_layers(self, spec, module):
        self.set_position_encodings(spec.position_encodings, module.embed_positions)
        self.set_layer_norm(spec.layer_norm, module.layer_norm)

    def set_conv1d(self, spec, module):
        spec.weight = module.weight
        spec.bias = module.bias


_WHISPER_ALIGNMENT_HEADS = {
    "openai/whisper-tiny.en": [
        (1, 0),
        (2, 0),
        (2, 5),
        (3, 0),
        (3, 1),
        (3, 2),
        (3, 3),
        (3, 4),
    ],
    "openai/whisper-tiny": [(2, 2), (3, 0), (3, 2), (3, 3), (3, 4), (3, 5)],
    "openai/whisper-base.en": [(3, 3), (4, 7), (5, 1), (5, 5), (5, 7)],
    "openai/whisper-base": [
        (3, 1),
        (4, 2),
        (4, 3),
        (4, 7),
        (5, 1),
        (5, 2),
        (5, 4),
        (5, 6),
    ],
    "openai/whisper-small.en": [
        (6, 6),
        (7, 0),
        (7, 3),
        (7, 8),
        (8, 2),
        (8, 5),
        (8, 7),
        (9, 0),
        (9, 4),
        (9, 8),
        (9, 10),
        (10, 0),
        (10, 1),
        (10, 2),
        (10, 3),
        (10, 6),
        (10, 11),
        (11, 2),
        (11, 4),
    ],
    "openai/whisper-small": [
        (5, 3),
        (5, 9),
        (8, 0),
        (8, 4),
        (8, 7),
        (8, 8),
        (9, 0),
        (9, 7),
        (9, 9),
        (10, 5),
    ],
    "openai/whisper-medium.en": [
        (11, 4),
        (14, 1),
        (14, 12),
        (14, 14),
        (15, 4),
        (16, 0),
        (16, 4),
        (16, 9),
        (17, 12),
        (17, 14),
        (18, 7),
        (18, 10),
        (18, 15),
        (20, 0),
        (20, 3),
        (20, 9),
        (20, 14),
        (21, 12),
    ],
    "openai/whisper-medium": [(13, 15), (15, 4), (15, 15), (16, 1), (20, 0), (23, 4)],
    "openai/whisper-large": [
        (9, 19),
        (11, 2),
        (11, 4),
        (11, 17),
        (22, 7),
        (22, 11),
        (22, 17),
        (23, 2),
        (23, 15),
    ],
    "openai/whisper-large-v2": [
        (10, 12),
        (13, 17),
        (16, 11),
        (16, 12),
        (16, 13),
        (17, 15),
        (17, 16),
        (18, 4),
        (18, 11),
        (18, 19),
        (19, 11),
        (21, 2),
        (21, 3),
        (22, 3),
        (22, 9),
        (22, 12),
        (23, 5),
        (23, 7),
        (23, 13),
        (25, 5),
        (26, 1),
        (26, 12),
        (27, 15),
    ],
    "openai/whisper-large-v3": [
        (7, 0),
        (10, 17),
        (12, 18),
        (13, 12),
        (16, 1),
        (17, 14),
        (19, 11),
        (21, 4),
        (24, 1),
        (25, 6),
    ],
}


# Paper: https://arxiv.org/pdf/2504.06225


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model",
        required=True,
        help=(
            "Name of the pretrained model to download, "
            "or path to a directory containing the pretrained model."
        ),
    )
    parser.add_argument(
        "--activation_scales",
        help=(
            "Path to the pre-computed activation scales. Models may "
            "use them to rescale some weights to smooth the intermediate activations "
            "and improve the quantization accuracy. See "
            "https://github.com/mit-han-lab/smoothquant."
        ),
    )
    parser.add_argument(
        "--copy_files",
        nargs="+",
        help=(
            "List of filenames to copy from the Hugging Face model to the converted "
            "model directory."
        ),
    )
    parser.add_argument(
        "--revision",
        help="Revision of the model to download from the Hugging Face Hub.",
    )
    parser.add_argument(
        "--low_cpu_mem_usage",
        action="store_true",
        help="Enable the flag low_cpu_mem_usage when loading the model with from_pretrained.",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Allow converting models using custom code.",
    )

    Converter.declare_arguments(parser)
    args = parser.parse_args()
    converter = TransformersConverter(
        args.model,
        activation_scales=args.activation_scales,
        copy_files=args.copy_files,
        load_as_float16=args.quantization == "float16",
        revision=args.revision,
        low_cpu_mem_usage=args.low_cpu_mem_usage,
        trust_remote_code=args.trust_remote_code,
    )
    converter.convert_from_args(args)


if __name__ == "__main__":
    main()


# Cross-attention heads that are highly correlated to the word-level timing,
# i.e. the alignment between audio and text tokens.
# Obtained from https://github.com/openai/whisper/blob/v20231106/whisper/__init__.py#L32-L47
