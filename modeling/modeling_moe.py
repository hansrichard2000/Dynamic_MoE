# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch LLaMA model."""
import math
from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import torch.nn.functional as F
from typing import Optional

from transformers import PreTrainedModel
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from transformers.utils import logging, add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings

from .configuration_moe import MoEOPTConfig

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LlamaConfig"


# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(input_ids_shape, dtype, device, past_key_values_length=0):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), float("-inf"), device=device)
    mask = torch.triu(mask, diagonal=1)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.bool(), float("-inf")).to(dtype)

class OPTLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.layernorm = nn.LayerNorm(hidden_size, eps=eps)

    def forward(self, hidden_states):
        mean = hidden_states.mean(-1, keepdim=True)
        variance = hidden_states.var(-1, unbiased=False, keepdim=True)
        normalized_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon) 

        return self.weight * normalized_states + self.bias

class OPTMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, activation_function):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.act_fn = ACT2FN[activation_function]

    def forward(self, x):
        return self.fc2(self.act_fn(self.fc1(x)))

def top_p_sampling_batched_all_sequence(logits, top_p=0.9, temperature=1.0):
    """
    Apply Top-p sampling to every element in the sequence for each item in the batch.
    Returns the selected token indices and the corresponding threshold indices.
    
    :param logits: Logits from a language model with shape (sequence length, batch size, L)
    :param top_p: Cumulative probability threshold (float)
    :param temperature: Sampling temperature (float)
    :return: Tuple of tensors (selected token indices, threshold indices) for each position in each sequence in the batch
    """
    # Apply temperature
    logits = logits / temperature
    
    # Convert logits to probabilities
    probs = torch.softmax(logits, dim=-1)
    # Sort probabilities and their indices in descending order
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)

    # Compute cumulative probabilities
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    mask = cumulative_probs > top_p

    # Find the threshold indices
    threshold_indices = mask.long().argmax(dim=-1)
    threshold_mask = torch.nn.functional.one_hot(threshold_indices, num_classes=sorted_indices.size(-1)).bool()
    
    mask = mask & ~threshold_mask
    sorted_indices = torch.where(mask, -1, sorted_indices)
    sorted_probs = torch.where(mask, 0.0, sorted_probs)   
    return sorted_probs, sorted_indices

class SwitchMLP(nn.Module):
    """
    Dynamic Routing Mixture of Experts (Switch MLP) for OPT.
    """
    def __init__(self, config, layer_idx):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.num_experts = config.num_experts
        self.top_p_threshold = config.top_p_threshold
        self.expert_frequency = config.expert_frequency
        
        self.layer_idx = layer_idx
        
        # Only apply MoE every 'expert_frequency' layers
        self.use_moe = (layer_idx % self.expert_frequency) == 0
        
        if self.use_moe:
            self.router = nn.Linear(self.hidden_size, self.num_experts, bias=False)
            self.experts = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.hidden_size, self.intermediate_size),
                    nn.GELU(),
                    nn.Linear(self.intermediate_size, self.hidden_size)
                )
                for _ in range(self.num_experts)
            ])
        else:
            # Fall back to standard MLP
            self.fc1 = nn.Linear(self.hidden_size, self.intermediate_size)
            self.fc2 = nn.Linear(self.intermediate_size, self.hidden_size)
            self.activation_fn = F.gelu
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not self.use_moe:
            hidden_states = self.activation_fn(self.fc1(hidden_states))
            hidden_states = self.fc2(hidden_states)
            return hidden_states

        # Dynamic Routing MLP
        bsz, seq_len, hidden_dim = hidden_states.shape
        route_logits = self.router(hidden_states)  # (bsz, seq_len, num_experts)
        route_probs = torch.softmax(route_logits, dim=-1)

        # Top-p sampling per token
        sorted_probs, sorted_indices = torch.sort(route_probs, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        mask = cumulative_probs > self.top_p_threshold
        

        threshold_indices = mask.long().argmax(dim=-1)
        threshold_mask = F.one_hot(threshold_indices, num_classes=sorted_probs.size(-1)).bool()
        
        mask = mask & ~threshold_mask
        selected_indices = torch.where(mask, -1, sorted_indices)
        flat_hidden_states = hidden_states.view(-1, hidden_dim)
        flat_selected_indices = selected_indices.view(-1, selected_indices.size(-1))
        
        output_total = torch.zeros_like(flat_hidden_states)
        
        for expert_idx, expert in enumerate(self.experts):
            expert_mask = (flat_selected_indices == expert_idx)
            sample_indices = expert_mask.any(dim=-1)

            if sample_indices.any():
                inputs_to_expert = flat_hidden_states[sample_indices]
                outputs_from_expert = expert(inputs_to_expert)
                # 🔍 Patch: debug NaNs, clamp, and log if bad expert
                if torch.any(torch.isnan(outputs_from_expert)) or torch.any(torch.isinf(outputs_from_expert)):
                    print(f"[DEBUG] NaNs/Infs detected in expert {expert_idx}")
                    print("Inputs to expert (min, max):", inputs_to_expert.min().item(), inputs_to_expert.max().item())
                    print("Outputs before clamp (min, max):", outputs_from_expert.min().item(), outputs_from_expert.max().item())
        
                    outputs_from_expert = torch.nan_to_num(outputs_from_expert, nan=0.0, posinf=1e4, neginf=-1e4)
                    outputs_from_expert = torch.clamp(outputs_from_expert, min=-1e4, max=1e4)
        
                    print("Outputs after clamp (min, max):", outputs_from_expert.min().item(), outputs_from_expert.max().item())
                # Accumulate outputs
                output_total[sample_indices] += outputs_from_expert

        output_total = output_total.view(bsz, seq_len, hidden_dim)
        return output_total
        
class OPTAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size)

        self.dropout = nn.Dropout(config.attention_dropout)

    def _split_heads(self, x, bsz, seq_len):
        return x.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x, bsz, seq_len):
        return x.transpose(1, 2).contiguous().view(bsz, seq_len, self.hidden_size)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
    ):
        bsz, seq_len, _ = hidden_states.size()

        query = self._split_heads(self.q_proj(hidden_states), bsz, seq_len)
        key = self._split_heads(self.k_proj(hidden_states), bsz, seq_len)
        value = self._split_heads(self.v_proj(hidden_states), bsz, seq_len)

        if past_key_value is not None:
            past_key, past_value = past_key_value
            key = torch.cat([past_key, key], dim=2)
            value = torch.cat([past_value, value], dim=2)

        present_key_value = (key, value) if use_cache else None

        attn_weights = torch.matmul(query, key.transpose(-1, -2)) * self.scale

        if attention_mask is not None:
            attn_weights += attention_mask

        attn_probs = nn.functional.softmax(attn_weights, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # ✅ Fix dtype mismatch for mixed precision
        attn_probs = attn_probs.to(value.dtype)

        attn_output = torch.matmul(attn_probs, value)
        attn_output = self._merge_heads(attn_output, bsz, seq_len)
        attn_output = self.out_proj(attn_output)

        # ✅ Always return 3-tuple for unpacking
        return (
            attn_output,
            attn_probs if output_attentions else None,
            present_key_value,
        )

class OPTDecoderLayer(nn.Module):
    def __init__(self, config: MoEOPTConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = OPTAttention(config)
        self.mlp = SwitchMLP(config, layer_idx)
        self.self_attn_layer_norm = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_eps)
        self.mlp = SwitchMLP(config, layer_idx)
        self.final_layer_norm = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_eps)

        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ):
        # Self Attention
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
        )

        hidden_states = residual + hidden_states
        hidden_states = self.dropout(hidden_states)

        # Feed Forward / Switch MLP
        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        hidden_states = self.dropout(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs
    
OPT_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its models (such as downloading or saving, resizing the input embeddings, pruning heads,
    etc.)

    This model is also a PyTorch [`torch.nn.Module`](https://pytorch.org/docs/stable/nn.html#torch.nn.Module)
    subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for general usage and behavior.

    Parameters:
        config ([`OPTConfig`] or [`MoEOPTConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

@add_start_docstrings(
    "The bare OPT Model outputting raw hidden-states without any specific head on top.",
    OPT_START_DOCSTRING,
)

class MoEPreTrainedModel(PreTrainedModel):
    config_class = MoEOPTConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["OPTDecoderLayer"]
    _keys_to_ignore_on_load_unexpected = [r"decoder\.version"]

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, MoEOPTModel):
            module.gradient_checkpointing = value

OPT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""
@add_start_docstrings(
    "The bare OPT Model outputting raw hidden-states without any specific head on top.",
    OPT_START_DOCSTRING,
)

class MoEOPTModel(MoEPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`OPTDecoderLayer`]

    Args:
        config: OPTConfig
    """

    def __init__(self, config: MoEOPTConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.embed_positions = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        
        self.dropout = nn.Dropout(config.dropout)
        
        self.layers = nn.ModuleList([OPTDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        # self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value
    # Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask
    @add_start_docstrings_to_model_forward(OPT_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
    
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    
        # Validate input_ids vs. inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time.")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You must specify either input_ids or inputs_embeds.")
    
        # Handle past key values
        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
    
        # Generate position_ids if not provided
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        else:
            position_ids = position_ids.view(batch_size, seq_length).long()
    
        # Input embedding
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
    
        hidden_states = inputs_embeds + self.embed_positions(position_ids)
        hidden_states = self.dropout(hidden_states)
    
        # Attention mask
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length + past_key_values_length),
                dtype=torch.bool,
                device=hidden_states.device,
            )
    
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask,
            input_shape=(batch_size, seq_length),
            inputs_embeds=hidden_states,
            past_key_values_length=past_key_values_length,
        )
    
        # Handle gradient checkpointing
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False
    
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None
    
        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
    
            past_key_value = past_key_values[idx] if past_key_values is not None else None
    
            if self.gradient_checkpointing and self.training:
            
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions, None)
                    return custom_forward
    
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    None,  # OPT layers do not use position_ids
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
    
            hidden_states = layer_outputs[0]
    
            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)
    
            if output_attentions:
                all_self_attns += (layer_outputs[1],)
    
        hidden_states = self.final_layer_norm(hidden_states)
    
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
    
        next_cache = next_decoder_cache if use_cache else None
    
        if not return_dict:
            return tuple(
                v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None
            )
    
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

class MoEForCausalLM(MoEPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = MoEOPTModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(OPT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you consciours? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you consciours? Can you talk to me?\nI'm not consciours, but I can talk to you."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        
        # 🔍 PATCH: clamp & debug unsafe logits
        if torch.any(torch.isnan(logits)) or torch.any(torch.isinf(logits)):
            print("[DEBUG] Detected NaNs or Infs in logits!")
            print("Logits stats before fix: min =", logits.min().item(), "max =", logits.max().item())
            logits = torch.nan_to_num(logits, nan=-1e4, posinf=1e4, neginf=-1e4)
            logits = torch.clamp(logits, min=-1e4, max=1e4)
            print("Logits stats after fix: min =", logits.min().item(), "max =", logits.max().item())

        # Also ensure logits are in float32 for sampling
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
        return reordered_past

@add_start_docstrings(
    """
    The OPT Model transformer with a sequence classification head on top (a linear layer).

    Like GPT-style models, [`OPTForSequenceClassification`] performs classification based on the last token
    in the sequence.

    If `pad_token_id` is defined, it finds the last non-padding token in each row. If `pad_token_id` is not defined,
    it defaults to the last token in each sequence row. When `inputs_embeds` are passed instead of `input_ids`,
    it cannot detect padding and will always use the last token.
    """,
    OPT_START_DOCSTRING,
)

class OPTForSequenceClassification(MoEPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = MoEOPTModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @add_start_docstrings_to_model_forward(OPT_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1).to(logits.device)
            else:
                sequence_lengths = -1

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"
            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
