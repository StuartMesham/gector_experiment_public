from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import AutoModel, PreTrainedModel
from transformers.file_utils import add_start_docstrings_to_model_forward, add_code_sample_docstrings, \
    add_start_docstrings, ModelOutput
from transformers.models.roberta.modeling_roberta import ROBERTA_INPUTS_DOCSTRING, _TOKENIZER_FOR_DOC, \
    _CHECKPOINT_FOR_DOC, _CONFIG_FOR_DOC, ROBERTA_START_DOCSTRING


@dataclass
class GectorOutput(ModelOutput):
    """
    Base class for outputs of token classification models.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) :
            Classification loss.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`):
            Classification scores (before SoftMax).
        logits_d (`torch.FloatTensor` of shape `(batch_size, sequence_length, 3)`):
            Error detector classification scores (before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    logits_d: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@add_start_docstrings(
    """
    GECToR Model with a tagging and detection heads. Can be used with any huggingface encoder.
    """,
    ROBERTA_START_DOCSTRING,
)
class GectorForTokenClassification(PreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    base_model_prefix = "transformer"

    def __init__(self, config, transformer=None):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.unk_tag_idx = config.label2id.get('@@UNKNOWN@@', None)

        if transformer is None:
            self.transformer = AutoModel.from_config(config)
        else:
            self.transformer = transformer

        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        if self.unk_tag_idx is not None:
            self.error_detector = nn.Linear(config.hidden_size, 3)
        else:
            self.error_detector = nn.Linear(config.hidden_size, 2)

        # Initialize weights and apply final processing
        if transformer is not None:  # if we're not starting from a gector checkpoint
            self._init_weights(self.classifier)
            self._init_weights(self.error_detector)

    @add_start_docstrings_to_model_forward(ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=GectorOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.classifier(sequence_output)
        logits_d = self.error_detector(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            labels_d = torch.ones_like(labels)
            labels_d[labels == 0] = 0
            labels_d[labels == -100] = -100
            if self.unk_tag_idx is not None:
                labels_d[labels == self.unk_tag_idx] = 2
                loss_d = loss_fct(logits_d.view(-1, 3), labels_d.view(-1))
            else:
                loss_d = loss_fct(logits_d.view(-1, 2), labels_d.view(-1))

            loss += loss_d

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return GectorOutput(
            loss=loss,
            logits=logits,
            logits_d=logits_d,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def get_input_embeddings(self):
        return self.transformer.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.transformer.set_input_embeddings(value)

    """
    This should only be called when using model.resize_token_embeddings() in run_gector.py
    and to initialize self.classifier and self.error_detector
    """
    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            assert module in [self.classifier, self.error_detector]
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        else:
            raise Exception('do not initialize this')
