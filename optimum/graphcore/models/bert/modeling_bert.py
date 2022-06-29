# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import poptorch
from optimum.utils import logging
from scipy.stats import truncnorm
from transformers import (
    BertForMaskedLM,
    BertForMultipleChoice,
    BertForPreTraining,
    BertForQuestionAnswering,
    BertForSequenceClassification,
    BertForTokenClassification,
)

from transformers.utils.fx import _gen_constructor_wrapper

from ....fx.optimization import ChangeTrueDivToMulByInverse, FuseBiasInLinear, MergeLinears, compose
from ...fx.transformations import (
    AddPoptorchBlock,
    AddPoptorchBlocksInSeries,
    ClipValues,
    LinearToSerializedLinear,
    OutlineAttribute,
    RecomputationCheckpoint,
    TupleOutput,
    VocabEmbeddingToSerializedEmbedding,
)
from ...fx.utils import symbolic_trace_pipelined_model
from ...modeling_utils import (
    OnehotGather,
    PipelineMixin,
    SerializedLinear,
    get_layer_ipu,
    outline_attribute,
    recomputation_checkpoint,
    register,
)


logger = logging.get_logger(__name__)


@register(BertForPreTraining)
class PipelinedBertForPreTraining(BertForPreTraining, PipelineMixin):
    """
    BertForPretraining transformed to run in an IPU pipeline.

    Recommended usage:
    ```
    model = PipelinedBertForPretraining(config).parallelize().half().train()
    ```
    """

    def __init__(self, config):
        super().__init__(config)
        self.gather_indices = OnehotGather()

    def get_ops_to_wrap_for_tracing(self):
        return [
            ("torch.topk", *_gen_constructor_wrapper(torch.topk)),
            ("torch.nn.functional.one_hot", *_gen_constructor_wrapper(torch.nn.functional.one_hot)),
        ]

    def get_transformations(self):
        log_insertions = self.ipu_config.log_insertions
        layer_ipu = get_layer_ipu(self.ipu_config.layers_per_ipu)
        transformations = [
            ChangeTrueDivToMulByInverse(),
            MergeLinears(),
            # FuseBiasInLinear(),
            AddPoptorchBlock(
                "Embedding", layer_ipu=0, module_name_regex="bert.embeddings", log_insertions=log_insertions
            ),
            OutlineAttribute("bert.embeddings.LayerNorm", "Embedding"),
            AddPoptorchBlocksInSeries(
                "Encoder", layer_ipu, module_name_regex=r"bert.encoder.layer.[0-9]+", log_insertions=log_insertions
            ),
            AddPoptorchBlock("Pooler Output", 0, "bert.pooler", log_insertions=log_insertions),
            AddPoptorchBlock("Classifier Output", 0, "cls", log_insertions=log_insertions),
        ]
        return transformations

    def parallelize(self):
        """
        Transform the model to run in an IPU pipeline.
        - Adds pipeline stages to the model
        - Replaces self-attention layers with fused-qkv self-attention layers
        - (If enabled) Replaces the word embedding projection with a SerializedLinear layer
        - Adds recomputation checkpoints
        """
        super().parallelize()
        traced = symbolic_trace_pipelined_model(self)
        transformations = self.get_transformations()
        # if self.ipu_config.embedding_serialization_factor > 1:
        #     transformations.append(LinearToSerializedLinear("cls.predictions.decoder"))
        composition = compose(*transformations)
        non_reversible_composition = compose(ClipValues(1e4), TupleOutput())
        traced = composition(traced)
        traced = non_reversible_composition(traced)
        return traced

    def deparallelize(self):
        """
        Undo the changes to the model done by `parallelize`.
        You should call this before doing `save_pretrained` so that the `model.state_dict` is
        compatible with the original model.
        """
        super().deparallelize()
        transformations = self.get_transformations()
        if self.ipu_config.embedding_serialization_factor > 1:
            transformations.append(LinearToSerializedLinear("cls.predictions.decoder"))
        composition = compose(*transformations)
        self = composition(self, reverse=True)
        return self

    def _init_weights(self, module):
        """Initialize the weights"""

        def truncated_normal_(tensor, mean=0, std=1):
            """
            Truncated Normal distribution, truncated at 2 sigma
            """
            r = torch.tensor(truncnorm.rvs(-2, 2, loc=mean, scale=std, size=tensor.shape))
            tensor.data.copy_(r)

        if isinstance(module, nn.Linear):
            truncated_normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            truncated_normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        next_sentence_label=None,
    ):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output, pooled_output = output[:2]

        if labels is not None:
            # Select only the masked tokens for the classifier
            max_number_of_masked_tokens = math.floor(labels.size(1) * 0.25)
            masked_lm_labels, masked_lm_positions = torch.topk(labels, k=max_number_of_masked_tokens, dim=1)
            masked_output = self.gather_indices(sequence_output, masked_lm_positions)
        else:
            # This case should never happen during training
            masked_output = sequence_output

        prediction_scores, sequential_relationship_score = self.cls(masked_output, pooled_output)
        output = (
            prediction_scores,
            sequential_relationship_score,
        ) + output[2:]

        if labels is not None and next_sentence_label is not None:
            masked_lm_loss = F.cross_entropy(
                prediction_scores.view(-1, self.config.vocab_size),
                masked_lm_labels.view(-1),
                ignore_index=-100,
            ).float()
            next_sentence_loss = F.cross_entropy(
                sequential_relationship_score.view(-1, 2), next_sentence_label.view(-1)
            ).float()
            total_loss = poptorch.identity_loss(masked_lm_loss + next_sentence_loss, reduction="none")
            return (total_loss, masked_lm_loss, next_sentence_loss)

        return output


@register(BertForMaskedLM)
class PipelinedBertForMaskedLM(BertForMaskedLM, PipelineMixin):
    """
    BertForMaskedLM transformed to run in an IPU pipeline.

    Recommended usage:
    ```
    model = PipelinedBertForMaskedLM(config).parallelize().half().train()
    ```
    """

    def __init__(self, config):
        super().__init__(config)
        self.gather_indices = OnehotGather()

    def get_ops_to_wrap_for_tracing(self):
        return [
            ("torch.topk", *_gen_constructor_wrapper(torch.topk)),
            ("torch.nn.functional.one_hot", *_gen_constructor_wrapper(torch.nn.functional.one_hot)),
        ]

    def get_transformations(self):
        log_insertions = self.ipu_config.log_insertions
        layer_ipu = get_layer_ipu(self.ipu_config.layers_per_ipu)
        transformations = [
            AddPoptorchBlock(
                "Embedding", layer_ipu=0, module_name_regex="bert.embeddings", log_insertions=log_insertions
            ),
            OutlineAttribute("bert.embeddings.LayerNorm", "Embedding"),
            AddPoptorchBlocksInSeries(
                "Encoder", layer_ipu, module_name_regex=r"bert.encoder.layer.[0-9]+", log_insertions=log_insertions
            ),
            AddPoptorchBlock("Classifier Output", 0, "cls", log_insertions=log_insertions),
        ]
        if self.ipu_config.recompute_checkpoint_every_layer:
            transformations.append(RecomputationCheckpoint("bert.encoder.layer.[0-9]+", to_exclude=f"bert.encoder.layer.{self.config.num_hidden_layers - 1}"))
        if self.ipu_config.embedding_serialization_factor > 1:
            transformations.append(LinearToSerializedLinear("cls.predictions.decoder"))
        transformations += [ChangeTrueDivToMulByInverse(), MergeLinears()]
        return transformations

    def parallelize(self):
        """
        Transform the model to run in an IPU pipeline.
        - Adds pipeline stages to the model
        - Replaces self-attention layers with fused-qkv self-attention layers
        - (If enabled) Replaces the word embedding projection with a SerializedLinear layer
        - Adds recomputation checkpoints
        """
        super().parallelize()
        traced = symbolic_trace_pipelined_model(self)
        transformations = self.get_transformations()
        composition = compose(*transformations)
        non_reversible_composition = compose(ClipValues(1e4), TupleOutput())

        traced = composition(traced)
        traced = non_reversible_composition(traced)
        import pdb; pdb.set_trace()

        return traced

    def deparallelize(self):
        """
        Undo the changes to the model done by `parallelize`.
        You should call this before doing `save_pretrained` so that the `model.state_dict` is
        compatible with the original model.
        """
        super().deparallelize()

        transformations = self.get_transformations()
        composition = compose(*transformations)
        self = composition(self, reverse=True)

        return self

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        if self.training:
            output = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            sequence_output = output[0]

            # Select only the masked tokens for the classifier
            max_number_of_masked_tokens = math.floor(labels.size(1) * 0.25)
            masked_lm_labels, masked_lm_positions = torch.topk(labels, k=max_number_of_masked_tokens, dim=1)
            masked_output = self.gather_indices(sequence_output, masked_lm_positions)

            prediction_scores = self.cls(masked_output)
            output = (prediction_scores,) + output[2:]

            masked_lm_loss = F.cross_entropy(
                prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1)
            ).float()
            return (masked_lm_loss,)

        else:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels,
                return_dict=False,
            )


class BertPipelineMixin(PipelineMixin):
    def get_transformations(self):
        log_insertions = self.ipu_config.log_insertions
        layer_ipu = get_layer_ipu(self.ipu_config.layers_per_ipu)
        last_ipu = len(self.ipu_config.layers_per_ipu) - 1
        transformations = [
            ChangeTrueDivToMulByInverse(),
            MergeLinears(),
            AddPoptorchBlock(
                "Embedding", layer_ipu=0, module_name_regex="bert.embeddings", log_insertions=log_insertions
            ),
            OutlineAttribute("bert.embeddings.LayerNorm", "Embedding"),
            AddPoptorchBlocksInSeries(
                "Encoder", layer_ipu, module_name_regex=r"bert.encoder.layer.[0-9]+", log_insertions=log_insertions
            ),
            # Only one of the following AddPoptorchBlock, will actually add a block.
            AddPoptorchBlock("Classifier Output", last_ipu, "classifier", log_insertions=log_insertions),
            AddPoptorchBlock("QA Outputs", last_ipu, "qa_outputs", log_insertions=log_insertions),
        ]
        return transformations

    @property
    def input_names(self):
        return ["input_ids", "attention_mask", "token_type_ids", "labels"]

    def parallelize(self):
        """
        Transform the model to run in an IPU pipeline.
        - Adds pipeline stages to the model
        - Replaces self-attention layers with fused-qkv self-attention layers
        - (If enabled) Replaces the word embedding with a SerializedEmbedding
        - Adds recomputation checkpoints
        """
        super().parallelize()

        # if self.ipu_config.recompute_checkpoint_every_layer:
        #     for layer in self.bert.encoder.layer[:-1]:
        #         h = recomputation_checkpoint(layer)
        #         self._hooks.append(h)

        traced = symbolic_trace_pipelined_model(self)

        transformations = self.get_transformations()

        if traced.ipu_config.embedding_serialization_factor > 1:
            transformations.append(VocabEmbeddingToSerializedEmbedding())

        composition = compose(*transformations)

        non_reversible_composition = compose(ClipValues(1e4), TupleOutput())

        traced = composition(traced)
        traced = non_reversible_composition(traced)

        return traced

    def deparallelize(self):
        """
        Undo the changes to the model done by `parallelize`.
        You should call this before doing `save_pretrained` so that the `model.state_dict` is
        compatible with the original model.
        """
        super().deparallelize()

        transformations = self.get_transformations()
        if self.ipu_config.embedding_serialization_factor > 1:
            transformations.append(VocabEmbeddingToSerializedEmbedding())

        # if self.ipu_config.recompute_checkpoint_every_layer:
        #     transformations.append(RecomputationCheckpoint())

        composition = compose(*transformations)
        self = composition(self, reverse=True)

        return self


@register(BertForSequenceClassification)
class PipelinedBertForSequenceClassification(BertForSequenceClassification, BertPipelineMixin):
    """
    BertForSequenceClassification transformed to run in an IPU pipeline.

    Recommended usage:
    ```
    model = PipelinedBertForSequenceClassification(config).parallelize().half()
    ```
    """
    pass


@register(BertForMultipleChoice)
class PipelinedBertForMultipleChoice(BertForMultipleChoice, BertPipelineMixin):
    """
    BertForMultipleChoice transformed to run in an IPU pipeline.

    Recommended usage:
    ```
    model = PipelinedBertForMultipleChoice(config).parallelize().half()
    ```
    """
    pass


@register(BertForTokenClassification)
class PipelinedBertForTokenClassification(BertForTokenClassification, BertPipelineMixin):
    """
    BertForTokenClassification transformed to run in an IPU pipeline.

    Recommended usage:
    ```
    model = PipelinedBertForTokenClassification(config).parallelize().half()
    ```
    """
    pass


@register(BertForQuestionAnswering)
class PipelinedBertForQuestionAnswering(BertForQuestionAnswering, BertPipelineMixin):
    """
    BertForQuestionAnswering transformed to run in an IPU pipeline.

    Recommended usage:
    ```
    model = PipelinedBertForQuestionAnswering(config).parallelize().half()
    ```
    """

    @property
    def input_names(self):
        return ["input_ids", "attention_mask", "token_type_ids", "start_positions", "end_positions"]

    def forward(self, input_ids, attention_mask, token_type_ids, start_positions=None, end_positions=None):
        output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            start_positions=start_positions,
            end_positions=end_positions,
            return_dict=False,
        )
        if start_positions is not None and end_positions is not None:
            output = (poptorch.identity_loss(output[0], reduction="none"),) + output[1:]
        return output
