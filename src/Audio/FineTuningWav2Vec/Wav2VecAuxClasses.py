"""
    Classes to create the architecture of the Wav2Vec2.0 model.
	author: Adapted from the tutorial: https://colab.research.google.com/github/m3hrdadfi/soxan/blob/main/notebooks/Emotion_recognition_in_Greek_speech_using_Wav2Vec2.ipynb
	date: 03/2022
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import torch
from transformers.file_utils import ModelOutput
import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss


from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2PreTrainedModel,
    Wav2Vec2Model
)

@dataclass
class SpeechClassifierOutput(ModelOutput):
    """Class to return certain outputs of the model"""
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None



class Wav2Vec2ClassificationHead(nn.Module):
    """Head for wav2vec classification task. This class stack an MLP on top of the output of the transformer,
    after a pooling layer, defined in Wav2Vec2ForSpeechClassification class"""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size) #Dense of 1024 neurons
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels) #Dense layer of as many neurons as the # of classes, 8 in our case

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class Wav2Vec2ForSpeechClassification(Wav2Vec2PreTrainedModel):
    """Wav2Vec2.0 model with the default architecture plus a pooling layer and the MLP defined in the class Wav2Vec2ClassificationHead."""
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.pooling_mode = config.pooling_mode
        self.config = config

        self.wav2vec2 = Wav2Vec2Model(config) #Default Wav2Vec model
        self.classifier = Wav2Vec2ClassificationHead(config) # MLP stacked on top of the transformer (after the pooling layer)

        self.init_weights()

    def freeze_feature_extractor(self):
        """Function to freeze the layers of the feature encoder composed by a set of CNN layers"""
        self.wav2vec2.feature_extractor._freeze_parameters()

    def merged_strategy(
            self,
            hidden_states,
            mode="mean"
    ):
        """'Pooling layer' that combines the outputs of the transformer for each timestep of the input recording
        by applying a mean, sum or max."""
        if mode == "mean":
            outputs = torch.mean(hidden_states, dim=1)
        elif mode == "sum":
            outputs = torch.sum(hidden_states, dim=1)
        elif mode == "max":
            outputs = torch.max(hidden_states, dim=1)[0]
        else:
            raise Exception(
                "The pooling method hasn't been defined! Your pooling mode must be one of these ['mean', 'sum', 'max']")

        return outputs

    def forward(
            self,
            input_values,
            attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            labels=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0] #Outputs of the transformer module of Wav2Vec2.0 (with their timesteps)
        hidden_states = self.merged_strategy(hidden_states, mode=self.pooling_mode) #output after passing the pooling layer (that reduced the timesteps into a single vector)
        logits = self.classifier(hidden_states) #output after passing from the MLP stacked on top of the transformer and pooling layer

        loss = None
        #Calculate loss depending on the problem: regression, single label classification or multilabel classification
        #  -regression: MSE loss.
        #  -single label  classification: CrossEntropy Loss
        #  -multilabel  classification: BCEWithLogits Loss
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SpeechClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
