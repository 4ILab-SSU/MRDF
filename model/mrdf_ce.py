from typing import Dict, List, Optional, Union, Sequence

import torch
import numpy as np
from torch import Tensor
import torch.nn as nn
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.loss import ContrastLoss, MarginLoss


import fairseq
from fairseq.modules import LayerNorm
from fairseq.data.dictionary import Dictionary
import model.avhubert.hubert as hubert
import model.avhubert.hubert_pretraining as hubert_pretraining
import torchmetrics


def Average(lst):
    return sum(lst) / len(lst)


def Opposite(a):
    a = a + 1
    a[a > 1.5] = 0
    return a


class MRDF_CE(nn.Module):

    def __init__(self,
                 margin_contrast=0.0, margin_audio=0.0, margin_visual=0.0, weight_decay=0.0001, learning_rate=0.0002, distributed=False
                 ):
        super().__init__()
        self.model = hubert.AVHubertModel(cfg=hubert.AVHubertConfig, task_cfg=hubert_pretraining.AVHubertPretrainingConfig, dictionaries=hubert_pretraining.AVHubertPretrainingTask)

        self.embed = 768
        self.dropout = 0.1

        self.feature_extractor_audio_hubert = self.model.feature_extractor_audio
        self.feature_extractor_video_hubert = self.model.feature_extractor_video

        self.project_audio = nn.Sequential(LayerNorm(self.embed), nn.Linear(self.embed, self.embed), nn.Dropout(self.dropout))

        self.project_video = nn.Sequential(LayerNorm(self.embed), nn.Linear(self.embed, self.embed), nn.Dropout(self.dropout))

        self.project_hubert = nn.Sequential(self.model.layer_norm, self.model.post_extract_proj, self.model.dropout_input)

        self.fusion_encoder_hubert = self.model.encoder

        self.final_proj_audio = self.model.final_proj
        self.final_proj_video = self.model.final_proj
        self.final_proj_hubert = self.model.final_proj

        self.video_classifier = nn.Sequential(nn.Linear(self.embed, 2))
        self.audio_classifier = nn.Sequential(nn.Linear(self.embed, 2))
        # #
        self.mm_classifier = nn.Sequential(nn.Linear(self.embed, self.embed), nn.ReLU(inplace=True), nn.Linear(self.embed, 2))

        self.contrast_loss = ContrastLoss(loss_fn=nn.CosineSimilarity(dim=-1), margin=margin_contrast)
        self.mm_cls = CrossEntropyLoss()
        self.a_cls = CrossEntropyLoss()
        self.v_cls = CrossEntropyLoss()

        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.distributed = distributed

        self.acc = torchmetrics.classification.BinaryAccuracy()
        self.auroc = torchmetrics.classification.BinaryAUROC(thresholds=None)
        self.f1score = torchmetrics.classification.BinaryF1Score()
        self.recall = torchmetrics.classification.BinaryRecall()
        self.precisions = torchmetrics.classification.BinaryPrecision()

        self.best_loss = 1e9
        self.best_acc, self.best_auroc = 0.0, 0.0
        self.best_real_f1score, self.best_real_recall,  self.best_real_precision = 0.0, 0.0, 0.0
        self.best_fake_f1score, self.best_fake_recall, self.best_fake_precision = 0.0, 0.0, 0.0

        self.softmax = nn.Softmax(dim=1)

    def forward(self, video: Tensor, audio: Tensor, mask: Tensor):
        # print(audio.shape, video.shape)
        a_features = self.feature_extractor_audio_hubert(audio).transpose(1, 2)
        v_features = self.feature_extractor_video_hubert(video).transpose(1, 2)
        av_features = torch.cat([a_features, v_features], dim=2)

        a_cross_embeds = a_features.mean(1)
        v_cross_embeds = v_features.mean(1)

        a_features = self.project_audio(a_features)
        v_features = self.project_video(v_features)
        av_features = self.project_hubert(av_features)

        a_embeds = a_features.mean(1)
        v_embeds = v_features.mean(1)

        a_embeds = self.audio_classifier(a_embeds)
        v_embeds = self.video_classifier(v_embeds)

        av_features, _ = self.fusion_encoder_hubert(av_features, padding_mask=mask)
        m_logits = self.mm_classifier(av_features[:, 0, :])

        return m_logits, v_cross_embeds, a_cross_embeds, v_embeds, a_embeds

    def get_avg_feat(self, feat, mask):
        mask_un = mask.to(dtype=torch.float).unsqueeze(1)
        feat = feat * mask_un
        mask_un_sum = torch.sum(mask_un, dim=1, dtype=torch.float)
        mask_un_sum[mask_un_sum == 0.] = 1.
        feat = torch.sum(feat, dim=1) / mask_un_sum
        return feat

    def loss_fn(self, m_logits, v_feats, a_feats, v_logits, a_logits, v_label, a_label, c_label, m_label) -> Dict[str, Tensor]:

        contrast_loss = self.contrast_loss(v_feats, a_feats, c_label)
        a_loss = self.a_cls(a_logits, a_label)
        v_loss = self.v_cls(v_logits, v_label)

        mm_loss = self.mm_cls(m_logits, m_label)
        loss = mm_loss + a_loss + v_loss + contrast_loss

        return {"loss": loss, "mm_loss": mm_loss}

    def training_step(self, batch: Optional[Union[Tensor, Sequence[Tensor]]] = None, batch_idx: Optional[int] = None, optimizer_idx: Optional[int] = None, hiddens: Optional[Tensor] = None, logger=None,) -> Tensor:

        m_logits, v_feats, a_feats, v_logits, a_logits = self(batch['video'], batch['audio'], batch['padding_mask'])
        loss_dict = self.loss_fn(m_logits, v_feats, a_feats, v_logits, a_logits, batch['v_label'], batch['a_label'], batch['c_label'], batch['m_label'])

        # common and multi-class
        preds = torch.argmax(self.softmax(m_logits), dim=1)

        # for item in loss_dict.items():
        #     logger.info(f"train_{item[0]}: {item[1].item()}")

        return {"loss": loss_dict["loss"], "preds": preds.detach(), "targets": batch['m_label'].detach()}

    def validation_step(self, batch: Optional[Union[Tensor, Sequence[Tensor]]] = None, batch_idx: Optional[int] = None, optimizer_idx: Optional[int] = None, hiddens: Optional[Tensor] = None, logger=None) -> Tensor:

        m_logits, v_feats, a_feats, v_logits, a_logits = self(batch['video'], batch['audio'], batch['padding_mask'])
        loss_dict = self.loss_fn(m_logits, v_feats, a_feats, v_logits, a_logits, batch['v_label'], batch['a_label'], batch['c_label'], batch['m_label'])

        # common and multi-class
        preds = torch.argmax(self.softmax(m_logits), dim=1)

        for item in loss_dict.items():
            logger.info(f"val_{item[0]}: {item[1].item()}")

        return {"loss": loss_dict["mm_loss"], "preds": preds.detach(), "targets": batch['m_label'].detach()}

    def test_step(self, batch: Optional[Union[Tensor, Sequence[Tensor]]] = None, batch_idx: Optional[int] = None, optimizer_idx: Optional[int] = None, hiddens: Optional[Tensor] = None) -> Tensor:

        m_logits, v_feats, a_feats, v_logits, a_logits = self(batch['video'], batch['audio'], batch['padding_mask'])
        loss_dict = self.loss_fn(m_logits, v_feats, a_feats, v_logits, a_logits, batch['v_label'], batch['a_label'], batch['c_label'], batch['m_label'])

        # common and multi-class
        preds = torch.argmax(self.softmax(m_logits), dim=1)

        return {"loss": loss_dict["mm_loss"], "preds": preds.detach(), "targets": batch['m_label'].detach()}

    def configure_optimizers(self):

        optimizer = Adam(self.parameters(), lr=self.learning_rate, betas=(0.5, 0.9), weight_decay=self.weight_decay)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optimizer, factor=0.5, patience=3, verbose=True, min_lr=1e-8),
                "monitor": "val_loss"
            }
        }
