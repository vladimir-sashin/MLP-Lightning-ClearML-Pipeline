from typing import Any, Dict, List

import torch
import torch.nn.functional as func
from lightning import LightningModule
from torch import Tensor
from torchmetrics import MeanMetric

from src.config import MLPExperimentConfig
from src.train.metrics import get_metrics
from src.train.model import get_mlp_model


class ClassificationLightningModule(LightningModule):
    def __init__(self, cfg: MLPExperimentConfig, in_features: int, num_classes: int):
        super().__init__()

        self.hyperparameters_cfg = cfg.hyperparameters_config

        self._train_loss = MeanMetric()
        self._valid_loss = MeanMetric()

        metrics = get_metrics(
            num_classes=num_classes,
            num_labels=num_classes,
            task='multiclass',
            average='macro',
        )
        self._valid_metrics = metrics.clone(prefix='valid_')
        self._test_metrics = metrics.clone(prefix='test_')

        self.model = get_mlp_model(
            mlp_cfg=cfg.mlp_model_config,
            in_dim=in_features,
            out_dim=num_classes,
        )

        self.save_hyperparameters()

    def forward(self, data: Tensor) -> Tensor:
        return self.model(data)

    def training_step(self, batch: List[Tensor]) -> Dict[str, Tensor]:  # noqa: WPS210
        features, targets = batch
        logits = self(features)
        loss = func.cross_entropy(logits, targets)
        self._train_loss(loss)
        self.log('step_loss', loss, on_step=True, prog_bar=True, logger=True)
        return {'loss': loss}

    def on_train_epoch_end(self) -> None:
        self.log('mean_train_loss', self._train_loss.compute(), on_step=False, prog_bar=True, on_epoch=True)
        self._train_loss.reset()

    def validation_step(self, batch: List[Tensor], batch_idx: int) -> None:
        images, targets = batch
        logits = self(images)
        self._valid_loss(func.cross_entropy(logits, targets))

        self._valid_metrics(logits, targets)

    def on_validation_epoch_end(self) -> None:
        self.log('mean_valid_loss', self._valid_loss.compute(), on_step=False, prog_bar=True, on_epoch=True)
        self._valid_loss.reset()

        self.log_dict(self._valid_metrics.compute(), prog_bar=True, on_epoch=True)
        self._valid_metrics.reset()

    def test_step(self, batch: List[Tensor], batch_idx: int) -> Tensor:
        images, targets = batch
        logits = self(images)

        preds = torch.argmax(logits, dim=1)
        self._test_metrics(logits, targets)
        return preds

    def on_test_epoch_end(self) -> None:
        self.log_dict(self._test_metrics.compute(), prog_bar=True, on_epoch=True)
        self._test_metrics.reset()

    def configure_optimizers(self) -> Dict[str, Any]:
        # TODO: add lr scheduler.
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.hyperparameters_cfg.lr)
        return {
            'optimizer': optimizer,
        }
