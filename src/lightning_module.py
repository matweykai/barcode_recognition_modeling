import torch
import pytorch_lightning as pl

from src.config import Config
from src.losses import get_losses
from src.metrics import get_metrics
from src.utils import load_object
from src.models import CRNN


class OCRModule(pl.LightningModule):
    def __init__(self, config: Config):
        super().__init__()
        self._config = config

        self._model = CRNN(**self._config.model_kwargs)

        self._losses = get_losses(self._config.losses)

        metrics = get_metrics()
        self._train_metrics = metrics.clone(prefix='train_')
        self._valid_metrics = metrics.clone(prefix='valid_')

        self.save_hyperparameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._model(x)

    def configure_optimizers(self):
        optimizer = load_object(self._config.optimizer)(
            self._model.parameters(),
            **self._config.optimizer_kwargs,
        )
        scheduler = load_object(self._config.scheduler)(optimizer, **self._config.scheduler_kwargs)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': self._config.monitor_metric,
                'interval': 'epoch',
                'frequency': 1,
            },
        }

    def training_step(self, batch, batch_idx):
        images, targets, target_lengths = batch
        log_probs = self(images)
        input_lengths = torch.IntTensor([log_probs.size(0)] * images.size(0))
        loss_value = self._calculate_loss(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            'train_',
        )

        self._train_metrics(log_probs, targets)

        return loss_value

    def validation_step(self, batch, batch_idx):
        images, targets, target_lengths = batch
        log_probs = self(images)
        input_lengths = torch.IntTensor([log_probs.size(0)] * images.size(0))
        self._calculate_loss(
            log_probs,
            targets,
            input_lengths,
            target_lengths,
            'valid_',
        )
        self._valid_metrics(log_probs, targets)

    def on_train_epoch_start(self) -> None:
        self._train_metrics.reset()

    def on_train_epoch_end(self) -> None:
        self.log_dict(self._train_metrics.compute(), on_epoch=True)

    def on_validation_epoch_start(self) -> None:
        self._valid_metrics.reset()

    def on_validation_epoch_end(self) -> None:
        self.log_dict(self._valid_metrics.compute(), on_epoch=True)

    def _calculate_loss(
        self,
        log_probs: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
        prefix: str,
    ) -> torch.Tensor:
        total_loss = torch.tensor(0.0, device=log_probs.device)
        for cur_loss in self._losses:
            loss = cur_loss.loss(
                log_probs=log_probs,
                targets=targets,
                input_lengths=input_lengths,
                target_lengths=target_lengths,
            )
            total_loss += cur_loss.weight * loss
            self.log(f'{prefix}{cur_loss.name}_loss', loss.item())
        self.log(f'{prefix}total_loss', total_loss.item())
        return total_loss
