"""RunSleuth telemetry for the Hugging Face Trainer.

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train,
        eval_dataset={"id": id_val, "ood": ood_val},
        callbacks=[RunSleuthCallback("runs/vit", head="classifier")],
    )
    trainer.train()

The callback wraps every training epoch in a RunMonitor epoch, so the run report
has the same format as a RunMonitor run and signature_matching can diagnose it.
Evaluation metrics are attached to the epoch they follow, optionally renamed.
Gradient accumulation is rejected: the monitor clears model gradients before each
training forward, which would discard accumulated micro-batch gradients.
"""

import math
from pathlib import Path

from transformers import TrainerCallback

from runsleuth.run_monitor import RunMonitor


class RunSleuthCallback(TrainerCallback):
    """Record RunSleuth evidence for each Trainer epoch and close the report at the end."""

    def __init__(self, run_dir: str | Path, *, head: str = "classifier", rename=None) -> None:
        self.run_dir = Path(run_dir)
        self.head = head
        self.rename = dict(rename or {})
        self.monitor: RunMonitor | None = None
        self._epoch = None

    def on_train_begin(self, args, state, control, model=None, optimizer=None, **kwargs):
        if args.gradient_accumulation_steps != 1:
            raise RuntimeError(
                "RunSleuthCallback does not support gradient accumulation: the monitor clears "
                "gradients before each training forward"
            )
        if model is None or optimizer is None:
            raise RuntimeError("RunSleuthCallback needs the Trainer's model and optimizer")
        # Accelerate wraps the optimizer; hooks and audits need the torch optimizer itself.
        inner = getattr(optimizer, "optimizer", optimizer)
        self.monitor = RunMonitor(model, inner, self.run_dir, head=self.head)
        self.monitor.report["hf_trainer"] = {
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "lr_scheduler_type": str(args.lr_scheduler_type),
            "warmup_steps": args.warmup_steps,
            "max_grad_norm": args.max_grad_norm,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "bf16": args.bf16,
            "seed": args.seed,
        }

    def observe_batch(self, split: str, inputs, labels=None) -> None:
        """Forward a collated batch to the monitor; batches before training starts are ignored."""
        if self.monitor is not None:
            self.monitor.observe_batch(split, inputs, labels)

    def on_epoch_begin(self, args, state, control, **kwargs):
        self._epoch = self.monitor.epoch()
        self._epoch.__enter__()

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch, self._epoch = self._epoch, None
        epoch.__exit__(None, None, None)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if self.monitor is None or not self.monitor.report["parameter_group_epochs"]:
            return
        values = {
            self.rename.get(name, name): float(value)
            for name, value in (metrics or {}).items()
            # The Trainer's fractional "epoch" would overwrite RunSleuth's own epoch index.
            if name != "epoch" and isinstance(value, (int, float)) and math.isfinite(value)
        }
        if values:
            self.monitor.log(**values)

    def on_train_end(self, args, state, control, **kwargs):
        if self._epoch is not None:
            self.on_epoch_end(args, state, control)
        if self.monitor is not None:
            self.monitor.close()
