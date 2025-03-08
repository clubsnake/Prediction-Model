# callbacks.py
"""
Custom TensorFlow Keras callbacks for dynamic learning rate, dropout, and optimizers.
Enhanced to work with PyTorch models through adapter layer.
"""

import datetime
import logging
import numpy as np
import tensorflow as tf
import torch

logger = logging.getLogger(__name__)


class DynamicLearningRateScheduler(tf.keras.callbacks.Callback):
    """
    Dynamically adjusts learning rate when monitored metric stops improving.
    Works with both TensorFlow and PyTorch models.
    """
    def __init__(self, initial_lr=0.001, factor=0.5, patience=5, min_lr=1e-6, monitor='val_loss', verbose=0):
        super().__init__()
        self.initial_lr = initial_lr
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.monitor = monitor
        self.verbose = verbose
        self.best = float('inf')
        self.wait = 0
        self.history = []  # Track LR changes
        
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor, None)
        
        if current is None:
            return
            
        if current < self.best:
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                old_lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
                if old_lr > self.min_lr:
                    new_lr = old_lr * self.factor
                    new_lr = max(new_lr, self.min_lr)
                    tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
                    self.history.append(new_lr)
                    if self.verbose > 0:
                        logger.info(f"Epoch {epoch+1}: reducing learning rate from {old_lr} to {new_lr}")
                    self.wait = 0


class AdaptiveDropout(tf.keras.callbacks.Callback):
    """
    Dynamically adjusts the dropout rate based on training/validation loss,
    also modifies step size based on improvement threshold.
    Compatible with PyTorch models through attribute references.
    """

    def __init__(
        self,
        min_rate=0.1,
        max_rate=0.5,
        step=0.01,
        improvement_threshold=0.001,
        step_adjust_factor=0.9,
        max_step=0.05,
        min_step=0.001,
    ):
        super().__init__()
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.step = step
        self.improvement_threshold = improvement_threshold
        self.step_adjust_factor = step_adjust_factor
        self.max_step = max_step
        self.min_step = min_step
        self.prev_loss = float('inf')
        self.current_dropout = min_rate
        self.pytorch_model = None
        self.dropout_changes = []

    def on_train_begin(self, logs=None):
        # Check if model is PyTorch
        if hasattr(self.model, 'model_') and isinstance(self.model.model_, torch.nn.Module):
            self.pytorch_model = self.model.model_

    def on_epoch_end(self, epoch, logs=None):
        if logs is None or 'val_loss' not in logs:
            return
            
        current_loss = logs['val_loss']
        loss_diff = self.prev_loss - current_loss
        
        # Modify dropout rate based on loss change
        if loss_diff < self.improvement_threshold:
            # Loss is not improving enough, increase dropout for regularization
            self.current_dropout = min(self.current_dropout + self.step, self.max_rate)
        else:
            # Loss is improving, decrease dropout to let model learn more
            self.current_dropout = max(self.current_dropout - self.step, self.min_rate)
        
        # Apply to TensorFlow model
        if self.pytorch_model is None:
            for layer in self.model.layers:
                if isinstance(layer, tf.keras.layers.Dropout):
                    layer._rate.assign(self.current_dropout)
        # Apply to PyTorch model
        else:
            for module in self.pytorch_model.modules():
                if isinstance(module, torch.nn.Dropout):
                    module.p = self.current_dropout
        
        self.dropout_changes.append((epoch, self.current_dropout))
        # Update previous loss
        self.prev_loss = current_loss


class AdaptiveOptimizer(tf.keras.callbacks.Callback):
    """
    Dynamically adjusts the optimizer's momentum or beta1 parameter based on changes in loss.
    Enhanced to work with PyTorch optimizers as well.
    """

    def __init__(
        self,
        optimizer=None,
        momentum_increase=1.05,
        momentum_decrease=0.95,
        min_momentum=0.85,
        max_momentum=0.99,
    ):
        super(AdaptiveOptimizer, self).__init__()
        self.optimizer = optimizer
        self.best_loss = float("inf")
        self.momentum_increase = momentum_increase
        self.momentum_decrease = momentum_decrease
        self.min_momentum = min_momentum
        self.max_momentum = max_momentum
        self.pytorch_optimizer = None
        self.momentum_changes = []

    def on_train_begin(self, logs=None):
        # Check if we're using PyTorch
        if hasattr(self.model, 'optimizer') and isinstance(self.model.optimizer, torch.optim.Optimizer):
            self.pytorch_optimizer = self.model.optimizer
            # Set optimizer if not already set
            if self.optimizer is None:
                self.optimizer = self.pytorch_optimizer

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current_loss = logs.get("val_loss") or logs.get("loss")
        if current_loss is None:
            return
        
        is_improvement = current_loss < self.best_loss
        if is_improvement:
            self.best_loss = current_loss
            momentum_factor = self.momentum_increase
        else:
            momentum_factor = self.momentum_decrease
            
        # Handle TensorFlow optimizer
        if self.pytorch_optimizer is None and self.optimizer:
            if hasattr(self.optimizer, "momentum"):
                current_momentum = float(self.optimizer.momentum)
                new_m = min(max(current_momentum * momentum_factor, self.min_momentum), self.max_momentum)
                tf.keras.backend.set_value(self.optimizer.momentum, new_m)
                self.momentum_changes.append((epoch, 'momentum', new_m))
            elif hasattr(self.optimizer, "beta_1"):
                current_beta1 = float(self.optimizer.beta_1)
                new_b1 = min(max(current_beta1 * momentum_factor, self.min_momentum), self.max_momentum)
                tf.keras.backend.set_value(self.optimizer.beta_1, new_b1)
                self.momentum_changes.append((epoch, 'beta_1', new_b1))
        
        # Handle PyTorch optimizer
        elif self.pytorch_optimizer:
            for param_group in self.pytorch_optimizer.param_groups:
                if 'momentum' in param_group:
                    current_momentum = param_group['momentum']
                    new_m = min(max(current_momentum * momentum_factor, self.min_momentum), self.max_momentum)
                    param_group['momentum'] = new_m
                    self.momentum_changes.append((epoch, 'momentum', new_m))
                elif 'betas' in param_group:
                    current_beta1 = param_group['betas'][0]
                    new_b1 = min(max(current_beta1 * momentum_factor, self.min_momentum), self.max_momentum)
                    param_group['betas'] = (new_b1, param_group['betas'][1])
                    self.momentum_changes.append((epoch, 'beta_1', new_b1))


class WarmupCosineDecayScheduler(tf.keras.callbacks.Callback):
    """
    Learning rate scheduler with warmup phase followed by cosine decay.
    Enhanced to work with PyTorch models as well.

    This implements:
    1. Linear warmup from `start_lr` to `max_lr` over `warmup_epochs`
    2. Cosine decay from `max_lr` to `min_lr` over remaining epochs
    """

    def __init__(
        self,
        start_lr=1e-6,
        max_lr=1e-3,
        min_lr=1e-6,
        warmup_epochs=5,
        total_epochs=50,
        verbose=0,
    ):
        super(WarmupCosineDecayScheduler, self).__init__()
        self.start_lr = start_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.verbose = verbose
        self.history = []
        self.pytorch_optimizer = None

    def on_train_begin(self, logs=None):
        # Check if model is PyTorch
        if hasattr(self.model, 'optimizer') and isinstance(self.model.optimizer, torch.optim.Optimizer):
            self.pytorch_optimizer = self.model.optimizer

    def on_epoch_begin(self, epoch, logs=None):
        # Warmup phase
        if epoch < self.warmup_epochs:
            lr = self.start_lr + (
                (self.max_lr - self.start_lr) * (epoch / max(1, self.warmup_epochs))
            )
        else:
            # Cosine decay phase
            decay_epochs = self.total_epochs - self.warmup_epochs
            if decay_epochs <= 0:
                lr = self.max_lr
            else:
                progress = (epoch - self.warmup_epochs) / decay_epochs
                cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
                lr = self.min_lr + cosine_decay * (self.max_lr - self.min_lr)

        # Apply to TensorFlow optimizer
        if self.pytorch_optimizer is None:
            tf.keras.backend.set_value(self.model.optimizer.learning_rate, lr)
        # Apply to PyTorch optimizer
        else:
            for param_group in self.pytorch_optimizer.param_groups:
                param_group['lr'] = lr
        
        self.history.append(lr)
        if self.verbose > 0:
            print(f"\nEpoch {epoch+1}: WarmupCosineDecay set learning rate to {lr:.6f}")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if self.pytorch_optimizer is None:
            logs["lr"] = tf.keras.backend.get_value(self.model.optimizer.learning_rate)
        else:
            logs["lr"] = self.pytorch_optimizer.param_groups[0]['lr']


class CyclicalLearningRate(tf.keras.callbacks.Callback):
    """
    Cyclical Learning Rate (CLR) implementation with PyTorch support.

    Cycles between `base_lr` and `max_lr` with specified policy.
    Implements the techniques in the paper: "Cyclical Learning Rates for Training
    Neural Networks" by Leslie N. Smith (2017).
    """

    def __init__(
        self,
        base_lr=1e-4,
        max_lr=1e-2,
        step_size=2000,
        mode="triangular",
        gamma=1.0,
        scale_fn=None,
        scale_mode="cycle",
    ):
        super(CyclicalLearningRate, self).__init__()

        if mode not in ["triangular", "triangular2", "exp_range"]:
            raise ValueError(
                f"mode must be one of 'triangular', 'triangular2', or 'exp_range', got {mode}"
            )

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        self.pytorch_optimizer = None

        if scale_fn is None:
            if self.mode == "triangular":
                self.scale_fn = lambda x: 1.0
                self.scale_mode = "cycle"
            elif self.mode == "triangular2":
                self.scale_fn = lambda x: 1 / (2.0 ** (x - 1))
                self.scale_mode = "cycle"
            elif self.mode == "exp_range":
                self.scale_fn = lambda x: gamma**x
                self.scale_mode = "iterations"
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode

        self.batch_idx = 0
        self.cycle_count = 0
        self.lrs = []

    def on_train_begin(self, logs=None):
        # Check if model is PyTorch
        if hasattr(self.model, 'optimizer') and isinstance(self.model.optimizer, torch.optim.Optimizer):
            self.pytorch_optimizer = self.model.optimizer

    def clr(self):
        cycle = np.floor(1 + self.batch_idx / (2 * self.step_size))
        x = np.abs(self.batch_idx / self.step_size - 2 * cycle + 1)
        self.cycle_count = cycle
        if self.scale_mode == "cycle":
            scale_factor = self.scale_fn(cycle)
        else:
            scale_factor = self.scale_fn(self.batch_idx)
        lr = (
            self.base_lr
            + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * scale_factor
        )
        return lr

    def on_batch_begin(self, batch, logs=None):
        lr = self.clr()
        # Apply to TensorFlow optimizer
        if self.pytorch_optimizer is None:
            tf.keras.backend.set_value(self.model.optimizer.learning_rate, lr)
        # Apply to PyTorch optimizer
        else:
            for param_group in self.pytorch_optimizer.param_groups:
                param_group['lr'] = lr
                
        self.lrs.append(lr)
        self.batch_idx += 1

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        if self.pytorch_optimizer is None:
            logs["lr"] = tf.keras.backend.get_value(self.model.optimizer.learning_rate)
        else:
            logs["lr"] = self.pytorch_optimizer.param_groups[0]['lr']


class AdaptiveBatchNormScheduler(tf.keras.callbacks.Callback):
    """
    Adaptively adjusts batch normalization momentum during training.
    Compatible with both TensorFlow and PyTorch models.
    """

    def __init__(self, start_momentum=0.99, end_momentum=0.9, decay_epochs=50):
        super(AdaptiveBatchNormScheduler, self).__init__()
        self.start_momentum = start_momentum
        self.end_momentum = end_momentum
        self.decay_epochs = decay_epochs
        self.pytorch_model = None
        self.momentum_history = []

    def on_train_begin(self, logs=None):
        # Check if model is PyTorch
        if hasattr(self.model, 'model_') and isinstance(self.model.model_, torch.nn.Module):
            self.pytorch_model = self.model.model_

    def on_epoch_begin(self, epoch, logs=None):
        progress = min(1.0, epoch / self.decay_epochs)
        new_momentum = self.start_momentum - progress * (
            self.start_momentum - self.end_momentum
        )
        self.momentum_history.append((epoch, new_momentum))
        
        # Apply to TensorFlow model
        if self.pytorch_model is None:
            for layer in self.model.layers:
                if isinstance(layer, tf.keras.layers.BatchNormalization):
                    layer.momentum = new_momentum
        # Apply to PyTorch model
        else:
            for module in self.pytorch_model.modules():
                if isinstance(module, torch.nn.BatchNorm1d) or \
                   isinstance(module, torch.nn.BatchNorm2d) or \
                   isinstance(module, torch.nn.BatchNorm3d):
                    module.momentum = 1 - new_momentum  # PyTorch uses opposite convention


class AdaptiveWeightDecay(tf.keras.callbacks.Callback):
    """
    Adaptively adjusts weight decay (L2 regularization) based on validation loss trend.
    Compatible with both TensorFlow and PyTorch optimizers.
    """

    def __init__(
        self,
        initial_decay=1e-5,
        min_decay=1e-7,
        max_decay=1e-3,
        patience=3,
        factor=2.0,
        monitor="val_loss",
    ):
        super(AdaptiveWeightDecay, self).__init__()
        self.initial_decay = initial_decay
        self.current_decay = initial_decay
        self.min_decay = min_decay
        self.max_decay = max_decay
        self.patience = patience
        self.factor = factor
        self.monitor = monitor
        self.best_loss = float("inf")
        self.wait = 0
        self.history = []
        self.pytorch_optimizer = None

    def on_train_begin(self, logs=None):
        # Check if model is PyTorch
        if hasattr(self.model, 'optimizer') and isinstance(self.model.optimizer, torch.optim.Optimizer):
            self.pytorch_optimizer = self.model.optimizer

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current_loss = logs.get(self.monitor)
        if current_loss is None:
            return
            
        self.history.append((epoch, current_loss, self.current_decay))
        if current_loss < self.best_loss * 0.995:
            self.current_decay = max(self.min_decay, self.current_decay / self.factor)
            self.best_loss = current_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.current_decay = min(
                    self.max_decay, self.current_decay * self.factor
                )
                self.wait = 0
                
        self._update_weight_decay()

    def _update_weight_decay(self):
        # Update TensorFlow optimizer
        if self.pytorch_optimizer is None:
            if hasattr(self.model.optimizer, "weight_decay"):
                if isinstance(self.model.optimizer.weight_decay, tf.Variable):
                    tf.keras.backend.set_value(
                        self.model.optimizer.weight_decay, self.current_decay
                    )
                else:
                    self.model.optimizer.weight_decay = self.current_decay
            elif hasattr(self.model.optimizer, "l2"):
                try:
                    tf.keras.backend.set_value(self.model.optimizer.l2, self.current_decay)
                except:
                    pass
        # Update PyTorch optimizer
        elif self.pytorch_optimizer:
            for param_group in self.pytorch_optimizer.param_groups:
                param_group['weight_decay'] = self.current_decay


class EarlyStoppingWithRestore(tf.keras.callbacks.Callback):
    """
    Early stopping with best model restoration.
    Compatible with both TensorFlow and PyTorch models.
    """

    def __init__(
        self,
        monitor="val_loss",
        min_delta=0,
        patience=10,
        verbose=1,
        mode="auto",
        baseline=None,
        restore_best_weights=True,
    ):
        super(EarlyStoppingWithRestore, self).__init__()
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose
        self.baseline = baseline
        self.restore_best_weights = restore_best_weights

        if mode == "auto":
            if "acc" in self.monitor or self.monitor.endswith("accuracy"):
                self.mode = "max"
            else:
                self.mode = "min"
        else:
            self.mode = mode

        if self.mode == "min":
            self.monitor_op = lambda a, b: a < b - self.min_delta
            self.best = float("inf")
        else:
            self.monitor_op = lambda a, b: a > b + self.min_delta
            self.best = float("-inf")

        self.wait = 0
        self.best_weights = None
        self.best_epoch = 0
        self.stopped_epoch = 0
        self.pytorch_model = None

    def on_train_begin(self, logs=None):
        # Check if model is PyTorch
        if hasattr(self.model, 'model_') and isinstance(self.model.model_, torch.nn.Module):
            self.pytorch_model = self.model.model_
            
        self.wait = 0
        self.best_weights = None
        self.best_epoch = 0
        if self.baseline is not None:
            self.best = self.baseline
        elif self.mode == "min":
            self.best = float("inf")
        else:
            self.best = float("-inf")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)
        if current is None:
            return
            
        if self.monitor_op(current, self.best):
            self.best = current
            self.wait = 0
            self.best_epoch = epoch
            if self.restore_best_weights:
                if self.pytorch_model is None:
                    self.best_weights = self.model.get_weights()
                else:
                    self.best_weights = {
                        k: v.cpu().detach().clone() 
                        for k, v in self.pytorch_model.state_dict().items()
                    }
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                if self.verbose > 0:
                    print(
                        f"\nEpoch {epoch + 1}: early stopping. Best epoch was {self.best_epoch + 1} with {self.monitor} = {self.best:.4f}"
                    )
                if self.restore_best_weights and self.best_weights is not None:
                    if self.verbose > 0:
                        print(
                            f"Restoring model weights from epoch {self.best_epoch + 1}"
                        )
                    if self.pytorch_model is None:
                        self.model.set_weights(self.best_weights)
                    else:
                        self.pytorch_model.load_state_dict(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print(f"\nEarly stopping occurred at epoch {self.stopped_epoch + 1}")
            if self.restore_best_weights and self.best_weights is not None:
                print(
                    f"Model weights restored to best found at epoch {self.best_epoch + 1}"
                )


class PyTorchLearningRateMonitor(tf.keras.callbacks.Callback):
    """
    Monitors the current learning rate for PyTorch optimizers.
    Useful for integrating with TensorBoard.
    """
    
    def __init__(self, verbose=0):
        super(PyTorchLearningRateMonitor, self).__init__()
        self.verbose = verbose
        self.pytorch_optimizer = None
        self.lr_history = []
        
    def on_train_begin(self, logs=None):
        if hasattr(self.model, 'optimizer') and isinstance(self.model.optimizer, torch.optim.Optimizer):
            self.pytorch_optimizer = self.model.optimizer
            
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if self.pytorch_optimizer:
            current_lr = self.pytorch_optimizer.param_groups[0]['lr']
            logs['lr'] = current_lr
            self.lr_history.append((epoch, current_lr))
            if self.verbose > 0:
                print(f"\nEpoch {epoch + 1}: Current learning rate is {current_lr:.6f}")


class TensorBoardCallbackWrapper(tf.keras.callbacks.Callback):
    """
    Wrapper for TensorBoard callback to work with PyTorch models.
    """
    
    def __init__(self, log_dir=None):
        super(TensorBoardCallbackWrapper, self).__init__()
        if log_dir is None:
            log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.log_dir = log_dir
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=self.log_dir)
            self.supports_tensorboard = True
        except ImportError:
            self.supports_tensorboard = False
            print("PyTorch SummaryWriter not available. TensorBoard logging disabled.")
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if self.supports_tensorboard:
            for name, value in logs.items():
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(name, value, epoch)
    
    def on_train_end(self, logs=None):
        if self.supports_tensorboard:
            self.writer.close()


def get_advanced_callbacks(
    learning_rate=0.001, epochs=50, batch_size=32, steps_per_epoch=100, pytorch_compatible=True
):
    """
    Create a list of advanced callbacks for both TensorFlow and PyTorch models.

    Args:
        learning_rate: Base learning rate
        epochs: Total number of epochs
        batch_size: Batch size
        steps_per_epoch: Steps per epoch
        pytorch_compatible: Whether to use PyTorch-compatible callbacks

    Returns:
        List of callbacks
    """
    callbacks = []

    # Learning rate scheduling (choose one)
    use_warmup_cosine = True
    use_cyclical = False

    if use_warmup_cosine:
        lr_scheduler = WarmupCosineDecayScheduler(
            start_lr=learning_rate * 0.1,
            max_lr=learning_rate,
            min_lr=learning_rate * 0.01,
            warmup_epochs=int(epochs * 0.1),
            total_epochs=epochs,
            verbose=1,
        )
        callbacks.append(lr_scheduler)
    elif use_cyclical:
        clr = CyclicalLearningRate(
            base_lr=learning_rate * 0.1,
            max_lr=learning_rate,
            step_size=2 * steps_per_epoch,
            mode="triangular2",
        )
        callbacks.append(clr)

    bn_scheduler = AdaptiveBatchNormScheduler(
        start_momentum=0.99, end_momentum=0.9, decay_epochs=epochs
    )
    callbacks.append(bn_scheduler)

    weight_decay = AdaptiveWeightDecay(
        initial_decay=1e-5,
        min_decay=1e-7,
        max_decay=1e-3,
        patience=3,
        factor=2.0,
        monitor="val_loss",
    )
    callbacks.append(weight_decay)

    early_stopping = EarlyStoppingWithRestore(
        monitor="val_loss",
        min_delta=0.0001,
        patience=10,
        verbose=1,
        mode="min",
        restore_best_weights=True,
    )
    callbacks.append(early_stopping)

    # Add PyTorch-specific monitoring if needed
    if pytorch_compatible:
        lr_monitor = PyTorchLearningRateMonitor(verbose=1)
        callbacks.append(lr_monitor)

    # Try to add TensorBoard support
    try:
        if pytorch_compatible:
            tensorboard_callback = TensorBoardCallbackWrapper()
        else:
            log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            tensorboard_callback = tf.keras.callbacks.TensorBoard(
                log_dir=log_dir, histogram_freq=1, update_freq="epoch"
            )
        callbacks.append(tensorboard_callback)
    except Exception as e:
        print(f"TensorBoard callback not added: {e}")

    return callbacks