# callbacks.py
"""
Custom TensorFlow Keras callbacks for dynamic learning rate, dropout, and optimizers.
"""

import datetime
import logging

import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


class DynamicLearningRateScheduler(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        try:
            logs = logs or {}
            current_loss = logs.get("val_loss") or logs.get("loss")
            if current_loss is None:
                return

            # Additional safety check for NaN/Inf values
            if np.isnan(current_loss) or np.isinf(current_loss):
                logger.warning(
                    f"Skipping LR adjustment due to invalid loss: {current_loss}"
                )
                return

            # Rest of implementation...

        except Exception as e:
            logger.error(f"Error in learning rate scheduler: {e}")
            # Don't raise - let training continue with current learning rate

        loss_ratio = self.first_loss / max(current_loss, 1e-8)
        new_lr = self.original_learning_rate * loss_ratio
        new_lr = max(self.min_lr, min(self.max_lr, new_lr))
        tf.keras.backend.set_value(self.model.optimizer.learning_rate, new_lr)
        logger.info(f"Epoch {epoch+1}: Adjusted learning rate to {new_lr:.6f}")


class AdaptiveDropout(tf.keras.callbacks.Callback):
    """
    Dynamically adjusts the dropout rate based on training/validation loss,
    also modifies step size based on improvement threshold.
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
        super(AdaptiveDropout, self).__init__()
        self.best_loss = float("inf")
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.step = step
        self.improvement_threshold = improvement_threshold
        self.step_adjust_factor = step_adjust_factor
        self.max_step = max_step
        self.min_step = min_step
        self.last_loss = None

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current_loss = logs.get("val_loss") or logs.get("loss")
        if current_loss is None:
            return
        if self.last_loss is not None:
            improvement = self.last_loss - current_loss
            rel_improvement = improvement / self.last_loss if self.last_loss != 0 else 0
            if rel_improvement < self.improvement_threshold:
                self.step = max(self.step * self.step_adjust_factor, self.min_step)
            else:
                self.step = min(self.step / self.step_adjust_factor, self.max_step)
        self.last_loss = current_loss

        for layer in self.model.layers:
            if isinstance(layer, tf.keras.layers.Dropout):
                if current_loss < self.best_loss:
                    new_rate = max(layer.rate - self.step, self.min_rate)
                else:
                    new_rate = min(layer.rate + self.step, self.max_rate)
                layer._rate.assign(new_rate)
        self.best_loss = min(self.best_loss, current_loss)


class AdaptiveOptimizer(tf.keras.callbacks.Callback):
    """
    Dynamically adjusts the optimizer's momentum or beta1 parameter based on changes in loss.
    """

    def __init__(
        self,
        optimizer,
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

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current_loss = logs.get("val_loss") or logs.get("loss")
        if current_loss is None:
            return
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            if hasattr(self.optimizer, "momentum"):
                new_m = min(
                    self.optimizer.momentum * self.momentum_increase, self.max_momentum
                )
                tf.keras.backend.set_value(self.optimizer.momentum, new_m)
            if hasattr(self.optimizer, "beta_1"):
                new_b1 = min(
                    self.optimizer.beta_1 * self.momentum_increase, self.max_momentum
                )
                tf.keras.backend.set_value(self.optimizer.beta_1, new_b1)
        else:
            if hasattr(self.optimizer, "momentum"):
                new_m = max(
                    self.optimizer.momentum * self.momentum_decrease, self.min_momentum
                )
                tf.keras.backend.set_value(self.optimizer.momentum, new_m)
            if hasattr(self.optimizer, "beta_1"):
                new_b1 = max(
                    self.optimizer.beta_1 * self.momentum_decrease, self.min_momentum
                )
                tf.keras.backend.set_value(self.optimizer.beta_1, new_b1)


class CustomLearningRateScheduler(tf.keras.callbacks.Callback):
    """
    Simple callback to reduce LR by 'factor' after 'patience' epochs of no improvement.
    """

    def __init__(self, initial_lr, patience=5, factor=0.5, min_lr=1e-6):
        super().__init__()
        self.initial_lr = initial_lr
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.wait = 0
        self.previous_loss = float("inf")

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get("val_loss") or logs.get("loss")
        if current_loss is not None:
            if current_loss > self.previous_loss:
                self.wait += 1
                if self.wait >= self.patience:
                    new_lr = max(
                        self.model.optimizer.lr.numpy() * self.factor, self.min_lr
                    )
                    print(f"\nReducing learning rate to {new_lr:.8f}")
                    self.model.optimizer.lr.assign(new_lr)
                    self.wait = 0
            else:
                self.wait = 0
            self.previous_loss = current_loss


class WarmupCosineDecayScheduler(tf.keras.callbacks.Callback):
    """
    Learning rate scheduler with warmup phase followed by cosine decay.

    This implements:
    1. Linear warmup from `start_lr` to `max_lr` over `warmup_epochs`
    2. Cosine decay from `max_lr` to `min_lr` over remaining epochs

    This helps with better convergence by:
    - Gradually increasing LR to avoid unstable gradients at start
    - Smoothly decaying LR to find better local minima
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

        tf.keras.backend.set_value(self.model.optimizer.learning_rate, lr)
        self.history.append(lr)
        if self.verbose > 0:
            print(f"\nEpoch {epoch+1}: WarmupCosineDecay set learning rate to {lr:.6f}")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs["lr"] = tf.keras.backend.get_value(self.model.optimizer.learning_rate)


class CyclicalLearningRate(tf.keras.callbacks.Callback):
    """
    Cyclical Learning Rate (CLR) implementation.

    Cycles between `base_lr` and `max_lr` with specified policy.
    Implements the techniques in the paper: "Cyclical Learning Rates for Training
    Neural Networks" by Leslie N. Smith (2017).

    Supports triangular, triangular2, and exp_range policies.
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
        tf.keras.backend.set_value(self.model.optimizer.learning_rate, lr)
        self.lrs.append(lr)
        self.batch_idx += 1

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        logs["lr"] = tf.keras.backend.get_value(self.model.optimizer.learning_rate)


class AdaptiveBatchNormScheduler(tf.keras.callbacks.Callback):
    """
    Adaptively adjusts batch normalization momentum during training.

    This helps with better generalization by:
    - Using higher momentum at the beginning to learn faster
    - Decreasing momentum as training progresses to fine-tune more precisely
    """

    def __init__(self, start_momentum=0.99, end_momentum=0.9, decay_epochs=50):
        super(AdaptiveBatchNormScheduler, self).__init__()
        self.start_momentum = start_momentum
        self.end_momentum = end_momentum
        self.decay_epochs = decay_epochs

    def on_epoch_begin(self, epoch, logs=None):
        progress = min(1.0, epoch / self.decay_epochs)
        new_momentum = self.start_momentum - progress * (
            self.start_momentum - self.end_momentum
        )
        for layer in self.model.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.momentum = new_momentum


class AdaptiveWeightDecay(tf.keras.callbacks.Callback):
    """
    Adaptively adjusts weight decay (L2 regularization) based on validation loss trend.

    Increases regularization when validation loss stagnates to prevent overfitting.
    Decreases regularization when in the rapid learning phase.
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


class EarlyStoppingWithRestore(tf.keras.callbacks.Callback):
    """
    Early stopping with best model restoration.

    Stops training when monitored metric stagnates and restores best model weights.
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

    def on_train_begin(self, logs=None):
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
                self.best_weights = self.model.get_weights()
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
                    self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print(f"\nEarly stopping occurred at epoch {self.stopped_epoch + 1}")
            if self.restore_best_weights and self.best_weights is not None:
                print(
                    f"Model weights restored to best found at epoch {self.best_epoch + 1}"
                )


# Example of how to use these callbacks in a TensorFlow model
def get_advanced_callbacks(
    learning_rate=0.001, epochs=50, batch_size=32, steps_per_epoch=100
):
    """
    Create a list of advanced callbacks for TensorFlow models.

    Args:
        learning_rate: Base learning rate
        epochs: Total number of epochs
        batch_size: Batch size
        steps_per_epoch: Steps per epoch

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

    try:
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir, histogram_freq=1, update_freq="epoch"
        )
        callbacks.append(tensorboard_callback)
    except Exception as e:
        print(f"TensorBoard callback not added: {e}")

    return callbacks
