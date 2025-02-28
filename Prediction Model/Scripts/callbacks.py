# callbacks.py
"""
Custom TensorFlow Keras callbacks for dynamic learning rate, dropout, and optimizers.
"""

import tensorflow as tf
import logging

logger = logging.getLogger(__name__)

class DynamicLearningRateScheduler(tf.keras.callbacks.Callback):
    """
    Adjusts learning rate dynamically after each epoch, based on ratio of first_loss to current_loss.
    """
    def __init__(self, original_learning_rate, min_lr=1e-6, max_lr=0.01, adjustment_factor=0.1):
        super(DynamicLearningRateScheduler, self).__init__()
        self.original_learning_rate = original_learning_rate
        self.first_loss = None
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.adjustment_factor = adjustment_factor

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current_loss = logs.get("val_loss") or logs.get("loss")
        if current_loss is None:
            return
        if self.first_loss is None:
            self.first_loss = current_loss
            return
        
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
    def __init__(self, min_rate=0.1, max_rate=0.5, step=0.01, 
                 improvement_threshold=0.001, step_adjust_factor=0.9,
                 max_step=0.05, min_step=0.001):
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
                layer.rate = new_rate
        self.best_loss = min(self.best_loss, current_loss)

class AdaptiveOptimizer(tf.keras.callbacks.Callback):
    """
    Dynamically adjusts the optimizer's momentum or beta1 parameter based on changes in loss.
    """
    def __init__(self, optimizer, momentum_increase=1.05, momentum_decrease=0.95, min_momentum=0.85, max_momentum=0.99):
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
                new_m = min(self.optimizer.momentum * self.momentum_increase, self.max_momentum)
                tf.keras.backend.set_value(self.optimizer.momentum, new_m)
            if hasattr(self.optimizer, "beta_1"):
                new_b1 = min(self.optimizer.beta_1 * self.momentum_increase, self.max_momentum)
                tf.keras.backend.set_value(self.optimizer.beta_1, new_b1)
        else:
            if hasattr(self.optimizer, "momentum"):
                new_m = max(self.optimizer.momentum * self.momentum_decrease, self.min_momentum)
                tf.keras.backend.set_value(self.optimizer.momentum, new_m)
            if hasattr(self.optimizer, "beta_1"):
                new_b1 = max(self.optimizer.beta_1 * self.momentum_decrease, self.min_momentum)
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
                    new_lr = max(self.model.optimizer.lr.numpy() * self.factor, self.min_lr)
                    print(f"\nReducing learning rate to {new_lr:.8f}")
                    self.model.optimizer.lr.assign(new_lr)
                    self.wait = 0
            else:
                self.wait = 0
            self.previous_loss = current_loss
