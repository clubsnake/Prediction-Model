# Add this to a new file Scripts/model_optimization.py

import time
import os
from datetime import datetime

import numpy as np
import tensorflow as tf

# Import centralized GPU memory management
from config.resource_config import configure_gpu_memory


def apply_mixed_precision():
    """Enable mixed precision training for faster computation on supporting GPUs."""
    # Use centralized configuration
    config = {
        "mixed_precision": True,
        "allow_growth": True,
    }
    result = configure_gpu_memory(config)
    return result.get("success", False)


def optimize_gpu_memory_growth():
    """Configure GPU memory growth to prevent OOM errors."""
    # Use centralized configuration
    config = {
        "allow_growth": True,
    }
    result = configure_gpu_memory(config)
    return result.get("success", False)


def enable_xla_compilation():
    """Enable XLA (Accelerated Linear Algebra) JIT compilation for better performance."""
    # Use centralized configuration
    config = {
        "use_xla": True,
    }
    configure_gpu_memory(config)

    # Verify with a simple test function
    try:
        import tensorflow as tf

        @tf.function(jit_compile=True)
        def test_func(x):
            return tf.reduce_sum(x * x)

        test_func(tf.random.normal([100, 100]))
        return True
    except Exception as e:
        print(f"Could not verify XLA compilation: {e}")
        return False


def ensure_xla_compilation():
    """
    Verify XLA compilation is working and enabled for TensorFlow models.
    Returns True if XLA is operational, False otherwise.
    """
    import tensorflow as tf

    # Check current XLA status
    xla_enabled = tf.config.optimizer.get_jit() is not None
    if not xla_enabled:
        try:
            # Enable XLA JIT compilation
            tf.config.optimizer.set_jit(True)
            print("XLA JIT compilation enabled")
            xla_enabled = True
        except Exception as e:
            print(f"Could not enable XLA: {e}")
            return False

    # Verify XLA works with a simple test
    try:

        @tf.function(jit_compile=True)
        def test_func(x):
            return tf.reduce_sum(tf.square(x))

        test_input = tf.random.normal([1000, 1000])

        # Time execution both with and without XLA
        import time

        # Warm-up run
        _ = test_func(test_input).numpy()

        # XLA-enabled timing
        start = time.time()
        _ = test_func(test_input).numpy()
        xla_time = time.time() - start

        # Regular execution timing
        @tf.function(jit_compile=False)
        def test_func_no_xla(x):
            return tf.reduce_sum(tf.square(x))

        # Warm-up
        _ = test_func_no_xla(test_input).numpy()

        start = time.time()
        _ = test_func_no_xla(test_input).numpy()
        no_xla_time = time.time() - start

        # Check if XLA is actually faster (should be)
        speedup = no_xla_time / xla_time
        print(f"XLA test completed. Speedup: {speedup:.2f}x")

        # Return success if we got here without errors
        return True

    except Exception as e:
        print(f"XLA test failed: {e}")
        return False


def apply_xla_to_model(model):
    """
    Apply XLA compilation to an existing Keras model.

    Args:
        model: A Keras model

    Returns:
        Model with XLA compilation applied
    """
    import tensorflow as tf

    # First check if XLA is available
    if not ensure_xla_compilation():
        print("XLA not available, returning unmodified model")
        return model

    # Convert Keras model to a tf.function with XLA compilation
    try:
        # Extract the call method
        call_fn = model.call

        # Apply XLA compilation
        @tf.function(jit_compile=True)
        def xla_call(*args, **kwargs):
            return call_fn(*args, **kwargs)

        # Replace the call method
        model.call = xla_call

        # Add a marker to indicate XLA compilation
        model.xla_compiled = True

        return model
    except Exception as e:
        print(f"Could not apply XLA to model: {e}")
        return model


def optimize_parallel_execution(num_threads=None):
    """Configure thread parallelism for better CPU performance."""
    if num_threads is None:
        import multiprocessing

        num_threads = multiprocessing.cpu_count()

    try:
        tf.config.threading.set_inter_op_parallelism_threads(num_threads)
        tf.config.threading.set_intra_op_parallelism_threads(num_threads)
        print(f"TensorFlow thread parallelism set to {num_threads} threads")
        return True
    except Exception as e:
        print(f"Could not configure thread parallelism: {e}")
        return False


def quantize_model(model, dataset_gen=None, optimizations=None):
    """
    Apply post-training quantization to reduce model size and improve inference speed.

    Args:
        model: A trained Keras model
        dataset_gen: Generator providing representative data for quantization
                    (needed for full integer quantization)
        optimizations: List of optimization types, defaults to ['DEFAULT']

    Returns:
        Quantized TFLite model
    """
    try:
        pass

        if optimizations is None:
            optimizations = ["DEFAULT"]

        # Convert to TFLite format
        converter = tf.lite.TFLiteConverter.from_keras_model(model)

        # Set optimization flags
        converter.optimizations = optimizations

        # If we have a dataset generator, use it for full integer quantization
        if dataset_gen is not None:

            def representative_dataset():
                for data in dataset_gen:
                    yield [np.array(data, dtype=np.float32)]

            converter.representative_dataset = representative_dataset
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8

        # Convert the model
        quantized_model = converter.convert()

        # Report size reduction
        original_size = model.count_params() * 4  # Assuming float32 (4 bytes)
        quantized_size = len(quantized_model)
        reduction = (1 - quantized_size / original_size) * 100

        print(f"Original model size: {original_size / 1e6:.2f} MB")
        print(f"Quantized model size: {quantized_size / 1e6:.2f} MB")
        print(f"Size reduction: {reduction:.2f}%")

        return quantized_model
    except Exception as e:
        print(f"Quantization failed: {e}")
        return None


def prune_model(model, pruning_schedule=None, epochs=10, validation_data=None):
    """
    Apply weight pruning to reduce model size while maintaining accuracy.

    Args:
        model: A trained Keras model
        pruning_schedule: Custom pruning schedule or None for default
        epochs: Number of fine-tuning epochs
        validation_data: Validation data for fine-tuning

    Returns:
        Pruned model
    """
    try:
        import tensorflow_model_optimization as tfmot

        # Define pruning schedule if not provided
        if pruning_schedule is None:
            pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(
                initial_sparsity=0.0,
                final_sparsity=0.5,  # Remove 50% of weights
                begin_step=0,
                end_step=epochs * 1000,  # Adjust based on your dataset
            )

        # Apply pruning to all layers except BatchNormalization
        def apply_pruning_to_layer(layer):
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                return layer
            return tfmot.sparsity.keras.prune_low_magnitude(
                layer, pruning_schedule=pruning_schedule
            )

        # Create pruned model
        pruned_model = tf.keras.models.clone_model(
            model, clone_function=apply_pruning_to_layer
        )

        # Compile the model (use the same settings as original model)
        pruned_model.compile(
            optimizer=model.optimizer, loss=model.loss, metrics=model.metrics
        )

        # Add pruning callbacks
        callbacks = [
            tfmot.sparsity.keras.UpdatePruningStep(),
            tfmot.sparsity.keras.PruningSummaries(),
        ]

        # Fine-tune if validation data provided
        if validation_data is not None:
            pruned_model.fit(
                validation_data[0],
                validation_data[1],
                epochs=epochs,
                callbacks=callbacks,
                validation_split=0.1,
            )

        # Strip pruning wrappers for inference
        final_model = tfmot.sparsity.keras.strip_pruning(pruned_model)

        return final_model
    except Exception as e:
        print(f"Pruning failed: {e}")
        return model  # Return original model on failure


def optimize_model(
    model, quantize=True, prune=False, dataset_gen=None, validation_data=None
):
    """
    Apply multiple optimizations to a model and save to the proper directory
    
    Args:
        model: Model to optimize
        quantize: Whether to apply quantization
        prune: Whether to apply pruning
        dataset_gen: Dataset generator for quantization
        validation_data: Validation data for pruning fine-tuning
        
    Returns:
        Optimized model
    """
    from config.config_loader import DATA_DIR
    
    # Create Models directory if it doesn't exist
    models_dir = os.path.join(DATA_DIR, "Models", "optimized")
    os.makedirs(models_dir, exist_ok=True)
    
    optimized_model = model
    quantized_tflite = None

    # Step 1: Pruning (if enabled)
    if prune:
        print("Applying weight pruning...")
        pruned_model = prune_model(model, validation_data=validation_data)
        if pruned_model is not None:
            optimized_model = pruned_model
            print("Pruning complete")

    # Step 2: Quantization (if enabled)
    if quantize:
        print("Applying quantization...")
        quantized_tflite = quantize_model(optimized_model, dataset_gen)
        if quantized_tflite is not None:
            print("Quantization complete")

    # Save the optimized model
    if hasattr(model, 'save'):
        model_path = os.path.join(models_dir, f"optimized_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        model.save(model_path)
    
    return optimized_model, quantized_tflite


def apply_performance_optimizations():
    """Apply all TensorFlow performance optimizations at once."""
    # Use centralized configuration with all optimizations enabled
    config = {"allow_growth": True, "mixed_precision": True, "use_xla": True}
    result = configure_gpu_memory(config)

    print(f"Applied performance optimizations: {result}")
    return result


def benchmark_model(
    model, input_shape, warmup_runs=5, benchmark_runs=20, tflite_model=None
):
    """
    Benchmark model inference performance.

    Args:
        model: Keras model to benchmark
        input_shape: Input shape for test data
        warmup_runs: Number of warmup runs
        benchmark_runs: Number of benchmark runs
        tflite_model: Optional TFLite model to benchmark against

    Returns:
        Dictionary with benchmark results
    """
    results = {}

    # Generate random test data
    test_data = np.random.normal(size=input_shape).astype(np.float32)

    # Benchmark Keras model
    if model is not None:
        # Warmup
        for _ in range(warmup_runs):
            _ = model.predict(test_data, verbose=0)

        # Benchmark
        keras_times = []
        for _ in range(benchmark_runs):
            start_time = time.time()
            _ = model.predict(test_data, verbose=0)
            keras_times.append(time.time() - start_time)

        results["keras_avg_time"] = np.mean(keras_times) * 1000  # Convert to ms
        results["keras_std_time"] = np.std(keras_times) * 1000  # Convert to ms
        print(f"Keras model average inference time: {results['keras_avg_time']:.2f} ms")

    # Benchmark TFLite model if provided
    if tflite_model is not None:
        # Create interpreter
        interpreter = tf.lite.Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()

        # Get input and output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Warmup
        for _ in range(warmup_runs):
            interpreter.set_tensor(input_details[0]["index"], test_data)
            interpreter.invoke()
            _ = interpreter.get_tensor(output_details[0]["index"])

        # Benchmark
        tflite_times = []
        for _ in range(benchmark_runs):
            start_time = time.time()
            interpreter.set_tensor(input_details[0]["index"], test_data)
            interpreter.invoke()
            _ = interpreter.get_tensor(output_details[0]["index"])
            tflite_times.append(time.time() - start_time)

        results["tflite_avg_time"] = np.mean(tflite_times) * 1000  # Convert to ms
        results["tflite_std_time"] = np.std(tflite_times) * 1000  # Convert to ms
        print(
            f"TFLite model average inference time: {results['tflite_avg_time']:.2f} ms"
        )

        # Calculate speedup
        if "keras_avg_time" in results:
            speedup = results["keras_avg_time"] / results["tflite_avg_time"]
            results["speedup"] = speedup
            print(f"TFLite speedup: {speedup:.2f}x")

    return results


if __name__ == "__main__":
    # Run all optimizations when the module is executed directly
    apply_performance_optimizations()
