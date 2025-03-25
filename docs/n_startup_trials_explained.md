# N_STARTUP_TRIALS Explained

## What N_STARTUP_TRIALS Actually Does

`N_STARTUP_TRIALS` in your implementation serves two purposes:

1. **Optuna Sampling Control**: In standard Optuna, it controls how many trials use random sampling before switching to the intelligent TPE algorithm. After N_STARTUP_TRIALS, the optimizer becomes smarter about which parameter values to try next.

2. **Intended as Cycle Barrier** (but not fully implemented): The intention seems to be that models should wait at a synchronization point ("cycle barrier") after completing N_STARTUP_TRIALS trials, before proceeding to the next cycle.

## The Key Problem

**Looking at your code in study_manager.py, there's a cycle barrier mechanism, but it's not actually connected to N_STARTUP_TRIALS.** This is why setting N_STARTUP_TRIALS to 10000 doesn't make each model run 10000 trials for the first cycle.

The cycle barrier is set up in `setup_cycle_barrier()` based on the number of active model types, but there's no code that tells models to wait at this barrier after completing N_STARTUP_TRIALS trials.

## Why Your Trial Count Is Wrong in progress.yaml

The trial count in progress.yaml is incorrect because:

1. **Disconnected Implementation**: N_STARTUP_TRIALS isn't actually controlling when models wait at the barrier.

2. **Trial Count Source**: The total trial count comes from `n_trials * len(ACTIVE_MODEL_TYPES)` where `n_trials` is determined by TUNING_TRIALS_PER_CYCLE_min/max settings.

3. **Missed Connection**: There's no code connecting N_STARTUP_TRIALS to either the total trials or the cycle barrier mechanism.

## How to Fix This

To make N_STARTUP_TRIALS actually work as a cycle barrier:

1. **Connect N_STARTUP_TRIALS to the barrier**: Modify `_run_model_optimization` in study_manager.py to check if the number of completed trials has reached N_STARTUP_TRIALS, and only then wait at the barrier.

2. **Fix Trial Count Display**: Set the expected total trials correctly based on what you want:
   - If you want each model to run 10000 trials in the first cycle, explicitly set n_trials=10000 in tune_for_combo()
   - If you want a different count, adjust TUNING_TRIALS_PER_CYCLE_min/max values

## Code Sample to Fix This

Add this to your study_manager.py in the `_run_model_optimization` method, replacing the current barrier code:

```python
# Wait at cycle barrier if we've completed N_STARTUP_TRIALS trials
try:
    if hasattr(self, 'cycle_barrier') and self.cycle_barrier.parties > 1:
        if len(study.trials) >= self.n_startup_trials:
            logger.info(f"Model {model_type} completed {len(study.trials)} trials (>= {self.n_startup_trials}), waiting at cycle barrier")
            self.cycle_barrier.wait()
            logger.info(f"All models reached barrier, continuing to next cycle")
        else:
            logger.info(f"Model {model_type} completed only {len(study.trials)} trials (< {self.n_startup_trials}), not waiting at barrier")
except Exception as e:
    logger.error(f"Error at cycle barrier for {model_type}: {e}")
```