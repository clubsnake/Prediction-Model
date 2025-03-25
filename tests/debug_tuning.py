import debugpy
import logging
import os
import time
import traceback

# Attach debugger to VSCode
debugpy.listen(("localhost", 5678))
print("🚀 Waiting for debugger to attach... Open VSCode and start debugging.")
debugpy.wait_for_client()

# Set up logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

try:
    # ✅ Step 1: Check if the tuning module loads correctly
    print("🚀 Importing tuning modules...")
    import src.tuning.meta_tuning as meta_tuning
    import src.tuning.study_manager as study_manager
    import src.tuning.progress_helper as progress_helper
    from config.config_loader import get_config  # Ensure this exists

    logger.info("✅ Successfully imported tuning modules.")

    # ✅ Step 2: List all functions available in meta_tuning
    import inspect
    tuning_functions = [name for name, obj in inspect.getmembers(meta_tuning) if inspect.isfunction(obj)]
    print(f"🔍 Available functions in meta_tuning: {tuning_functions}")

    # ✅ Step 3: Find and run the actual tuning function
    tuning_function_name = None
    for func in tuning_functions:
        if "tune" in func or "optuna" in func or "hyperparameter" in func:
            tuning_function_name = func
            break

    if not tuning_function_name:
        raise Exception("❌ No valid tuning function found in meta_tuning.py")

    logger.info(f"✅ Using tuning function: {tuning_function_name}")

    # ✅ Step 4: Ensure study_manager is ready
    study_file = "data/hyperparameter_study.lock"
    if os.path.exists(study_file):
        os.remove(study_file)
        print("🚀 DEBUG: Removed stale study lock file.")

    # ✅ Step 5: Run the tuning function
    print(f"🚀 Starting tuning with `{tuning_function_name}`...")
    tuning_function = getattr(meta_tuning, tuning_function_name)

    # Force debugging inside the function
    debugpy.breakpoint()
    tuning_function()  # Run tuning

    print("✅ Tuning started successfully. Check Optuna logs.")

except Exception as e:
    print(f"❌ ERROR: {e}")
    traceback.print_exc()
    logger.error(f"❌ Exception occurred: {e}")
