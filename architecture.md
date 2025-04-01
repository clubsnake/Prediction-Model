# System Architecture Diagram

```mermaid
graph TD
    subgraph Main ["Main Entry Points"]
        launcher["launcher.py"]
        run_dashboard["run_dashboard.bat"]
        app_launcher["app/launcher.py"]
        tuning_launcher["src/tuning/launcher.py"]
    end

    subgraph Data ["Data Layer"]
        data_fetch["src/data/data.py<br>Data Fetching"]
        data_preprocessing["src/data/preprocessing.py<br>Preprocessing"]
        data_utils["src/data/data_utils.py<br>Data Utilities"]
        data_loader["src/data/data_loader.py<br>Data Loading"]
        data_manager["src/data/data_manager.py<br>Data Management"]
        sequence_utils["src/data/sequence_utils.py<br>Sequence Creation"]
        vectors["src/data/vectorized_indicators.py<br>Vectorized Indicators"]
    end

    subgraph Features ["Feature Engineering"]
        features["src/features/features.py<br>Feature Creation"]
        feature_selection["src/features/feature_selection.py<br>Feature Selection"]
        indicator_tuning["src/features/indicator_tuning.py<br>Indicator Tuning"]
        optimized_params["src/features/optimized_params.py<br>Optimization"]
        optimized_processor["src/features/optimized_processor.py<br>Optimized Processing"]
        vmli_indicator["src/features/vmli_indicator.py<br>VMLI Indicator"]
    end

    subgraph Models ["Model Implementations"]
        model_base["src/models/model.py<br>Base Models"]
        model_factory["src/models/model_factory.py<br>Model Factory"]
        model_manager["src/models/model_manager.py<br>Model Management"]
        
        subgraph Neural ["Neural Network Models"]
            lstm_rnn["LSTM/RNN Models"]
            cnn["src/models/cnn_model.py<br>CNN Model"]
            tft["src/models/temporal_fusion_transformer.py<br>TFT Model"]
            nbeats["src/models/nbeats_model.py<br>N-BEATS Model"]
            tabnet["src/models/tabnet_model.py<br>TabNet Model"]
            ltc["src/models/ltc_model.py<br>LTC Model"]
        end
        
        subgraph Tree ["Tree-based Models"]
            rf["Random Forest"]
            xgb["XGBoost"]
        end
        
        subgraph Ensemble ["Ensemble Models"]
            ensemble_model["src/models/ensemble_model.py<br>Ensemble Model"]
            ensemble_utils["src/models/ensemble_utils.py<br>Ensemble Utilities"]
            ensemble_weighting["src/models/ensemble_weighting.py<br>Ensemble Weighting"]
        end
    end

    subgraph Training ["Training"]
        training_base["src/training/training_base.py<br>Training Base"]
        trainer["src/training/trainer.py<br>Trainer"]
        walk_forward["src/training/walk_forward.py<br>Walk Forward"]
        callbacks["src/training/callbacks.py<br>Callbacks"]
        adaptive_params["src/training/adaptive_params.py<br>Adaptive Parameters"]
    end

    subgraph Drift ["Concept Drift"]
        concept_drift["src/drift/concept_drift.py<br>Concept Drift"]
        drift_scheduler["src/drift/drift_scheduler.py<br>Drift Scheduler"]
        market_regime["src/drift/market_regime.py<br>Market Regime"]
    end

    subgraph Incremental ["Incremental Learning"]
        incremental["src/incremental/incremental.py<br>Incremental Learning"]
        confidence["src/incremental/confidence.py<br>Confidence Metrics"]
    end

    subgraph Tuning ["Hyperparameter Tuning"]
        study_manager["src/tuning/study_manager.py<br>Study Manager"]
        model_evaluation["src/tuning/model_evaluation.py<br>Model Evaluation"]
        tuning_coordinator["src/tuning/tuning_coordinator.py<br>Tuning Coordinator"]
        meta_tuning["src/tuning/meta_tuning.py<br>Meta Tuning"]
        progress_helper["src/tuning/progress_helper.py<br>Progress Helper"]
        drift_optimizer["src/tuning/drift_optimizer.py<br>Drift Optimizer"]
        ensemble_tuning["src/tuning/ensemble_tuning.py<br>Ensemble Tuning"]
        hyperparameter_integration["src/tuning/hyperparameter_integration.py<br>Hyperparameter Integration"]
    end

    subgraph Dashboard ["Dashboard"]
        dashboard_core["src/dashboard/dashboard_core.py<br>Dashboard Core"]
        UI["src/dashboard/UI.py<br>UI"]
        dashboard_ui["src/dashboard/dashboard_ui.py<br>Dashboard UI"]
        dashboard_state["src/dashboard/dashboard_state.py<br>Dashboard State"]
        dashboard_model["src/dashboard/dashboard_model.py<br>Dashboard Model"]
        dashboard_data["src/dashboard/dashboard_data.py<br>Dashboard Data"]
        dashboard_viz["src/dashboard/dashboard_viz.py<br>Dashboard Visualization"]
        dashboard_utils["src/dashboard/dashboard_utils.py<br>Dashboard Utilities"]
        dashboard_error["src/dashboard/dashboard_error.py<br>Dashboard Error"]
        
        subgraph Visualization ["Visualization"]
            visualization["src/visualization/visualization.py<br>Visualization"]
            model_viz["src/visualization/model_viz.py<br>Model Visualization"]
            enhanced_weight_viz["src/visualization/enhanced_weight_viz.py<br>Enhanced Weight Visualization"]
            drift_dashboard["src/visualization/drift_dashboard.py<br>Drift Dashboard"]
            xai_integration["src/visualization/xai_integration.py<br>XAI Integration"]
        end
        
        subgraph PatternDiscovery ["Pattern Discovery"]
            pattern_discovery["src/pattern_discovery/pattern_discovery.py<br>Pattern Discovery"]
            pattern_management["src/pattern_discovery/pattern_management.py<br>Pattern Management"]
        end
        
        prediction_service["src/dashboard/prediction_service.py<br>Prediction Service"]
        reporter["src/dashboard/reporter.py<br>Reporter"]
        monitoring["src/dashboard/monitoring.py<br>Monitoring"]
    end

    subgraph Utilities ["Utilities"]
        utils["src/utils/utils.py<br>Utilities"]
        GPU["src/utils/GPU.py<br>GPU Management"]
        gpu_mem_management["src/utils/gpu_mem_management.py<br>GPU Memory Management"]
        gpu_mem_manager["src/utils/gpu_mem_manager.py<br>GPU Memory Manager"]
        training_optimizer["src/utils/training_optimizer.py<br>Training Optimizer"]
        Memory["src/utils/memory.py<br>Memory Management"]
        memory_utils["src/utils/memory_utils.py<br>Memory Utilities"]
        model_optimization["src/utils/model_optimization.py<br>Model Optimization"]
        ErrorHandling["src/utils/error_handling.py<br>Error Handling"]
        error_handling["src/utils/error_handling.py<br>Error Handling"]
        robust_handler["src/utils/robust_handler.py<br>Robust Handler"]
        env_setup["src/utils/env_setup.py<br>Environment Setup"]
        threadsafe["src/utils/threadsafe.py<br>Thread Safety"]
        vectorized_ops["src/utils/vectorized_ops.py<br>Vectorized Operations"]
        file_cleanup["src/utils/file_cleanup.py<br>File Cleanup"]
        watchdog["src/utils/watchdog.py<br>Watchdog"]
    end

    subgraph Configuration ["Configuration"]
        config_loader["src/config/config_loader.py<br>Config Loader"]
        system_config["src/config/system_config.json<br>System Config"]
        user_config["src/config/user_config.yaml<br>User Config"]
        hyperparameter_config["src/config/hyperparameter_config.py<br>Hyperparameter Config"]
        resource_config["src/config/resource_config.py<br>Resource Config"]
        advanced_loss_config["src/config/advanced_loss_config.py<br>Advanced Loss Config"]
        logger_config["src/config/logger_config.py<br>Logger Config"]
        api_keys["src/config/api_keys.py<br>API Keys"]
    end

    subgraph Testing ["Testing"]
        test_preprocessing["src/testing/test_preprocessing.py<br>Test Preprocessing"]
        test_model_layers["src/testing/test_model_layers.py<br>Test Model Layers"]
        test_tabnet["src/testing/test_tabnet.py<br>Test TabNet"]
        debug_tuning["src/testing/debug_tuning.py<br>Debug Tuning"]
        check_gpu["src/testing/check_gpu.py<br>Check GPU"]
        seed_test["src/testing/seed_test.py<br>Seed Test"]
    end

    classDef main fill:#f4a460,stroke:#333,stroke-width:2px;
    classDef data fill:#87cefa,stroke:#333,stroke-width:1px;
    classDef model fill:#90ee90,stroke:#333,stroke-width:1px;
    classDef train fill:#ff9999,stroke:#333,stroke-width:1px;
    classDef tune fill:#ffb6c1,stroke:#333,stroke-width:1px;
    classDef dash fill:#ffa07a,stroke:#333,stroke-width:1px;
    classDef util fill:#d3d3d3,stroke:#333,stroke-width:1px;
    classDef config fill:#ffe4b5,stroke:#333,stroke-width:1px;
    classDef test fill:#e6e6fa,stroke:#333,stroke-width:1px;
    classDef feature fill:#98fb98,stroke:#333,stroke-width:1px;

    class Main main;
    class Data,data_fetch,data_preprocessing,data_utils,data_loader,data_manager,sequence_utils,vectors data;
    class Features,features,feature_selection,indicator_tuning,optimized_params,optimized_processor,vmli_indicator feature;
    class Models,model_base,model_factory,model_manager,Neural,Tree,Ensemble,lstm_rnn,cnn,tft,nbeats,tabnet,ltc,rf,xgb,ensemble_model,ensemble_utils,ensemble_weighting model;
    class Training,training_base,trainer,walk_forward,callbacks,adaptive_params,Drift,concept_drift,drift_scheduler,market_regime,incremental,confidence train;
    class Tuning,study_manager,model_evaluation,tuning_coordinator,meta_tuning,progress_helper,drift_optimizer,ensemble_tuning,hyperparameter_integration tune;
    class Dashboard,dashboard_core,UI,dashboard_ui,dashboard_state,dashboard_model,dashboard_data,dashboard_viz,dashboard_utils,dashboard_error,Visualization,visualization,model_viz,enhanced_weight_viz,drift_dashboard,xai_integration,PatternDiscovery,pattern_discovery,pattern_management,prediction_service,reporter,monitoring dash;
    class Utilities,utils,GPU,gpu_mem_management,gpu_mem_manager,training_optimizer,Memory,memory_utils,model_optimization,ErrorHandling,error_handling,robust_handler,env_setup,threadsafe,vectorized_ops,file_cleanup,watchdog util;
    class Configuration,config_loader,system_config,user_config,hyperparameter_config,resource_config,advanced_loss_config,logger_config,api_keys config;
    class Testing,test_preprocessing,test_model_layers,test_tabnet,debug_tuning,check_gpu,seed_test test;