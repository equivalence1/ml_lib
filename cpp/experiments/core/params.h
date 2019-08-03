#pragma once

namespace {
    constexpr const char *ModelKey = "model";
    constexpr const char *ConvKey = "conv";
    constexpr const char *ClassifierKey = "classifier";

    constexpr const char *ModelArchKey = "model_arch";
    constexpr const char *ModelArchVersionKey = "arch_version";
    constexpr const char *ModelCheckpointFileKey = "checkpoint_file";
    constexpr const char *ClassifierMainKey = "main";
    constexpr const char *ClassifierBaselineKey = "baseline";

    constexpr const char *DeviceKey = "device";
    constexpr const char *DatasetKey = "dataset";
    constexpr const char *BatchSizeKey = "batch_size";
    constexpr const char *DimsKey = "dims";
    constexpr const char *ReportsPerEpochKey = "reports_per_epoch";
    constexpr const char *NIterationsKey = "n_iterations";
    constexpr const char *SgdStepSizeKey = "sgd_step";
    constexpr const char *LambdaKey = "lambda";
    constexpr const char *BaseClassesKey = "base_classes";
    constexpr const char *NameKey = "name";
    constexpr const char *TrainingLimitKey = "training_limit";
    constexpr const char *TestLimitKey = "test_limit";
    constexpr const char *ScheduledParamModifiersKey = "scheduled_param_modifiers";
    constexpr const char *FieldKey = "field";
    constexpr const char *ValuesKey = "values";
    constexpr const char *ItersKey = "iters";
    constexpr const char *DropoutKey = "dropout";
    constexpr const char *MonomTypeKey = "monom_type";

    constexpr const char *CatboostParamsKey = "catboost_params";
    constexpr const char *InitParamsKey = "init_params";
    constexpr const char *IntermediateParamsKey = "intermediate_params";
    constexpr const char *FinalParamsKey = "final_params";
}
