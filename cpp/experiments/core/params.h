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
    constexpr const char *StepSizeKey = "step";
    constexpr const char *LambdaKey = "lambda";
    constexpr const char *OneVsAllKey = "one_vs_all";
    constexpr const char *NameKey = "name";
    constexpr const char *TrainingLimitKey = "training_limit";
    constexpr const char *TestLimitKey = "test_limit";
}
