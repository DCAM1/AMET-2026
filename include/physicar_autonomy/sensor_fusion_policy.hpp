// Copyright 2026 AICASTLE Inc.
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cmath>

namespace physicar_autonomy
{

enum class SensorFusionStatus
{
  kOk,
  kStale,
  kUnsynced,
  kMissingTransform
};

inline SensorFusionStatus classify_sensor_fusion(
  double receipt_age_sec, double receipt_timeout_sec,
  bool timestamp_delta_valid, double timestamp_delta_sec,
  double maximum_timestamp_delta_sec, bool transform_available)
{
  if (!std::isfinite(receipt_age_sec) || receipt_age_sec > receipt_timeout_sec) {
    return SensorFusionStatus::kStale;
  }
  if (timestamp_delta_valid &&
    (!std::isfinite(timestamp_delta_sec) ||
    timestamp_delta_sec > maximum_timestamp_delta_sec))
  {
    return SensorFusionStatus::kUnsynced;
  }
  if (!transform_available) {
    return SensorFusionStatus::kMissingTransform;
  }
  return SensorFusionStatus::kOk;
}

}  // namespace physicar_autonomy
