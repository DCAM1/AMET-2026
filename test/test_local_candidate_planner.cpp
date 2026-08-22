// Copyright 2026 AICASTLE Inc.
// SPDX-License-Identifier: GPL-3.0-or-later

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "physicar_autonomy/local_candidate_planner.hpp"
#include "physicar_autonomy/sensor_fusion_policy.hpp"

namespace physicar_autonomy
{
namespace
{

std::vector<VehiclePoint> path_with_bend(double bend)
{
  std::vector<VehiclePoint> path;
  for (int index = 0; index <= 100; ++index) {
    const double forward = 0.015 * static_cast<double>(index);
    path.push_back(VehiclePoint{forward, bend * forward * forward});
  }
  return path;
}

CandidatePlannerConfig test_config()
{
  CandidatePlannerConfig config;
  config.candidate_count = 9;
  config.maximum_lateral_offset_m = 0.24;
  config.approach_distance_m = 0.30;
  config.rejoin_distance_m = 0.40;
  config.rejoin_tolerance_m = 0.02;
  config.obstacle_inflation_m = 0.10;
  config.maximum_curvature_per_m = 2.30;
  return config;
}

TEST(LocalCandidatePlanner, AvoidsObstacleOnStraight)
{
  LocalCandidatePlanner planner;
  const auto plan = planner.plan(
    path_with_bend(0.0),
    {PlanningObstacle{VehiclePoint{0.75, 0.0}, 0.03}},
    test_config(), 0, 0.0, 1);
  ASSERT_FALSE(plan.blocked);
  ASSERT_GE(plan.selected_index, 0);
  EXPECT_NEAR(
    plan.candidates[static_cast<std::size_t>(plan.selected_index)].points.back().lateral_left_m,
    0.0, 0.02);
}

TEST(LocalCandidatePlanner, ChoosesOutsideOfLeftCorner)
{
  LocalCandidatePlanner planner;
  const auto base = path_with_bend(0.16);
  const auto plan = planner.plan(
    base, {PlanningObstacle{VehiclePoint{0.75, 0.14}, 0.03}},
    test_config(), 0, 0.0, -1);
  ASSERT_FALSE(plan.blocked);
  ASSERT_GE(plan.selected_index, 0);
  EXPECT_LT(
    plan.candidates[static_cast<std::size_t>(plan.selected_index)].target_offset_m, 0.0);
}

TEST(LocalCandidatePlanner, ChoosesOutsideOfRightCorner)
{
  LocalCandidatePlanner planner;
  const auto base = path_with_bend(-0.16);
  const auto plan = planner.plan(
    base, {PlanningObstacle{VehiclePoint{0.75, -0.14}, 0.03}},
    test_config(), 0, 0.0, 1);
  ASSERT_FALSE(plan.blocked);
  ASSERT_GE(plan.selected_index, 0);
  EXPECT_GT(
    plan.candidates[static_cast<std::size_t>(plan.selected_index)].target_offset_m, 0.0);
}

TEST(LocalCandidatePlanner, OutsideCornerObstacleDoesNotBlockBasePath)
{
  LocalCandidatePlanner planner;
  const auto plan = planner.plan(
    path_with_bend(0.16),
    {PlanningObstacle{VehiclePoint{0.75, -0.28}, 0.03}},
    test_config(), 0, 0.0, 1);
  EXPECT_FALSE(plan.blocked);
  EXPECT_GE(plan.selected_index, 0);
}

TEST(LocalCandidatePlanner, ReportsTrulyBlockedCorridor)
{
  LocalCandidatePlanner planner;
  std::vector<PlanningObstacle> obstacles;
  for (int index = -4; index <= 4; ++index) {
    obstacles.push_back(PlanningObstacle{
      VehiclePoint{0.75, 0.06 * static_cast<double>(index)}, 0.05});
  }
  const auto plan = planner.plan(
    path_with_bend(0.0), obstacles, test_config(), 0, 0.0, 1);
  EXPECT_TRUE(plan.all_candidates_evaluated);
  EXPECT_TRUE(plan.blocked);
  EXPECT_LT(plan.selected_index, 0);
}

TEST(LocalCandidatePlanner, RejoinsWithinConfiguredDistanceWhenCurvatureAllows)
{
  LocalCandidatePlanner planner;
  const auto base = path_with_bend(0.0);
  auto config = test_config();
  config.maximum_curvature_per_m = 20.0;
  const auto plan = planner.plan(
    base, {PlanningObstacle{VehiclePoint{0.75, 0.0}, 0.03}},
    config, 0, 0.0, 1);
  ASSERT_GE(plan.selected_index, 0);
  const auto & selected = plan.candidates[static_cast<std::size_t>(plan.selected_index)];
  EXPECT_LE(selected.rejoin_distance_m, 0.40 + 1.0e-6);
  EXPECT_LE(std::abs(selected.points.back().lateral_left_m), 0.02);
}

TEST(LocalCandidatePlanner, ExtendsRejoinRatherThanViolatingCurvatureLimit)
{
  LocalCandidatePlanner planner;
  const auto plan = planner.plan(
    path_with_bend(0.0),
    {PlanningObstacle{VehiclePoint{0.75, 0.0}, 0.03}},
    test_config(), 0, 0.0, 1);
  ASSERT_GE(plan.selected_index, 0);
  const auto & selected = plan.candidates[static_cast<std::size_t>(plan.selected_index)];
  EXPECT_LE(selected.maximum_curvature_per_m, 2.30);
  EXPECT_LE(std::abs(selected.points.back().lateral_left_m), 0.02);
  EXPECT_GT(selected.rejoin_distance_m, 0.40);
}

TEST(LocalCandidatePlanner, CurvatureDoesNotRejectCandidates)
{
  LocalCandidatePlanner planner;
  auto config = test_config();
  config.maximum_curvature_per_m = 0.10;
  const auto plan = planner.plan(
    path_with_bend(0.25),
    {PlanningObstacle{VehiclePoint{0.75, 0.12}, 0.03}},
    config, 0, 0.0, -1);
  ASSERT_TRUE(plan.all_candidates_evaluated);
  EXPECT_FALSE(plan.blocked);
  EXPECT_GE(plan.selected_index, 0);
  for (const auto & candidate : plan.candidates) {
    EXPECT_NE(candidate.rejection_reason, "CURVATURE_LIMIT");
  }
}

TEST(LocalCandidatePlanner, PreservesActivePathFromCurrentVehiclePosition)
{
  LocalCandidatePlanner planner;
  auto shifted_base = path_with_bend(0.0);
  for (auto & point : shifted_base) {
    point.lateral_left_m -= 0.12;
  }
  const auto plan = planner.plan(
    shifted_base,
    {PlanningObstacle{VehiclePoint{-0.10, -0.12}, 0.03, false}},
    test_config(), 1, 0.12, 1);
  ASSERT_FALSE(plan.blocked);
  ASSERT_GE(plan.selected_index, 0);
  const auto & selected = plan.candidates[static_cast<std::size_t>(plan.selected_index)];
  EXPECT_NEAR(selected.target_offset_m, 0.12, 1.0e-6);
  EXPECT_NEAR(selected.points.front().lateral_left_m, 0.0, 1.0e-6);
  EXPECT_LE(std::abs(selected.points.back().lateral_left_m + 0.12), 0.02);
}

TEST(SensorFusionPolicy, RejectsStaleLidar)
{
  EXPECT_EQ(
    classify_sensor_fusion(0.61, 0.60, true, 0.01, 0.15, true),
    SensorFusionStatus::kStale);
}

TEST(SensorFusionPolicy, RejectsMissingTransform)
{
  EXPECT_EQ(
    classify_sensor_fusion(0.01, 0.60, true, 0.01, 0.15, false),
    SensorFusionStatus::kMissingTransform);
}

TEST(SensorFusionPolicy, RejectsUnsynchronizedSensorData)
{
  EXPECT_EQ(
    classify_sensor_fusion(0.01, 0.60, true, 0.16, 0.15, true),
    SensorFusionStatus::kUnsynced);
}

TEST(SensorFusionPolicy, ZeroTimestampFallsBackToReceiptAge)
{
  EXPECT_EQ(
    classify_sensor_fusion(0.01, 0.60, false, 0.0, 0.15, true),
    SensorFusionStatus::kOk);
}

TEST(BevGeometry, VehiclePixelConversionRoundTrips)
{
  const BevGeometry geometry(480, 360, 364.0, 0.70, 1.50);
  const auto pixel = geometry.vehicleToBevPixel(0.82, -0.16);
  ASSERT_TRUE(pixel.has_value());
  const auto vehicle = geometry.bevPixelToVehicle(pixel->x, pixel->y);
  ASSERT_TRUE(vehicle.has_value());
  EXPECT_NEAR(vehicle->forward_m, 0.82, 1.0e-9);
  EXPECT_NEAR(vehicle->lateral_left_m, -0.16, 1.0e-9);
}

}  // namespace
}  // namespace physicar_autonomy
