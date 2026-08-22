// Copyright 2026 AICASTLE Inc.
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "physicar_autonomy/bev_geometry.hpp"

namespace physicar_autonomy
{

struct PlanningObstacle
{
  VehiclePoint center;
  double radius_m{0.0};
  bool collision_check{true};
};

struct CandidatePlannerConfig
{
  int candidate_count{9};
  double maximum_lateral_offset_m{0.20};
  double approach_distance_m{0.30};
  double rejoin_distance_m{0.40};
  double rejoin_tolerance_m{0.02};
  double obstacle_inflation_m{0.14};
  double maximum_curvature_per_m{2.30};
  double side_switch_penalty{0.35};
  double preferred_side_penalty{0.03};
};

struct CandidatePath
{
  std::vector<VehiclePoint> points;
  double target_offset_m{0.0};
  double minimum_obstacle_clearance_m{0.0};
  double maximum_curvature_per_m{0.0};
  double curvature_change_cost{0.0};
  double path_length_m{0.0};
  double deviation_cost{0.0};
  double rejoin_distance_m{0.0};
  double remaining_rejoin_distance_m{0.0};
  double total_cost{0.0};
  bool valid{false};
  std::string rejection_reason;
};

struct CandidatePlan
{
  std::vector<CandidatePath> candidates;
  int selected_index{-1};
  bool all_candidates_evaluated{false};
  bool blocked{false};
  double obstacle_arc_m{0.0};
  double remaining_rejoin_distance_m{0.0};
};

class LocalCandidatePlanner
{
public:
  CandidatePlan plan(
    const std::vector<VehiclePoint> & base_path,
    const std::vector<PlanningObstacle> & obstacles,
    const CandidatePlannerConfig & config,
    int previous_side_sign,
    double previous_target_offset_m,
    int preferred_side_sign) const;

  static double pointToSegmentDistance(
    const VehiclePoint & point,
    const VehiclePoint & segment_start,
    const VehiclePoint & segment_end);

  static double maximumCurvature(const std::vector<VehiclePoint> & path);
};

}  // namespace physicar_autonomy
