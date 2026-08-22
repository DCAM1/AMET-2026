// Copyright 2026 AICASTLE Inc.
// SPDX-License-Identifier: GPL-3.0-or-later

#include "physicar_autonomy/local_candidate_planner.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace physicar_autonomy
{
namespace
{

constexpr double kEpsilon = 1.0e-8;

double distance(const VehiclePoint & a, const VehiclePoint & b)
{
  return std::hypot(a.forward_m - b.forward_m, a.lateral_left_m - b.lateral_left_m);
}

double smoothstep5(double value)
{
  const double t = std::clamp(value, 0.0, 1.0);
  return t * t * t * (10.0 + t * (-15.0 + 6.0 * t));
}

std::vector<double> arc_lengths(const std::vector<VehiclePoint> & path)
{
  std::vector<double> arc(path.size(), 0.0);
  for (std::size_t i = 1; i < path.size(); ++i) {
    arc[i] = arc[i - 1] + distance(path[i - 1], path[i]);
  }
  return arc;
}

VehiclePoint normalized_tangent(const std::vector<VehiclePoint> & path, std::size_t index)
{
  const std::size_t before = index == 0 ? 0 : index - 1;
  const std::size_t after = std::min(index + 1, path.size() - 1);
  VehiclePoint tangent{
    path[after].forward_m - path[before].forward_m,
    path[after].lateral_left_m - path[before].lateral_left_m};
  const double length = std::hypot(tangent.forward_m, tangent.lateral_left_m);
  if (length <= kEpsilon) {
    return VehiclePoint{1.0, 0.0};
  }
  tangent.forward_m /= length;
  tangent.lateral_left_m /= length;
  return tangent;
}

double path_distance(const VehiclePoint & point, const std::vector<VehiclePoint> & path)
{
  if (path.empty()) {
    return std::numeric_limits<double>::infinity();
  }
  if (path.size() == 1U) {
    return distance(point, path.front());
  }
  double best = std::numeric_limits<double>::infinity();
  for (std::size_t i = 1; i < path.size(); ++i) {
    best = std::min(
      best, LocalCandidatePlanner::pointToSegmentDistance(point, path[i - 1], path[i]));
  }
  return best;
}

double obstacle_arc_position(
  const PlanningObstacle & obstacle,
  const std::vector<VehiclePoint> & path,
  const std::vector<double> & arc)
{
  double best_distance = std::numeric_limits<double>::infinity();
  double best_arc = 0.0;
  for (std::size_t i = 1; i < path.size(); ++i) {
    const double dx = path[i].forward_m - path[i - 1].forward_m;
    const double dy = path[i].lateral_left_m - path[i - 1].lateral_left_m;
    const double length_squared = dx * dx + dy * dy;
    const double projection = length_squared <= kEpsilon ? 0.0 : std::clamp(
      ((obstacle.center.forward_m - path[i - 1].forward_m) * dx +
      (obstacle.center.lateral_left_m - path[i - 1].lateral_left_m) * dy) /
      length_squared, 0.0, 1.0);
    const VehiclePoint nearest{
      path[i - 1].forward_m + projection * dx,
      path[i - 1].lateral_left_m + projection * dy};
    const double candidate_distance = distance(obstacle.center, nearest);
    if (candidate_distance < best_distance) {
      best_distance = candidate_distance;
      best_arc = arc[i - 1] + projection * std::sqrt(length_squared);
    }
  }
  return best_arc;
}

bool finite_path(const std::vector<VehiclePoint> & path)
{
  return std::all_of(path.begin(), path.end(), [](const VehiclePoint & point) {
      return std::isfinite(point.forward_m) && std::isfinite(point.lateral_left_m);
    });
}

}  // namespace

double LocalCandidatePlanner::pointToSegmentDistance(
  const VehiclePoint & point,
  const VehiclePoint & segment_start,
  const VehiclePoint & segment_end)
{
  const double dx = segment_end.forward_m - segment_start.forward_m;
  const double dy = segment_end.lateral_left_m - segment_start.lateral_left_m;
  const double length_squared = dx * dx + dy * dy;
  if (length_squared <= kEpsilon) {
    return distance(point, segment_start);
  }
  const double projection = std::clamp(
    ((point.forward_m - segment_start.forward_m) * dx +
    (point.lateral_left_m - segment_start.lateral_left_m) * dy) / length_squared,
    0.0, 1.0);
  return std::hypot(
    point.forward_m - (segment_start.forward_m + projection * dx),
    point.lateral_left_m - (segment_start.lateral_left_m + projection * dy));
}

double LocalCandidatePlanner::maximumCurvature(const std::vector<VehiclePoint> & path)
{
  double maximum = 0.0;
  for (std::size_t i = 1; i + 1 < path.size(); ++i) {
    std::size_t before = i - 1;
    std::size_t after = i + 1;
    if (path.size() > 5U) {
      constexpr double kCurvatureChordM = 0.04;
      while (before > 0U && distance(path[before], path[i]) < kCurvatureChordM) {
        --before;
      }
      while (after + 1U < path.size() &&
        distance(path[i], path[after]) < kCurvatureChordM)
      {
        ++after;
      }
      if (distance(path[before], path[i]) < kCurvatureChordM ||
        distance(path[i], path[after]) < kCurvatureChordM)
      {
        continue;
      }
    }
    const double a = distance(path[before], path[i]);
    const double b = distance(path[i], path[after]);
    const double c = distance(path[before], path[after]);
    const double denominator = a * b * c;
    if (denominator <= kEpsilon) {
      continue;
    }
    const double twice_area = std::abs(
      (path[i].forward_m - path[before].forward_m) *
      (path[after].lateral_left_m - path[before].lateral_left_m) -
      (path[i].lateral_left_m - path[before].lateral_left_m) *
      (path[after].forward_m - path[before].forward_m));
    maximum = std::max(maximum, 2.0 * twice_area / denominator);
  }
  return maximum;
}

CandidatePlan LocalCandidatePlanner::plan(
  const std::vector<VehiclePoint> & base_path,
  const std::vector<PlanningObstacle> & obstacles,
  const CandidatePlannerConfig & config,
  int previous_side_sign,
  double previous_target_offset_m,
  int preferred_side_sign) const
{
  CandidatePlan result;
  const int candidate_count = std::max(3, config.candidate_count | 1);
  result.candidates.reserve(static_cast<std::size_t>(candidate_count));
  if (base_path.size() < 3U || !finite_path(base_path)) {
    result.blocked = false;
    return result;
  }

  const std::vector<double> detected_arc = arc_lengths(base_path);
  const double detected_total_arc = detected_arc.back();
  if (detected_total_arc <= 0.1) {
    return result;
  }

  double obstacle_arc = std::numeric_limits<double>::infinity();
  double obstacle_pass_radius = 0.0;
  for (const auto & obstacle : obstacles) {
    if (!std::isfinite(obstacle.center.forward_m) ||
      !std::isfinite(obstacle.center.lateral_left_m) ||
      !std::isfinite(obstacle.radius_m) || obstacle.radius_m < 0.0)
    {
      continue;
    }
    if (!obstacle.collision_check) {
      continue;
    }
    const double centerline_clearance = path_distance(obstacle.center, base_path) -
      obstacle.radius_m;
    const double along = obstacle_arc_position(obstacle, base_path, detected_arc);
    if ((centerline_clearance < config.obstacle_inflation_m || previous_side_sign != 0) &&
      along < obstacle_arc)
    {
      obstacle_arc = along;
      obstacle_pass_radius = obstacle.radius_m + config.obstacle_inflation_m;
    }
  }
  if (!std::isfinite(obstacle_arc) && previous_side_sign != 0) {
    for (const auto & obstacle : obstacles) {
      if (!std::isfinite(obstacle.center.forward_m) ||
        !std::isfinite(obstacle.center.lateral_left_m) ||
        !std::isfinite(obstacle.radius_m) || obstacle.radius_m < 0.0)
      {
        continue;
      }
      const double along = obstacle_arc_position(obstacle, base_path, detected_arc);
      if (along < obstacle_arc) {
        obstacle_arc = along;
        obstacle_pass_radius = obstacle.radius_m + config.obstacle_inflation_m;
      }
    }
  }
  if (!std::isfinite(obstacle_arc)) {
    obstacle_arc = 0.5 * detected_total_arc;
  }
  result.obstacle_arc_m = obstacle_arc;
  const double pass_end = obstacle_arc + std::max(0.08, obstacle_pass_radius);
  const double maximum_transition_distance = std::max(
    config.rejoin_distance_m,
    std::sqrt(
      6.0 * config.maximum_lateral_offset_m /
      std::max(0.1, config.maximum_curvature_per_m)));
  std::vector<VehiclePoint> planning_path = base_path;
  const double required_total_arc = pass_end + maximum_transition_distance;
  if (detected_total_arc < required_total_arc) {
    const VehiclePoint tangent = normalized_tangent(base_path, base_path.size() - 1U);
    const double spacing = std::clamp(
      detected_total_arc / static_cast<double>(base_path.size() - 1U), 0.01, 0.05);
    double extended_arc = detected_total_arc;
    while (extended_arc + kEpsilon < required_total_arc) {
      const double step = std::min(spacing, required_total_arc - extended_arc);
      planning_path.push_back(VehiclePoint{
        planning_path.back().forward_m + step * tangent.forward_m,
        planning_path.back().lateral_left_m + step * tangent.lateral_left_m});
      extended_arc += step;
    }
  }
  const std::vector<double> arc = arc_lengths(planning_path);
  double initial_path_offset_m = 0.0;
  if (previous_side_sign != 0) {
    const VehiclePoint initial_tangent = normalized_tangent(planning_path, 0U);
    const VehiclePoint initial_left_normal{
      -initial_tangent.lateral_left_m, initial_tangent.forward_m};
    initial_path_offset_m = std::clamp(
      -planning_path.front().forward_m * initial_left_normal.forward_m -
      planning_path.front().lateral_left_m * initial_left_normal.lateral_left_m,
      -config.maximum_lateral_offset_m, config.maximum_lateral_offset_m);
  }

  for (int candidate_index = 0; candidate_index < candidate_count; ++candidate_index) {
    CandidatePath candidate;
    const double ratio = static_cast<double>(candidate_index) /
      static_cast<double>(candidate_count - 1);
    candidate.target_offset_m =
      (2.0 * ratio - 1.0) * config.maximum_lateral_offset_m;
    candidate.points.reserve(planning_path.size());
    candidate.minimum_obstacle_clearance_m = std::numeric_limits<double>::infinity();

    const double transition_distance = std::max(
      config.approach_distance_m,
      std::sqrt(
        6.0 * std::abs(candidate.target_offset_m) /
        std::max(0.1, config.maximum_curvature_per_m)));
    const double departure_start = std::max(0.0, obstacle_arc - transition_distance);
    const double rejoin_distance = std::max(config.rejoin_distance_m, transition_distance);
    const double rejoin_end = pass_end + rejoin_distance;
    for (std::size_t i = 0; i < planning_path.size(); ++i) {
      double lateral_offset_m = 0.0;
      if (arc[i] >= departure_start && arc[i] <= obstacle_arc) {
        const double transition = smoothstep5(
          (arc[i] - departure_start) /
          std::max(kEpsilon, obstacle_arc - departure_start));
        lateral_offset_m = previous_side_sign == 0 ?
          candidate.target_offset_m * transition :
          initial_path_offset_m +
          (candidate.target_offset_m - initial_path_offset_m) * transition;
      } else if (previous_side_sign != 0 && arc[i] < departure_start) {
        lateral_offset_m = initial_path_offset_m;
      } else if (arc[i] > obstacle_arc && arc[i] <= pass_end) {
        lateral_offset_m = obstacle_arc <= kEpsilon && previous_side_sign != 0 ?
          initial_path_offset_m : candidate.target_offset_m;
      } else if (arc[i] > pass_end && arc[i] < rejoin_end) {
        const double pass_offset_m = obstacle_arc <= kEpsilon && previous_side_sign != 0 ?
          initial_path_offset_m : candidate.target_offset_m;
        lateral_offset_m = pass_offset_m * (1.0 - smoothstep5(
            (arc[i] - pass_end) / std::max(kEpsilon, rejoin_end - pass_end)));
      }
      const VehiclePoint tangent = normalized_tangent(planning_path, i);
      const VehiclePoint left_normal{-tangent.lateral_left_m, tangent.forward_m};
      candidate.points.push_back(VehiclePoint{
        planning_path[i].forward_m + lateral_offset_m * left_normal.forward_m,
        planning_path[i].lateral_left_m + lateral_offset_m * left_normal.lateral_left_m});
      candidate.deviation_cost += std::abs(lateral_offset_m);
    }
    candidate.deviation_cost /= static_cast<double>(candidate.points.size());

    if (!finite_path(candidate.points)) {
      candidate.rejection_reason = "INVALID_GEOMETRY";
      result.candidates.push_back(std::move(candidate));
      continue;
    }

    bool lost_progress = false;
    double accumulated_reverse_distance_m = 0.0;
    for (std::size_t i = 1; i < candidate.points.size(); ++i) {
      const VehiclePoint tangent = normalized_tangent(planning_path, i);
      const VehiclePoint delta{
        candidate.points[i].forward_m - candidate.points[i - 1].forward_m,
        candidate.points[i].lateral_left_m - candidate.points[i - 1].lateral_left_m};
      const double segment_length = std::hypot(delta.forward_m, delta.lateral_left_m);
      candidate.path_length_m += segment_length;
      if (segment_length <= 1.0e-4) {
        continue;
      }
      const double normalized_progress =
        (delta.forward_m * tangent.forward_m +
        delta.lateral_left_m * tangent.lateral_left_m) / segment_length;
      if (normalized_progress < -0.10) {
        accumulated_reverse_distance_m += segment_length;
        if (accumulated_reverse_distance_m > 0.03) {
          lost_progress = true;
          break;
        }
      } else {
        accumulated_reverse_distance_m = 0.0;
      }
    }
    if (lost_progress) {
      candidate.rejection_reason = "NO_FORWARD_PROGRESS";
      result.candidates.push_back(std::move(candidate));
      continue;
    }

    bool collision = false;
    for (const auto & obstacle : obstacles) {
      if (!obstacle.collision_check) {
        continue;
      }
      double obstacle_distance = std::numeric_limits<double>::infinity();
      for (std::size_t i = 1; i < candidate.points.size(); ++i) {
        obstacle_distance = std::min(
          obstacle_distance,
          pointToSegmentDistance(obstacle.center, candidate.points[i - 1], candidate.points[i]));
      }
      const double clearance = obstacle_distance - obstacle.radius_m;
      candidate.minimum_obstacle_clearance_m = std::min(
        candidate.minimum_obstacle_clearance_m, clearance);
      if (clearance < config.obstacle_inflation_m) {
        collision = true;
      }
    }
    if (collision) {
      candidate.rejection_reason = "OBSTACLE_COLLISION";
      result.candidates.push_back(std::move(candidate));
      continue;
    }

    candidate.maximum_curvature_per_m = maximumCurvature(candidate.points);
    double previous_curvature = 0.0;
    for (std::size_t i = 1; i + 1 < candidate.points.size(); ++i) {
      std::vector<VehiclePoint> triple{
        candidate.points[i - 1], candidate.points[i], candidate.points[i + 1]};
      const double curvature = maximumCurvature(triple);
      candidate.curvature_change_cost += std::abs(curvature - previous_curvature);
      previous_curvature = curvature;
    }

    candidate.rejoin_distance_m = std::max(0.0, rejoin_end - pass_end);
    candidate.remaining_rejoin_distance_m = std::max(0.0, rejoin_end);
    const double end_deviation = path_distance(candidate.points.back(), planning_path);
    if (std::abs(candidate.target_offset_m) > config.rejoin_tolerance_m &&
      end_deviation > config.rejoin_tolerance_m)
    {
      candidate.rejection_reason = "REJOIN_INCOMPLETE";
      result.candidates.push_back(std::move(candidate));
      continue;
    }

    const int side_sign = candidate.target_offset_m > 1.0e-4 ? 1 :
      (candidate.target_offset_m < -1.0e-4 ? -1 : 0);
    const double clearance_reward = std::isfinite(candidate.minimum_obstacle_clearance_m) ?
      0.08 / std::max(0.02, candidate.minimum_obstacle_clearance_m) : 0.0;
    candidate.total_cost =
      0.25 * candidate.path_length_m +
      1.2 * candidate.deviation_cost +
      0.20 * candidate.maximum_curvature_per_m +
      0.01 * candidate.curvature_change_cost +
      0.10 * candidate.rejoin_distance_m + clearance_reward;
    if (previous_side_sign != 0 && side_sign != 0 && side_sign != previous_side_sign) {
      candidate.total_cost += config.side_switch_penalty;
    }
    if (std::abs(previous_target_offset_m) > 1.0e-4) {
      candidate.total_cost += 2.0 * std::abs(
        candidate.target_offset_m - previous_target_offset_m);
      if (side_sign == 0) {
        candidate.total_cost += config.side_switch_penalty;
      }
    }
    if (previous_side_sign == 0 && preferred_side_sign != 0 &&
      side_sign != 0 && side_sign != preferred_side_sign)
    {
      candidate.total_cost += config.preferred_side_penalty;
    }
    candidate.valid = true;
    result.candidates.push_back(std::move(candidate));
  }

  result.all_candidates_evaluated =
    result.candidates.size() == static_cast<std::size_t>(candidate_count);
  double best_cost = std::numeric_limits<double>::infinity();
  for (std::size_t i = 0; i < result.candidates.size(); ++i) {
    if (result.candidates[i].valid && result.candidates[i].total_cost < best_cost) {
      best_cost = result.candidates[i].total_cost;
      result.selected_index = static_cast<int>(i);
    }
  }
  if (std::abs(previous_target_offset_m) > 1.0e-4) {
    for (std::size_t i = 0; i < result.candidates.size(); ++i) {
      if (result.candidates[i].valid &&
        std::abs(result.candidates[i].target_offset_m - previous_target_offset_m) < 1.0e-4)
      {
        result.selected_index = static_cast<int>(i);
        break;
      }
    }
  }
  result.blocked = result.all_candidates_evaluated && result.selected_index < 0;
  if (result.selected_index >= 0) {
    result.remaining_rejoin_distance_m =
      result.candidates[
      static_cast<std::size_t>(result.selected_index)].remaining_rejoin_distance_m;
  }
  return result;
}

}  // namespace physicar_autonomy
