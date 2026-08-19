// Copyright 2026 AICASTLE Inc.
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <geometry_msgs/msg/point.hpp>
#include <physicar_interfaces/msg/obstacle.hpp>
#include <physicar_interfaces/msg/obstacle_array.hpp>
#include <rcl_interfaces/msg/parameter_descriptor.hpp>
#include <rcl_interfaces/msg/set_parameters_result.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

namespace physicar_autonomy
{

struct LidarConfig
{
  double angle_min_deg{-75.0};
  double angle_max_deg{75.0};
  double range_min_m{0.15};
  double range_max_m{3.0};
  double cluster_distance_m{0.12};
  int minimum_cluster_size{3};
  double minimum_cluster_width_m{0.01};
  double maximum_cluster_width_m{0.80};
};

class LidarObstacleNode : public rclcpp::Node
{
public:
  LidarObstacleNode()
  : Node("lidar_obstacle"),
    steady_clock_(RCL_STEADY_TIME),
    started_at_(std::chrono::steady_clock::now()),
    last_scan_at_(started_at_)
  {
    rcl_interfaces::msg::ParameterDescriptor topic_descriptor;
    topic_descriptor.read_only = true;
    topic_descriptor.description = "Restart the node to change a ROS topic name";
    const auto scan_topic = declare_parameter<std::string>(
      "scan_topic", "/scan", topic_descriptor);
    const auto obstacle_topic = declare_parameter<std::string>(
      "obstacle_topic", "/obstacles", topic_descriptor);
    const auto marker_topic = declare_parameter<std::string>(
      "marker_topic", "/obstacle/debug/markers", topic_descriptor);

    config_.angle_min_deg = declare_parameter<double>("angle_min_deg", -75.0);
    config_.angle_max_deg = declare_parameter<double>("angle_max_deg", 75.0);
    config_.range_min_m = declare_parameter<double>("range_min_m", 0.15);
    config_.range_max_m = declare_parameter<double>("range_max_m", 3.0);
    config_.cluster_distance_m = declare_parameter<double>("cluster_distance_m", 0.12);
    config_.minimum_cluster_size = declare_parameter<int>("minimum_cluster_size", 3);
    config_.minimum_cluster_width_m = declare_parameter<double>(
      "minimum_cluster_width_m", 0.01);
    config_.maximum_cluster_width_m = declare_parameter<double>(
      "maximum_cluster_width_m", 0.80);
    scan_timeout_sec_ = declare_parameter<double>("scan_timeout_sec", 1.0);
    timing_log_enabled_ = declare_parameter<bool>("timing_log_enabled", false);
    timing_log_period_ms_ = static_cast<int64_t>(1000.0 * declare_parameter<double>(
      "timing_log_period_sec", 2.0));

    std::string error;
    if (!validate(config_, scan_timeout_sec_, timing_log_period_ms_, error)) {
      throw std::invalid_argument("Invalid LiDAR parameters: " + error);
    }
    parameter_callback_ = add_on_set_parameters_callback(
      std::bind(&LidarObstacleNode::on_parameters, this, std::placeholders::_1));

    const auto sensor_qos = rclcpp::QoS(rclcpp::KeepLast(1))
      .best_effort().durability_volatile();
    const auto output_qos = rclcpp::QoS(rclcpp::KeepLast(1))
      .reliable().durability_volatile();
    obstacle_pub_ = create_publisher<physicar_interfaces::msg::ObstacleArray>(
      obstacle_topic, output_qos);
    marker_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(
      marker_topic, output_qos);
    scan_sub_ = create_subscription<sensor_msgs::msg::LaserScan>(
      scan_topic, sensor_qos,
      std::bind(&LidarObstacleNode::on_scan, this, std::placeholders::_1));
    watchdog_ = create_wall_timer(
      std::chrono::milliseconds(250),
      std::bind(&LidarObstacleNode::check_scan_timeout, this));

    RCLCPP_INFO(
      get_logger(),
      "Phase 9 ready: %s -> {%s, %s}; front FOV %.1f..%.1f deg, %.2f..%.2f m",
      scan_topic.c_str(), obstacle_topic.c_str(), marker_topic.c_str(),
      config_.angle_min_deg, config_.angle_max_deg,
      config_.range_min_m, config_.range_max_m);
  }

private:
  static bool validate(
    const LidarConfig & config, double timeout_sec,
    int64_t timing_period_ms, std::string & reason)
  {
    if (config.angle_min_deg < -180.0 || config.angle_max_deg > 180.0 ||
      config.angle_min_deg >= config.angle_max_deg)
    {
      reason = "angles must satisfy -180 <= min < max <= 180 degrees";
      return false;
    }
    if (config.range_min_m < 0.05 || config.range_max_m <= config.range_min_m ||
      config.range_max_m > 16.0)
    {
      reason = "ranges must satisfy 0.05 <= min < max <= 16.0 m";
      return false;
    }
    if (config.cluster_distance_m < 0.01 || config.cluster_distance_m > 1.0) {
      reason = "cluster_distance_m must be within [0.01, 1.0]";
      return false;
    }
    if (config.minimum_cluster_size < 2 || config.minimum_cluster_size > 1000) {
      reason = "minimum_cluster_size must be within [2, 1000]";
      return false;
    }
    if (config.minimum_cluster_width_m < 0.0 ||
      config.maximum_cluster_width_m <= config.minimum_cluster_width_m ||
      config.maximum_cluster_width_m > 5.0)
    {
      reason = "cluster widths must satisfy 0 <= min < max <= 5.0 m";
      return false;
    }
    if (timeout_sec < 0.1 || timeout_sec > 10.0 ||
      timing_period_ms < 100 || timing_period_ms > 60000)
    {
      reason = "timeout/timing periods are outside safe bounds";
      return false;
    }
    return true;
  }

  rcl_interfaces::msg::SetParametersResult on_parameters(
    const std::vector<rclcpp::Parameter> & parameters)
  {
    auto candidate = config_;
    double candidate_timeout = scan_timeout_sec_;
    bool candidate_timing_enabled = timing_log_enabled_;
    int64_t candidate_timing_period = timing_log_period_ms_;
    for (const auto & parameter : parameters) {
      const auto & name = parameter.get_name();
      if (name == "angle_min_deg") {
        candidate.angle_min_deg = parameter.as_double();
      } else if (name == "angle_max_deg") {
        candidate.angle_max_deg = parameter.as_double();
      } else if (name == "range_min_m") {
        candidate.range_min_m = parameter.as_double();
      } else if (name == "range_max_m") {
        candidate.range_max_m = parameter.as_double();
      } else if (name == "cluster_distance_m") {
        candidate.cluster_distance_m = parameter.as_double();
      } else if (name == "minimum_cluster_size") {
        candidate.minimum_cluster_size = static_cast<int>(parameter.as_int());
      } else if (name == "minimum_cluster_width_m") {
        candidate.minimum_cluster_width_m = parameter.as_double();
      } else if (name == "maximum_cluster_width_m") {
        candidate.maximum_cluster_width_m = parameter.as_double();
      } else if (name == "scan_timeout_sec") {
        candidate_timeout = parameter.as_double();
      } else if (name == "timing_log_enabled") {
        candidate_timing_enabled = parameter.as_bool();
      } else if (name == "timing_log_period_sec") {
        candidate_timing_period = static_cast<int64_t>(parameter.as_double() * 1000.0);
      }
    }
    rcl_interfaces::msg::SetParametersResult result;
    std::string error;
    result.successful = validate(
      candidate, candidate_timeout, candidate_timing_period, error);
    result.reason = error;
    if (result.successful) {
      config_ = candidate;
      scan_timeout_sec_ = candidate_timeout;
      timing_log_enabled_ = candidate_timing_enabled;
      timing_log_period_ms_ = candidate_timing_period;
    }
    return result;
  }

  void append_cluster(
    const std::vector<geometry_msgs::msg::Point> & points,
    physicar_interfaces::msg::ObstacleArray & output) const
  {
    if (static_cast<int>(points.size()) < config_.minimum_cluster_size) {
      return;
    }
    double sum_x = 0.0;
    double sum_y = 0.0;
    double min_x = std::numeric_limits<double>::infinity();
    double max_x = -std::numeric_limits<double>::infinity();
    double min_y = std::numeric_limits<double>::infinity();
    double max_y = -std::numeric_limits<double>::infinity();
    for (const auto & point : points) {
      sum_x += point.x;
      sum_y += point.y;
      min_x = std::min(min_x, point.x);
      max_x = std::max(max_x, point.x);
      min_y = std::min(min_y, point.y);
      max_y = std::max(max_y, point.y);
    }
    const double width = std::hypot(max_x - min_x, max_y - min_y);
    if (width < config_.minimum_cluster_width_m ||
      width > config_.maximum_cluster_width_m)
    {
      return;
    }
    physicar_interfaces::msg::Obstacle obstacle;
    obstacle.centroid.x = sum_x / static_cast<double>(points.size());
    obstacle.centroid.y = sum_y / static_cast<double>(points.size());
    obstacle.centroid.z = 0.0;
    obstacle.width = static_cast<float>(width);
    obstacle.distance = static_cast<float>(std::hypot(
      obstacle.centroid.x, obstacle.centroid.y));
    obstacle.angle = static_cast<float>(std::atan2(
      obstacle.centroid.y, obstacle.centroid.x));
    obstacle.point_count = static_cast<uint32_t>(points.size());
    output.obstacles.push_back(obstacle);
  }

  void publish_markers(const physicar_interfaces::msg::ObstacleArray & obstacles)
  {
    visualization_msgs::msg::MarkerArray array;
    visualization_msgs::msg::Marker clear;
    clear.header = obstacles.header;
    clear.action = visualization_msgs::msg::Marker::DELETEALL;
    array.markers.push_back(clear);
    int marker_id = 0;
    for (const auto & obstacle : obstacles.obstacles) {
      visualization_msgs::msg::Marker body;
      body.header = obstacles.header;
      body.ns = "obstacle_clusters";
      body.id = marker_id++;
      body.type = visualization_msgs::msg::Marker::CYLINDER;
      body.action = visualization_msgs::msg::Marker::ADD;
      body.pose.position = obstacle.centroid;
      body.pose.position.z = 0.08;
      body.pose.orientation.w = 1.0;
      body.scale.x = std::max(0.08, static_cast<double>(obstacle.width));
      body.scale.y = body.scale.x;
      body.scale.z = 0.16;
      body.color.r = 1.0F;
      body.color.g = 0.15F;
      body.color.b = 0.05F;
      body.color.a = 0.90F;
      body.lifetime.sec = 0;
      body.lifetime.nanosec = 300000000U;
      array.markers.push_back(body);
    }
    marker_pub_->publish(array);
  }

  void on_scan(const sensor_msgs::msg::LaserScan::ConstSharedPtr scan)
  {
    const auto started = std::chrono::steady_clock::now();
    last_scan_at_ = started;
    received_scan_ = true;
    physicar_interfaces::msg::ObstacleArray output;
    output.header = scan->header;
    std::vector<geometry_msgs::msg::Point> cluster;
    geometry_msgs::msg::Point previous;
    bool have_previous = false;
    const double degrees_per_radian = 180.0 / 3.14159265358979323846;

    for (std::size_t index = 0; index < scan->ranges.size(); ++index) {
      const double angle = scan->angle_min + static_cast<double>(index) * scan->angle_increment;
      const double angle_deg = angle * degrees_per_radian;
      const double range = scan->ranges[index];
      const bool valid = angle_deg >= config_.angle_min_deg &&
        angle_deg <= config_.angle_max_deg && std::isfinite(range) &&
        range >= std::max(config_.range_min_m, static_cast<double>(scan->range_min)) &&
        range <= std::min(config_.range_max_m, static_cast<double>(scan->range_max));
      if (!valid) {
        append_cluster(cluster, output);
        cluster.clear();
        have_previous = false;
        continue;
      }

      geometry_msgs::msg::Point point;
      point.x = range * std::cos(angle);
      point.y = range * std::sin(angle);
      point.z = 0.0;
      if (have_previous &&
        std::hypot(point.x - previous.x, point.y - previous.y) >
        config_.cluster_distance_m)
      {
        append_cluster(cluster, output);
        cluster.clear();
      }
      cluster.push_back(point);
      previous = point;
      have_previous = true;
    }
    append_cluster(cluster, output);
    obstacle_pub_->publish(output);
    publish_markers(output);

    if (timing_log_enabled_) {
      const auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - started).count();
      RCLCPP_INFO_THROTTLE(
        get_logger(), steady_clock_, timing_log_period_ms_,
        "LiDAR clustering: %.3f ms, %zu input points, %zu obstacles",
        static_cast<double>(elapsed_us) / 1000.0,
        scan->ranges.size(), output.obstacles.size());
    }
  }

  void check_scan_timeout()
  {
    const auto now = std::chrono::steady_clock::now();
    const auto reference = received_scan_ ? last_scan_at_ : started_at_;
    const double age_sec = std::chrono::duration<double>(now - reference).count();
    if (age_sec <= scan_timeout_sec_) {
      return;
    }
    RCLCPP_WARN_THROTTLE(
      get_logger(), steady_clock_, 2000,
      received_scan_ ? "LiDAR scan is stale (%.2f s)" : "Waiting for LiDAR scan (%.2f s)",
      age_sec);
  }

  rclcpp::Clock steady_clock_;
  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
  rclcpp::Publisher<physicar_interfaces::msg::ObstacleArray>::SharedPtr obstacle_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
  rclcpp::TimerBase::SharedPtr watchdog_;
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr parameter_callback_;
  LidarConfig config_;
  std::chrono::steady_clock::time_point started_at_;
  std::chrono::steady_clock::time_point last_scan_at_;
  double scan_timeout_sec_{1.0};
  int64_t timing_log_period_ms_{2000};
  bool timing_log_enabled_{false};
  bool received_scan_{false};
};

}  // namespace physicar_autonomy

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<physicar_autonomy::LidarObstacleNode>());
  rclcpp::shutdown();
  return 0;
}
