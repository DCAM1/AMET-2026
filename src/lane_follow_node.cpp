// Copyright 2026 AICASTLE Inc.
// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <iomanip>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <physicar_interfaces/msg/obstacle_array.hpp>
#include <rcl_interfaces/msg/parameter_descriptor.hpp>
#include <rcl_interfaces/msg/set_parameters_result.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/float64.hpp>

namespace physicar_autonomy
{

struct RoiConfig
{
  double top_y_ratio{0.54};
  double bottom_y_ratio{0.99};
  double top_left_x_ratio{0.40};
  double top_right_x_ratio{0.60};
  double bottom_left_x_ratio{0.00};
  double bottom_right_x_ratio{1.00};
  int line_thickness{2};
};

struct PerspectiveConfig
{
  int output_width{480};
  int output_height{360};
  double dst_top_y_ratio{0.0};
  double dst_bottom_y_ratio{1.0};
  double dst_top_left_x_ratio{0.0};
  double dst_top_right_x_ratio{1.0};
  double dst_bottom_left_x_ratio{0.0};
  double dst_bottom_right_x_ratio{1.0};
};

struct OrangeMaskConfig
{
  int h_min{10};
  int h_max{28};
  int s_min{90};
  int s_max{255};
  int v_min{80};
  int v_max{255};
  int open_kernel{3};
  int close_kernel{5};
  int morphology_iterations{1};
};

struct WhiteMaskConfig
{
  int saturation_max{80};
  int value_min{160};
  int open_kernel{3};
  int close_kernel{5};
  int morphology_iterations{1};
  int minimum_pixels_per_bin{5};
  double minimum_lane_width_ratio{0.45};
  double maximum_lane_width_ratio{1.55};
};

struct PathDetectionConfig
{
  int polynomial_degree{2};
  int min_component_area_px{12};
  int max_component_area_px{20000};
  int row_bin_height_px{8};
  int min_detection_points{5};
  double fit_max_residual_px{18.0};
  int path_sample_step_px{6};
  double lookahead_distance_m{0.60};
  double bev_forward_range_m{1.50};
  double lane_width_m{0.70};
  double component_chain_max_gap_m{0.45};
  double spline_sample_spacing_m{0.02};
  double minimum_lane_confidence{0.35};
};

struct LaneFitResult
{
  bool valid{false};
  std::vector<double> coefficients;
  std::vector<cv::Point2f> detection_points;
  std::vector<cv::Point2f> component_centers;
  std::vector<cv::Point2f> left_boundary_points;
  std::vector<cv::Point2f> right_boundary_points;
  std::vector<cv::Point> sampled_path;
  std::vector<cv::Point> base_sampled_path;
  cv::Point lookahead_point;
  int component_count{0};
  double residual_rms_px{0.0};
  double vertical_coverage{0.0};
  double confidence{0.0};
  std::string model{"NONE"};
};

struct AvoidanceConfig
{
  bool enabled{true};
  double obstacle_timeout_sec{0.60};
  double minimum_forward_m{0.25};
  double passed_forward_m{-0.25};
  double detection_distance_m{1.50};
  double safety_distance_m{0.14};
  double lateral_offset_m{0.27};
  double offset_rate_mps{0.35};
  double clear_time_sec{0.25};
  double return_distance_m{0.80};
  double association_distance_m{0.30};
  double active_lost_timeout_sec{0.25};
  double avoidance_speed_mps{0.25};
  std::string preferred_side{"left"};
};

struct ControlConfig
{
  bool enabled{false};
  bool adaptive_speed_enabled{true};
  double wheelbase_m{0.18};
  double max_steering_rad{0.349066};
  double max_steering_rate_rad_s{1.50};
  double test_speed_mps{0.40};
  double minimum_speed_mps{0.25};
  double maximum_speed_mps{0.50};
  double steering_slowdown_threshold_rad{0.10};
  double curvature_slowdown_gain{0.80};
  double tracking_lane_confidence{0.45};
  double low_confidence_speed_mps{0.20};
  double previous_path_hold_sec{0.35};
  int lost_frame_threshold{5};
  double minimum_lookahead_m{0.35};
  double maximum_lookahead_m{0.65};
  double lookahead_speed_gain{0.55};
  double minimum_target_distance_m{0.10};
};

class LaneFollowNode : public rclcpp::Node
{
public:
  LaneFollowNode()
  : Node("lane_follow"),
    steady_clock_(RCL_STEADY_TIME),
    started_at_(std::chrono::steady_clock::now()),
    last_frame_at_(started_at_)
  {
    rcl_interfaces::msg::ParameterDescriptor topic_descriptor;
    topic_descriptor.read_only = true;
    topic_descriptor.description = "Restart the node to change a ROS topic name";
    const auto image_topic = declare_parameter<std::string>(
      "image_topic", "/camera/image_raw", topic_descriptor);
    const auto debug_original_topic = declare_parameter<std::string>(
      "debug_original_topic", "/lane/debug/original", topic_descriptor);
    const auto debug_roi_topic = declare_parameter<std::string>(
      "debug_roi_topic", "/lane/debug/roi", topic_descriptor);
    const auto debug_birdseye_topic = declare_parameter<std::string>(
      "debug_birdseye_topic", "/lane/debug/birdseye", topic_descriptor);
    const auto debug_mask_topic = declare_parameter<std::string>(
      "debug_mask_topic", "/lane/debug/mask", topic_descriptor);
    const auto debug_white_mask_topic = declare_parameter<std::string>(
      "debug_white_mask_topic", "/lane/debug/white_mask", topic_descriptor);
    const auto debug_path_topic = declare_parameter<std::string>(
      "debug_path_topic", "/lane/debug/path", topic_descriptor);
    const auto speed_topic = declare_parameter<std::string>(
      "speed_topic", "/speed", topic_descriptor);
    const auto steering_topic = declare_parameter<std::string>(
      "steering_topic", "/steering", topic_descriptor);
    const auto obstacle_topic = declare_parameter<std::string>(
      "obstacle_topic", "/obstacles", topic_descriptor);
    frame_timeout_sec_ = std::max(
      0.1, declare_parameter<double>("frame_timeout_sec", 1.0));
    timing_log_enabled_ = declare_parameter<bool>("timing_log_enabled", false);
    timing_log_period_ms_ = static_cast<int64_t>(1000.0 * std::max(
      0.1, declare_parameter<double>("timing_log_period_sec", 2.0)));

    roi_.top_y_ratio = declare_parameter<double>("roi_top_y_ratio", 0.54);
    roi_.bottom_y_ratio = declare_parameter<double>("roi_bottom_y_ratio", 0.99);
    roi_.top_left_x_ratio = declare_parameter<double>("roi_top_left_x_ratio", 0.40);
    roi_.top_right_x_ratio = declare_parameter<double>("roi_top_right_x_ratio", 0.60);
    roi_.bottom_left_x_ratio = declare_parameter<double>("roi_bottom_left_x_ratio", 0.00);
    roi_.bottom_right_x_ratio = declare_parameter<double>("roi_bottom_right_x_ratio", 1.00);
    roi_.line_thickness = declare_parameter<int>("roi_line_thickness", 2);
    std::string roi_error;
    if (!validate_roi(roi_, roi_error)) {
      throw std::invalid_argument("Invalid ROI parameters: " + roi_error);
    }

    perspective_.output_width = declare_parameter<int>("bev_output_width", 480);
    perspective_.output_height = declare_parameter<int>("bev_output_height", 360);
    perspective_.dst_top_y_ratio = declare_parameter<double>(
      "perspective_dst_top_y_ratio", 0.0);
    perspective_.dst_bottom_y_ratio = declare_parameter<double>(
      "perspective_dst_bottom_y_ratio", 1.0);
    perspective_.dst_top_left_x_ratio = declare_parameter<double>(
      "perspective_dst_top_left_x_ratio", 0.0);
    perspective_.dst_top_right_x_ratio = declare_parameter<double>(
      "perspective_dst_top_right_x_ratio", 1.0);
    perspective_.dst_bottom_left_x_ratio = declare_parameter<double>(
      "perspective_dst_bottom_left_x_ratio", 0.0);
    perspective_.dst_bottom_right_x_ratio = declare_parameter<double>(
      "perspective_dst_bottom_right_x_ratio", 1.0);
    std::string perspective_error;
    if (!validate_perspective(perspective_, perspective_error)) {
      throw std::invalid_argument(
              "Invalid perspective parameters: " + perspective_error);
    }

    orange_mask_.h_min = declare_parameter<int>("orange_h_min", 10);
    orange_mask_.h_max = declare_parameter<int>("orange_h_max", 28);
    orange_mask_.s_min = declare_parameter<int>("orange_s_min", 90);
    orange_mask_.s_max = declare_parameter<int>("orange_s_max", 255);
    orange_mask_.v_min = declare_parameter<int>("orange_v_min", 80);
    orange_mask_.v_max = declare_parameter<int>("orange_v_max", 255);
    orange_mask_.open_kernel = declare_parameter<int>("orange_open_kernel", 3);
    orange_mask_.close_kernel = declare_parameter<int>("orange_close_kernel", 5);
    orange_mask_.morphology_iterations = declare_parameter<int>(
      "orange_morphology_iterations", 1);
    std::string mask_error;
    if (!validate_orange_mask(orange_mask_, mask_error)) {
      throw std::invalid_argument("Invalid orange mask parameters: " + mask_error);
    }
    update_morphology_kernels();

    white_mask_.saturation_max = declare_parameter<int>("white_s_max", 80);
    white_mask_.value_min = declare_parameter<int>("white_v_min", 160);
    white_mask_.open_kernel = declare_parameter<int>("white_open_kernel", 3);
    white_mask_.close_kernel = declare_parameter<int>("white_close_kernel", 5);
    white_mask_.morphology_iterations = declare_parameter<int>(
      "white_morphology_iterations", 1);
    white_mask_.minimum_pixels_per_bin = declare_parameter<int>(
      "white_minimum_pixels_per_bin", 5);
    white_mask_.minimum_lane_width_ratio = declare_parameter<double>(
      "white_minimum_lane_width_ratio", 0.45);
    white_mask_.maximum_lane_width_ratio = declare_parameter<double>(
      "white_maximum_lane_width_ratio", 1.55);
    std::string white_error;
    if (!validate_white_mask(white_mask_, white_error)) {
      throw std::invalid_argument("Invalid white mask parameters: " + white_error);
    }
    update_white_morphology_kernels();

    path_detection_.polynomial_degree = declare_parameter<int>(
      "polynomial_degree", 2);
    path_detection_.min_component_area_px = declare_parameter<int>(
      "min_component_area_px", 12);
    path_detection_.max_component_area_px = declare_parameter<int>(
      "max_component_area_px", 20000);
    path_detection_.row_bin_height_px = declare_parameter<int>(
      "path_row_bin_height_px", 8);
    path_detection_.min_detection_points = declare_parameter<int>(
      "min_detection_points", 5);
    path_detection_.fit_max_residual_px = declare_parameter<double>(
      "fit_max_residual_px", 18.0);
    path_detection_.path_sample_step_px = declare_parameter<int>(
      "path_sample_step_px", 6);
    path_detection_.lookahead_distance_m = declare_parameter<double>(
      "lookahead_distance_m", 0.60);
    path_detection_.bev_forward_range_m = declare_parameter<double>(
      "bev_forward_range_m", 1.50);
    path_detection_.lane_width_m = declare_parameter<double>(
      "lane_width_m", 0.70);
    path_detection_.component_chain_max_gap_m = declare_parameter<double>(
      "component_chain_max_gap_m", 0.45);
    path_detection_.spline_sample_spacing_m = declare_parameter<double>(
      "spline_sample_spacing_m", 0.02);
    path_detection_.minimum_lane_confidence = declare_parameter<double>(
      "minimum_lane_confidence", 0.35);
    std::string path_error;
    if (!validate_path_detection(path_detection_, path_error)) {
      throw std::invalid_argument("Invalid path detection parameters: " + path_error);
    }

    control_.enabled = declare_parameter<bool>("control_enabled", false);
    control_.adaptive_speed_enabled = declare_parameter<bool>(
      "adaptive_speed_enabled", true);
    control_.wheelbase_m = declare_parameter<double>("wheelbase_m", 0.18);
    control_.max_steering_rad = declare_parameter<double>("max_steering_rad", 0.349066);
    control_.max_steering_rate_rad_s = declare_parameter<double>(
      "max_steering_rate_rad_s", 1.50);
    control_.test_speed_mps = declare_parameter<double>("test_speed_mps", 0.40);
    control_.minimum_speed_mps = declare_parameter<double>("min_speed_mps", 0.25);
    control_.maximum_speed_mps = declare_parameter<double>("max_speed_mps", 0.50);
    control_.steering_slowdown_threshold_rad = declare_parameter<double>(
      "steering_slowdown_threshold_rad", 0.10);
    control_.curvature_slowdown_gain = declare_parameter<double>(
      "curvature_slowdown_gain", 0.80);
    control_.tracking_lane_confidence = declare_parameter<double>(
      "tracking_lane_confidence", 0.45);
    control_.low_confidence_speed_mps = declare_parameter<double>(
      "low_confidence_speed_mps", 0.20);
    control_.previous_path_hold_sec = declare_parameter<double>(
      "previous_path_hold_sec", 0.35);
    control_.lost_frame_threshold = declare_parameter<int>("lost_frame_threshold", 5);
    control_.minimum_lookahead_m = declare_parameter<double>(
      "min_lookahead_distance_m", 0.35);
    control_.maximum_lookahead_m = declare_parameter<double>(
      "max_lookahead_distance_m", 0.65);
    control_.lookahead_speed_gain = declare_parameter<double>(
      "lookahead_speed_gain", 0.55);
    control_.minimum_target_distance_m = declare_parameter<double>(
      "minimum_target_distance_m", 0.10);
    std::string control_error;
    if (!validate_control(control_, control_error)) {
      throw std::invalid_argument("Invalid control parameters: " + control_error);
    }
    if (control_.tracking_lane_confidence < path_detection_.minimum_lane_confidence ||
      control_.maximum_lookahead_m > path_detection_.bev_forward_range_m)
    {
      throw std::invalid_argument(
              "Control confidence/lookahead bounds conflict with path parameters");
    }

    avoidance_.enabled = declare_parameter<bool>("obstacle_avoidance_enabled", true);
    avoidance_.obstacle_timeout_sec = declare_parameter<double>(
      "obstacle_timeout_sec", 0.60);
    avoidance_.minimum_forward_m = declare_parameter<double>(
      "obstacle_min_forward_m", 0.25);
    avoidance_.passed_forward_m = declare_parameter<double>(
      "obstacle_passed_forward_m", -0.25);
    avoidance_.detection_distance_m = declare_parameter<double>(
      "obstacle_detection_distance_m", 1.50);
    avoidance_.safety_distance_m = declare_parameter<double>(
      "obstacle_safety_distance_m", 0.14);
    avoidance_.lateral_offset_m = declare_parameter<double>(
      "avoidance_lateral_offset_m", 0.27);
    avoidance_.offset_rate_mps = declare_parameter<double>(
      "avoidance_offset_rate_mps", 0.35);
    avoidance_.clear_time_sec = declare_parameter<double>(
      "obstacle_clear_time_sec", 0.25);
    avoidance_.return_distance_m = declare_parameter<double>(
      "avoidance_return_distance_m", 0.80);
    avoidance_.association_distance_m = declare_parameter<double>(
      "obstacle_association_distance_m", 0.30);
    avoidance_.active_lost_timeout_sec = declare_parameter<double>(
      "obstacle_active_lost_timeout_sec", 0.25);
    avoidance_.avoidance_speed_mps = declare_parameter<double>(
      "avoidance_speed_mps", 0.25);
    avoidance_.preferred_side = declare_parameter<std::string>(
      "preferred_avoidance_side", "left");
    std::string avoidance_error;
    if (!validate_avoidance(avoidance_, path_detection_, avoidance_error)) {
      throw std::invalid_argument("Invalid obstacle avoidance parameters: " + avoidance_error);
    }

    parameter_callback_ = add_on_set_parameters_callback(
      std::bind(&LaneFollowNode::on_parameters, this, std::placeholders::_1));

    // Camera data is time-sensitive. A depth-one best-effort queue prevents
    // processing an old backlog and remains compatible with both the real
    // camera and the simulator's reliable publisher.
    const auto sensor_qos = rclcpp::QoS(rclcpp::KeepLast(1))
      .best_effort()
      .durability_volatile();
    // RViz and rqt_image_view commonly request reliable image delivery. Keep
    // only the newest debug frame, but use reliable delivery for viewer
    // compatibility. This publisher is not part of the control path.
    const auto debug_qos = rclcpp::QoS(rclcpp::KeepLast(1))
      .reliable()
      .durability_volatile();

    debug_original_pub_ = create_publisher<sensor_msgs::msg::Image>(
      debug_original_topic, debug_qos);
    debug_roi_pub_ = create_publisher<sensor_msgs::msg::Image>(
      debug_roi_topic, debug_qos);
    debug_birdseye_pub_ = create_publisher<sensor_msgs::msg::Image>(
      debug_birdseye_topic, debug_qos);
    debug_mask_pub_ = create_publisher<sensor_msgs::msg::Image>(
      debug_mask_topic, debug_qos);
    debug_white_mask_pub_ = create_publisher<sensor_msgs::msg::Image>(
      debug_white_mask_topic, debug_qos);
    debug_path_pub_ = create_publisher<sensor_msgs::msg::Image>(
      debug_path_topic, debug_qos);
    speed_pub_ = create_publisher<std_msgs::msg::Float64>(speed_topic, 10);
    steering_pub_ = create_publisher<std_msgs::msg::Float64>(steering_topic, 10);
    image_sub_ = create_subscription<sensor_msgs::msg::Image>(
      image_topic,
      sensor_qos,
      std::bind(&LaneFollowNode::on_image, this, std::placeholders::_1));
    obstacle_sub_ = create_subscription<physicar_interfaces::msg::ObstacleArray>(
      obstacle_topic, rclcpp::QoS(rclcpp::KeepLast(1)).reliable().durability_volatile(),
      std::bind(&LaneFollowNode::on_obstacles, this, std::placeholders::_1));

    frame_watchdog_ = create_wall_timer(
      std::chrono::milliseconds(250),
      std::bind(&LaneFollowNode::check_frame_timeout, this));

    RCLCPP_INFO(
      get_logger(),
      "Phase 10 ready: %s -> {%s, %s, %s, %s, %s, %s}; control=%s (%s, %s)",
      image_topic.c_str(), debug_original_topic.c_str(), debug_roi_topic.c_str(),
      debug_birdseye_topic.c_str(), debug_mask_topic.c_str(), debug_white_mask_topic.c_str(),
      debug_path_topic.c_str(),
      control_.enabled ? "ENABLED" : "disabled", speed_topic.c_str(), steering_topic.c_str());
  }

private:
  static bool validate_roi(const RoiConfig & roi, std::string & reason)
  {
    const std::array<double, 6> ratios = {
      roi.top_y_ratio,
      roi.bottom_y_ratio,
      roi.top_left_x_ratio,
      roi.top_right_x_ratio,
      roi.bottom_left_x_ratio,
      roi.bottom_right_x_ratio,
    };
    if (std::any_of(ratios.begin(), ratios.end(), [](double value) {
        return value < 0.0 || value > 1.0;
      }))
    {
      reason = "all ROI ratios must be within [0.0, 1.0]";
      return false;
    }
    if (roi.top_y_ratio >= roi.bottom_y_ratio) {
      reason = "roi_top_y_ratio must be smaller than roi_bottom_y_ratio";
      return false;
    }
    if (roi.top_left_x_ratio >= roi.top_right_x_ratio) {
      reason = "top-left x must be smaller than top-right x";
      return false;
    }
    if (roi.bottom_left_x_ratio >= roi.bottom_right_x_ratio) {
      reason = "bottom-left x must be smaller than bottom-right x";
      return false;
    }
    if (roi.line_thickness < 1 || roi.line_thickness > 10) {
      reason = "roi_line_thickness must be between 1 and 10";
      return false;
    }
    return true;
  }

  static bool validate_perspective(
    const PerspectiveConfig & perspective, std::string & reason)
  {
    if (perspective.output_width < 32 || perspective.output_width > 1920 ||
      perspective.output_height < 32 || perspective.output_height > 1080)
    {
      reason = "BEV output dimensions must be within 32x32 and 1920x1080";
      return false;
    }
    const std::array<double, 6> ratios = {
      perspective.dst_top_y_ratio,
      perspective.dst_bottom_y_ratio,
      perspective.dst_top_left_x_ratio,
      perspective.dst_top_right_x_ratio,
      perspective.dst_bottom_left_x_ratio,
      perspective.dst_bottom_right_x_ratio,
    };
    if (std::any_of(ratios.begin(), ratios.end(), [](double value) {
        return value < 0.0 || value > 1.0;
      }))
    {
      reason = "all perspective destination ratios must be within [0.0, 1.0]";
      return false;
    }
    if (perspective.dst_top_y_ratio >= perspective.dst_bottom_y_ratio) {
      reason = "destination top y must be smaller than destination bottom y";
      return false;
    }
    if (perspective.dst_top_left_x_ratio >= perspective.dst_top_right_x_ratio ||
      perspective.dst_bottom_left_x_ratio >= perspective.dst_bottom_right_x_ratio)
    {
      reason = "destination left x must be smaller than destination right x";
      return false;
    }
    return true;
  }

  static bool validate_orange_mask(
    const OrangeMaskConfig & mask, std::string & reason)
  {
    if (mask.h_min < 0 || mask.h_max > 179 || mask.h_min > mask.h_max) {
      reason = "orange hue must satisfy 0 <= h_min <= h_max <= 179";
      return false;
    }
    if (mask.s_min < 0 || mask.s_max > 255 || mask.s_min > mask.s_max ||
      mask.v_min < 0 || mask.v_max > 255 || mask.v_min > mask.v_max)
    {
      reason = "orange saturation/value bounds must be ordered within [0, 255]";
      return false;
    }
    const auto valid_kernel = [](int size) {
        return size >= 1 && size <= 31 && (size % 2) == 1;
      };
    if (!valid_kernel(mask.open_kernel) || !valid_kernel(mask.close_kernel)) {
      reason = "morphology kernel sizes must be odd values within [1, 31]";
      return false;
    }
    if (mask.morphology_iterations < 1 || mask.morphology_iterations > 5) {
      reason = "orange_morphology_iterations must be within [1, 5]";
      return false;
    }
    return true;
  }

  static bool validate_white_mask(const WhiteMaskConfig & mask, std::string & reason)
  {
    if (mask.saturation_max < 0 || mask.saturation_max > 255 ||
      mask.value_min < 0 || mask.value_min > 255)
    {
      reason = "white S/V thresholds must be within [0, 255]";
      return false;
    }
    const auto valid_kernel = [](int size) {
        return size >= 1 && size <= 31 && (size % 2) == 1;
      };
    if (!valid_kernel(mask.open_kernel) || !valid_kernel(mask.close_kernel)) {
      reason = "white morphology kernels must be odd values within [1, 31]";
      return false;
    }
    if (mask.morphology_iterations < 1 || mask.morphology_iterations > 5) {
      reason = "white_morphology_iterations must be within [1, 5]";
      return false;
    }
    if (mask.minimum_pixels_per_bin < 1 || mask.minimum_pixels_per_bin > 1000) {
      reason = "white_minimum_pixels_per_bin must be within [1, 1000]";
      return false;
    }
    if (mask.minimum_lane_width_ratio <= 0.1 ||
      mask.maximum_lane_width_ratio <= mask.minimum_lane_width_ratio ||
      mask.maximum_lane_width_ratio > 3.0)
    {
      reason = "white lane-width ratios must satisfy 0.1 < min < max <= 3.0";
      return false;
    }
    return true;
  }

  static bool validate_path_detection(
    const PathDetectionConfig & path, std::string & reason)
  {
    if (path.polynomial_degree < 1 || path.polynomial_degree > 3) {
      reason = "polynomial_degree must be within [1, 3]";
      return false;
    }
    if (path.min_component_area_px < 1 ||
      path.max_component_area_px <= path.min_component_area_px)
    {
      reason = "component areas must satisfy 1 <= min < max";
      return false;
    }
    if (path.row_bin_height_px < 2 || path.row_bin_height_px > 64) {
      reason = "path_row_bin_height_px must be within [2, 64]";
      return false;
    }
    if (path.min_detection_points < path.polynomial_degree + 1 ||
      path.min_detection_points > 100)
    {
      reason = "min_detection_points must exceed polynomial degree and be <= 100";
      return false;
    }
    if (path.fit_max_residual_px < 1.0 || path.fit_max_residual_px > 100.0) {
      reason = "fit_max_residual_px must be within [1, 100]";
      return false;
    }
    if (path.path_sample_step_px < 1 || path.path_sample_step_px > 50) {
      reason = "path_sample_step_px must be within [1, 50]";
      return false;
    }
    if (path.bev_forward_range_m <= 0.05 || path.bev_forward_range_m > 20.0) {
      reason = "bev_forward_range_m must be within (0.05, 20]";
      return false;
    }
    if (path.lookahead_distance_m <= 0.0 ||
      path.lookahead_distance_m > path.bev_forward_range_m)
    {
      reason = "lookahead_distance_m must be within (0, bev_forward_range_m]";
      return false;
    }
    if (path.lane_width_m < 0.1 || path.lane_width_m > 5.0) {
      reason = "lane_width_m must be within [0.1, 5.0]";
      return false;
    }
    if (path.component_chain_max_gap_m < 0.05 ||
      path.component_chain_max_gap_m > 2.0)
    {
      reason = "component_chain_max_gap_m must be within [0.05, 2.0]";
      return false;
    }
    if (path.spline_sample_spacing_m < 0.005 ||
      path.spline_sample_spacing_m > 0.2)
    {
      reason = "spline_sample_spacing_m must be within [0.005, 0.2]";
      return false;
    }
    if (path.minimum_lane_confidence < 0.0 || path.minimum_lane_confidence > 1.0) {
      reason = "minimum_lane_confidence must be within [0, 1]";
      return false;
    }
    return true;
  }

  static bool validate_control(const ControlConfig & control, std::string & reason)
  {
    if (control.wheelbase_m < 0.05 || control.wheelbase_m > 2.0) {
      reason = "wheelbase_m must be within [0.05, 2.0]";
      return false;
    }
    if (control.max_steering_rad <= 0.0 || control.max_steering_rad > 0.349066) {
      reason = "max_steering_rad must be within (0, 0.349066] (20 degrees)";
      return false;
    }
    if (control.max_steering_rate_rad_s <= 0.0 ||
      control.max_steering_rate_rad_s > 20.0)
    {
      reason = "max_steering_rate_rad_s must be within (0, 20]";
      return false;
    }
    if (control.test_speed_mps < 0.0 || control.test_speed_mps > 3.0) {
      reason = "test_speed_mps must be within [0, 3.0]";
      return false;
    }
    if (control.minimum_speed_mps < 0.0 ||
      control.maximum_speed_mps < control.minimum_speed_mps ||
      control.maximum_speed_mps > 3.0)
    {
      reason = "speed bounds must satisfy 0 <= min_speed_mps <= max_speed_mps <= 3.0";
      return false;
    }
    if (control.steering_slowdown_threshold_rad < 0.0 ||
      control.steering_slowdown_threshold_rad >= control.max_steering_rad)
    {
      reason = "steering_slowdown_threshold_rad must be within [0, max_steering_rad)";
      return false;
    }
    if (control.curvature_slowdown_gain < 0.0 ||
      control.curvature_slowdown_gain > 20.0)
    {
      reason = "curvature_slowdown_gain must be within [0, 20]";
      return false;
    }
    if (control.tracking_lane_confidence < 0.0 ||
      control.tracking_lane_confidence > 1.0)
    {
      reason = "tracking_lane_confidence must be within [0, 1]";
      return false;
    }
    if (control.low_confidence_speed_mps < 0.0 ||
      control.low_confidence_speed_mps > control.maximum_speed_mps)
    {
      reason = "low_confidence_speed_mps must be within [0, max_speed_mps]";
      return false;
    }
    if (control.previous_path_hold_sec < 0.0 || control.previous_path_hold_sec > 2.0) {
      reason = "previous_path_hold_sec must be within [0, 2.0]";
      return false;
    }
    if (control.lost_frame_threshold < 1 || control.lost_frame_threshold > 100) {
      reason = "lost_frame_threshold must be within [1, 100]";
      return false;
    }
    if (control.minimum_lookahead_m <= 0.0 ||
      control.maximum_lookahead_m < control.minimum_lookahead_m ||
      control.maximum_lookahead_m > 5.0)
    {
      reason = "lookahead bounds must satisfy 0 < min <= max <= 5.0";
      return false;
    }
    if (control.lookahead_speed_gain < 0.0 || control.lookahead_speed_gain > 5.0) {
      reason = "lookahead_speed_gain must be within [0, 5.0]";
      return false;
    }
    if (control.minimum_target_distance_m < 0.02 ||
      control.minimum_target_distance_m > 2.0)
    {
      reason = "minimum_target_distance_m must be within [0.02, 2.0]";
      return false;
    }
    return true;
  }

  static bool validate_avoidance(
    const AvoidanceConfig & avoidance,
    const PathDetectionConfig & path,
    std::string & reason)
  {
    if (avoidance.obstacle_timeout_sec < 0.1 || avoidance.obstacle_timeout_sec > 5.0) {
      reason = "obstacle_timeout_sec must be within [0.1, 5.0]";
      return false;
    }
    if (avoidance.minimum_forward_m < 0.0 ||
      avoidance.detection_distance_m <= avoidance.minimum_forward_m ||
      avoidance.detection_distance_m > path.bev_forward_range_m)
    {
      reason = "obstacle forward range must fit within the BEV forward range";
      return false;
    }
    if (avoidance.passed_forward_m < -1.0 ||
      avoidance.passed_forward_m >= avoidance.minimum_forward_m)
    {
      reason = "obstacle_passed_forward_m must be >= -1.0 and below obstacle_min_forward_m";
      return false;
    }
    if (avoidance.safety_distance_m < 0.05 || avoidance.safety_distance_m > 1.0) {
      reason = "obstacle_safety_distance_m must be within [0.05, 1.0]";
      return false;
    }
    const double maximum_lane_offset = 0.5 * path.lane_width_m - 0.05;
    if (avoidance.lateral_offset_m <= 0.0 ||
      avoidance.lateral_offset_m > maximum_lane_offset)
    {
      reason = "avoidance_lateral_offset_m exceeds the configured lane half-width";
      return false;
    }
    if (avoidance.offset_rate_mps <= 0.0 || avoidance.offset_rate_mps > 2.0) {
      reason = "avoidance_offset_rate_mps must be within (0, 2.0]";
      return false;
    }
    if (avoidance.clear_time_sec < 0.0 || avoidance.clear_time_sec > 5.0 ||
      avoidance.return_distance_m < 0.1 || avoidance.return_distance_m > 5.0)
    {
      reason = "avoidance clear time/return distance is outside safe bounds";
      return false;
    }
    if (avoidance.association_distance_m < 0.05 ||
      avoidance.association_distance_m > 1.0 ||
      avoidance.active_lost_timeout_sec < 0.05 ||
      avoidance.active_lost_timeout_sec > 2.0)
    {
      reason = "obstacle association distance/lost timeout is outside safe bounds";
      return false;
    }
    if (avoidance.avoidance_speed_mps <= 0.0 || avoidance.avoidance_speed_mps > 3.0) {
      reason = "avoidance_speed_mps must be within (0, 3.0]";
      return false;
    }
    if (avoidance.preferred_side != "left" && avoidance.preferred_side != "right") {
      reason = "preferred_avoidance_side must be 'left' or 'right'";
      return false;
    }
    return true;
  }

  void update_morphology_kernels()
  {
    open_kernel_ = cv::getStructuringElement(
      cv::MORPH_ELLIPSE,
      cv::Size(orange_mask_.open_kernel, orange_mask_.open_kernel));
    close_kernel_ = cv::getStructuringElement(
      cv::MORPH_ELLIPSE,
      cv::Size(orange_mask_.close_kernel, orange_mask_.close_kernel));
    morphology_kernels_dirty_ = false;
  }

  void update_white_morphology_kernels()
  {
    white_open_kernel_ = cv::getStructuringElement(
      cv::MORPH_ELLIPSE, cv::Size(white_mask_.open_kernel, white_mask_.open_kernel));
    white_close_kernel_ = cv::getStructuringElement(
      cv::MORPH_ELLIPSE, cv::Size(white_mask_.close_kernel, white_mask_.close_kernel));
    white_morphology_kernels_dirty_ = false;
  }

  rcl_interfaces::msg::SetParametersResult on_parameters(
    const std::vector<rclcpp::Parameter> & parameters)
  {
    rcl_interfaces::msg::SetParametersResult result;
    result.successful = true;
    RoiConfig candidate_roi = roi_;
    PerspectiveConfig candidate_perspective = perspective_;
    OrangeMaskConfig candidate_orange_mask = orange_mask_;
    WhiteMaskConfig candidate_white_mask = white_mask_;
    PathDetectionConfig candidate_path_detection = path_detection_;
    ControlConfig candidate_control = control_;
    AvoidanceConfig candidate_avoidance = avoidance_;
    double candidate_frame_timeout_sec = frame_timeout_sec_;
    bool candidate_timing_log_enabled = timing_log_enabled_;
    int64_t candidate_timing_log_period_ms = timing_log_period_ms_;

    for (const auto & parameter : parameters) {
      if (parameter.get_name() == "frame_timeout_sec") {
        if (parameter.as_double() < 0.1) {
          result.successful = false;
          result.reason = "frame_timeout_sec must be at least 0.1";
          return result;
        }
        candidate_frame_timeout_sec = parameter.as_double();
      } else if (parameter.get_name() == "timing_log_enabled") {
        candidate_timing_log_enabled = parameter.as_bool();
      } else if (parameter.get_name() == "timing_log_period_sec") {
        if (parameter.as_double() < 0.1) {
          result.successful = false;
          result.reason = "timing_log_period_sec must be at least 0.1";
          return result;
        }
        candidate_timing_log_period_ms = static_cast<int64_t>(
          parameter.as_double() * 1000.0);
      } else if (parameter.get_name() == "roi_top_y_ratio") {
        candidate_roi.top_y_ratio = parameter.as_double();
      } else if (parameter.get_name() == "roi_bottom_y_ratio") {
        candidate_roi.bottom_y_ratio = parameter.as_double();
      } else if (parameter.get_name() == "roi_top_left_x_ratio") {
        candidate_roi.top_left_x_ratio = parameter.as_double();
      } else if (parameter.get_name() == "roi_top_right_x_ratio") {
        candidate_roi.top_right_x_ratio = parameter.as_double();
      } else if (parameter.get_name() == "roi_bottom_left_x_ratio") {
        candidate_roi.bottom_left_x_ratio = parameter.as_double();
      } else if (parameter.get_name() == "roi_bottom_right_x_ratio") {
        candidate_roi.bottom_right_x_ratio = parameter.as_double();
      } else if (parameter.get_name() == "roi_line_thickness") {
        candidate_roi.line_thickness = static_cast<int>(parameter.as_int());
      } else if (parameter.get_name() == "bev_output_width") {
        candidate_perspective.output_width = static_cast<int>(parameter.as_int());
      } else if (parameter.get_name() == "bev_output_height") {
        candidate_perspective.output_height = static_cast<int>(parameter.as_int());
      } else if (parameter.get_name() == "perspective_dst_top_y_ratio") {
        candidate_perspective.dst_top_y_ratio = parameter.as_double();
      } else if (parameter.get_name() == "perspective_dst_bottom_y_ratio") {
        candidate_perspective.dst_bottom_y_ratio = parameter.as_double();
      } else if (parameter.get_name() == "perspective_dst_top_left_x_ratio") {
        candidate_perspective.dst_top_left_x_ratio = parameter.as_double();
      } else if (parameter.get_name() == "perspective_dst_top_right_x_ratio") {
        candidate_perspective.dst_top_right_x_ratio = parameter.as_double();
      } else if (parameter.get_name() == "perspective_dst_bottom_left_x_ratio") {
        candidate_perspective.dst_bottom_left_x_ratio = parameter.as_double();
      } else if (parameter.get_name() == "perspective_dst_bottom_right_x_ratio") {
        candidate_perspective.dst_bottom_right_x_ratio = parameter.as_double();
      } else if (parameter.get_name() == "orange_h_min") {
        candidate_orange_mask.h_min = static_cast<int>(parameter.as_int());
      } else if (parameter.get_name() == "orange_h_max") {
        candidate_orange_mask.h_max = static_cast<int>(parameter.as_int());
      } else if (parameter.get_name() == "orange_s_min") {
        candidate_orange_mask.s_min = static_cast<int>(parameter.as_int());
      } else if (parameter.get_name() == "orange_s_max") {
        candidate_orange_mask.s_max = static_cast<int>(parameter.as_int());
      } else if (parameter.get_name() == "orange_v_min") {
        candidate_orange_mask.v_min = static_cast<int>(parameter.as_int());
      } else if (parameter.get_name() == "orange_v_max") {
        candidate_orange_mask.v_max = static_cast<int>(parameter.as_int());
      } else if (parameter.get_name() == "orange_open_kernel") {
        candidate_orange_mask.open_kernel = static_cast<int>(parameter.as_int());
      } else if (parameter.get_name() == "orange_close_kernel") {
        candidate_orange_mask.close_kernel = static_cast<int>(parameter.as_int());
      } else if (parameter.get_name() == "orange_morphology_iterations") {
        candidate_orange_mask.morphology_iterations = static_cast<int>(parameter.as_int());
      } else if (parameter.get_name() == "white_s_max") {
        candidate_white_mask.saturation_max = static_cast<int>(parameter.as_int());
      } else if (parameter.get_name() == "white_v_min") {
        candidate_white_mask.value_min = static_cast<int>(parameter.as_int());
      } else if (parameter.get_name() == "white_open_kernel") {
        candidate_white_mask.open_kernel = static_cast<int>(parameter.as_int());
      } else if (parameter.get_name() == "white_close_kernel") {
        candidate_white_mask.close_kernel = static_cast<int>(parameter.as_int());
      } else if (parameter.get_name() == "white_morphology_iterations") {
        candidate_white_mask.morphology_iterations = static_cast<int>(parameter.as_int());
      } else if (parameter.get_name() == "white_minimum_pixels_per_bin") {
        candidate_white_mask.minimum_pixels_per_bin = static_cast<int>(parameter.as_int());
      } else if (parameter.get_name() == "white_minimum_lane_width_ratio") {
        candidate_white_mask.minimum_lane_width_ratio = parameter.as_double();
      } else if (parameter.get_name() == "white_maximum_lane_width_ratio") {
        candidate_white_mask.maximum_lane_width_ratio = parameter.as_double();
      } else if (parameter.get_name() == "polynomial_degree") {
        candidate_path_detection.polynomial_degree = static_cast<int>(parameter.as_int());
      } else if (parameter.get_name() == "min_component_area_px") {
        candidate_path_detection.min_component_area_px = static_cast<int>(parameter.as_int());
      } else if (parameter.get_name() == "max_component_area_px") {
        candidate_path_detection.max_component_area_px = static_cast<int>(parameter.as_int());
      } else if (parameter.get_name() == "path_row_bin_height_px") {
        candidate_path_detection.row_bin_height_px = static_cast<int>(parameter.as_int());
      } else if (parameter.get_name() == "min_detection_points") {
        candidate_path_detection.min_detection_points = static_cast<int>(parameter.as_int());
      } else if (parameter.get_name() == "fit_max_residual_px") {
        candidate_path_detection.fit_max_residual_px = parameter.as_double();
      } else if (parameter.get_name() == "path_sample_step_px") {
        candidate_path_detection.path_sample_step_px = static_cast<int>(parameter.as_int());
      } else if (parameter.get_name() == "lookahead_distance_m") {
        candidate_path_detection.lookahead_distance_m = parameter.as_double();
      } else if (parameter.get_name() == "bev_forward_range_m") {
        candidate_path_detection.bev_forward_range_m = parameter.as_double();
      } else if (parameter.get_name() == "lane_width_m") {
        candidate_path_detection.lane_width_m = parameter.as_double();
      } else if (parameter.get_name() == "component_chain_max_gap_m") {
        candidate_path_detection.component_chain_max_gap_m = parameter.as_double();
      } else if (parameter.get_name() == "spline_sample_spacing_m") {
        candidate_path_detection.spline_sample_spacing_m = parameter.as_double();
      } else if (parameter.get_name() == "minimum_lane_confidence") {
        candidate_path_detection.minimum_lane_confidence = parameter.as_double();
      } else if (parameter.get_name() == "control_enabled") {
        candidate_control.enabled = parameter.as_bool();
      } else if (parameter.get_name() == "adaptive_speed_enabled") {
        candidate_control.adaptive_speed_enabled = parameter.as_bool();
      } else if (parameter.get_name() == "wheelbase_m") {
        candidate_control.wheelbase_m = parameter.as_double();
      } else if (parameter.get_name() == "max_steering_rad") {
        candidate_control.max_steering_rad = parameter.as_double();
      } else if (parameter.get_name() == "max_steering_rate_rad_s") {
        candidate_control.max_steering_rate_rad_s = parameter.as_double();
      } else if (parameter.get_name() == "test_speed_mps") {
        candidate_control.test_speed_mps = parameter.as_double();
      } else if (parameter.get_name() == "min_speed_mps") {
        candidate_control.minimum_speed_mps = parameter.as_double();
      } else if (parameter.get_name() == "max_speed_mps") {
        candidate_control.maximum_speed_mps = parameter.as_double();
      } else if (parameter.get_name() == "steering_slowdown_threshold_rad") {
        candidate_control.steering_slowdown_threshold_rad = parameter.as_double();
      } else if (parameter.get_name() == "curvature_slowdown_gain") {
        candidate_control.curvature_slowdown_gain = parameter.as_double();
      } else if (parameter.get_name() == "tracking_lane_confidence") {
        candidate_control.tracking_lane_confidence = parameter.as_double();
      } else if (parameter.get_name() == "low_confidence_speed_mps") {
        candidate_control.low_confidence_speed_mps = parameter.as_double();
      } else if (parameter.get_name() == "previous_path_hold_sec") {
        candidate_control.previous_path_hold_sec = parameter.as_double();
      } else if (parameter.get_name() == "lost_frame_threshold") {
        candidate_control.lost_frame_threshold = static_cast<int>(parameter.as_int());
      } else if (parameter.get_name() == "min_lookahead_distance_m") {
        candidate_control.minimum_lookahead_m = parameter.as_double();
      } else if (parameter.get_name() == "max_lookahead_distance_m") {
        candidate_control.maximum_lookahead_m = parameter.as_double();
      } else if (parameter.get_name() == "lookahead_speed_gain") {
        candidate_control.lookahead_speed_gain = parameter.as_double();
      } else if (parameter.get_name() == "minimum_target_distance_m") {
        candidate_control.minimum_target_distance_m = parameter.as_double();
      } else if (parameter.get_name() == "obstacle_avoidance_enabled") {
        candidate_avoidance.enabled = parameter.as_bool();
      } else if (parameter.get_name() == "obstacle_timeout_sec") {
        candidate_avoidance.obstacle_timeout_sec = parameter.as_double();
      } else if (parameter.get_name() == "obstacle_min_forward_m") {
        candidate_avoidance.minimum_forward_m = parameter.as_double();
      } else if (parameter.get_name() == "obstacle_passed_forward_m") {
        candidate_avoidance.passed_forward_m = parameter.as_double();
      } else if (parameter.get_name() == "obstacle_detection_distance_m") {
        candidate_avoidance.detection_distance_m = parameter.as_double();
      } else if (parameter.get_name() == "obstacle_safety_distance_m") {
        candidate_avoidance.safety_distance_m = parameter.as_double();
      } else if (parameter.get_name() == "avoidance_lateral_offset_m") {
        candidate_avoidance.lateral_offset_m = parameter.as_double();
      } else if (parameter.get_name() == "avoidance_offset_rate_mps") {
        candidate_avoidance.offset_rate_mps = parameter.as_double();
      } else if (parameter.get_name() == "obstacle_clear_time_sec") {
        candidate_avoidance.clear_time_sec = parameter.as_double();
      } else if (parameter.get_name() == "avoidance_return_distance_m") {
        candidate_avoidance.return_distance_m = parameter.as_double();
      } else if (parameter.get_name() == "obstacle_association_distance_m") {
        candidate_avoidance.association_distance_m = parameter.as_double();
      } else if (parameter.get_name() == "obstacle_active_lost_timeout_sec") {
        candidate_avoidance.active_lost_timeout_sec = parameter.as_double();
      } else if (parameter.get_name() == "avoidance_speed_mps") {
        candidate_avoidance.avoidance_speed_mps = parameter.as_double();
      } else if (parameter.get_name() == "preferred_avoidance_side") {
        candidate_avoidance.preferred_side = parameter.as_string();
      }
    }

    std::string roi_error;
    if (!validate_roi(candidate_roi, roi_error)) {
      result.successful = false;
      result.reason = roi_error;
      return result;
    }
    std::string perspective_error;
    if (!validate_perspective(candidate_perspective, perspective_error)) {
      result.successful = false;
      result.reason = perspective_error;
      return result;
    }
    std::string mask_error;
    if (!validate_orange_mask(candidate_orange_mask, mask_error)) {
      result.successful = false;
      result.reason = mask_error;
      return result;
    }
    std::string white_error;
    if (!validate_white_mask(candidate_white_mask, white_error)) {
      result.successful = false;
      result.reason = white_error;
      return result;
    }
    std::string path_error;
    if (!validate_path_detection(candidate_path_detection, path_error)) {
      result.successful = false;
      result.reason = path_error;
      return result;
    }
    std::string control_error;
    if (!validate_control(candidate_control, control_error)) {
      result.successful = false;
      result.reason = control_error;
      return result;
    }
    if (candidate_control.tracking_lane_confidence <
      candidate_path_detection.minimum_lane_confidence)
    {
      result.successful = false;
      result.reason = "tracking_lane_confidence must be >= minimum_lane_confidence";
      return result;
    }
    if (candidate_control.maximum_lookahead_m >
      candidate_path_detection.bev_forward_range_m)
    {
      result.successful = false;
      result.reason = "max_lookahead_distance_m must be <= bev_forward_range_m";
      return result;
    }
    std::string avoidance_error;
    if (!validate_avoidance(candidate_avoidance, candidate_path_detection, avoidance_error)) {
      result.successful = false;
      result.reason = avoidance_error;
      return result;
    }
    roi_ = candidate_roi;
    perspective_ = candidate_perspective;
    perspective_dirty_ = true;
    if (candidate_orange_mask.open_kernel != orange_mask_.open_kernel ||
      candidate_orange_mask.close_kernel != orange_mask_.close_kernel)
    {
      morphology_kernels_dirty_ = true;
    }
    if (candidate_white_mask.open_kernel != white_mask_.open_kernel ||
      candidate_white_mask.close_kernel != white_mask_.close_kernel)
    {
      white_morphology_kernels_dirty_ = true;
    }
    orange_mask_ = candidate_orange_mask;
    white_mask_ = candidate_white_mask;
    path_detection_ = candidate_path_detection;
    const bool control_was_disabled = !control_.enabled && candidate_control.enabled;
    const bool control_was_enabled = control_.enabled && !candidate_control.enabled;
    control_ = candidate_control;
    avoidance_ = candidate_avoidance;
    if (control_was_disabled) {
      last_steering_command_rad_ = 0.0;
      last_control_at_ = std::chrono::steady_clock::now();
      has_reliable_control_ = false;
      lost_frame_count_ = 0;
      RCLCPP_WARN(get_logger(), "Vehicle control ENABLED by parameter change");
    } else if (control_was_enabled) {
      publish_control(0.0, 0.0);
      controller_state_ = "DISABLED";
      RCLCPP_INFO(get_logger(), "Vehicle control disabled; stop command published");
    }
    frame_timeout_sec_ = candidate_frame_timeout_sec;
    timing_log_enabled_ = candidate_timing_log_enabled;
    timing_log_period_ms_ = candidate_timing_log_period_ms;
    return result;
  }

  std::array<cv::Point, 4> roi_points(uint32_t width, uint32_t height) const
  {
    const auto pixel_x = [width](double ratio) {
        return static_cast<int>(std::lround(ratio * static_cast<double>(width - 1U)));
      };
    const auto pixel_y = [height](double ratio) {
        return static_cast<int>(std::lround(ratio * static_cast<double>(height - 1U)));
      };
    return {
      cv::Point(pixel_x(roi_.top_left_x_ratio), pixel_y(roi_.top_y_ratio)),
      cv::Point(pixel_x(roi_.top_right_x_ratio), pixel_y(roi_.top_y_ratio)),
      cv::Point(pixel_x(roi_.bottom_right_x_ratio), pixel_y(roi_.bottom_y_ratio)),
      cv::Point(pixel_x(roi_.bottom_left_x_ratio), pixel_y(roi_.bottom_y_ratio)),
    };
  }

  void update_perspective_transform(uint32_t source_width, uint32_t source_height)
  {
    if (!perspective_dirty_ && source_width == transform_source_width_ &&
      source_height == transform_source_height_)
    {
      return;
    }

    const auto source_pixels = roi_points(source_width, source_height);
    std::array<cv::Point2f, 4> source_points;
    for (std::size_t index = 0; index < source_pixels.size(); ++index) {
      source_points[index] = cv::Point2f(
        static_cast<float>(source_pixels[index].x),
        static_cast<float>(source_pixels[index].y));
    }

    const auto dst_x = [this](double ratio) {
        return static_cast<float>(
          ratio * static_cast<double>(perspective_.output_width - 1));
      };
    const auto dst_y = [this](double ratio) {
        return static_cast<float>(
          ratio * static_cast<double>(perspective_.output_height - 1));
      };
    const std::array<cv::Point2f, 4> destination_points = {
      cv::Point2f(
        dst_x(perspective_.dst_top_left_x_ratio),
        dst_y(perspective_.dst_top_y_ratio)),
      cv::Point2f(
        dst_x(perspective_.dst_top_right_x_ratio),
        dst_y(perspective_.dst_top_y_ratio)),
      cv::Point2f(
        dst_x(perspective_.dst_bottom_right_x_ratio),
        dst_y(perspective_.dst_bottom_y_ratio)),
      cv::Point2f(
        dst_x(perspective_.dst_bottom_left_x_ratio),
        dst_y(perspective_.dst_bottom_y_ratio)),
    };

    perspective_matrix_ = cv::getPerspectiveTransform(
      source_points.data(), destination_points.data());
    transform_source_width_ = source_width;
    transform_source_height_ = source_height;
    perspective_dirty_ = false;
  }

  bool fit_polynomial(
    const std::vector<cv::Point2f> & points,
    int image_height,
    std::vector<double> & coefficients) const
  {
    const int terms = path_detection_.polynomial_degree + 1;
    if (static_cast<int>(points.size()) < terms || image_height < 2) {
      return false;
    }

    cv::Mat design(static_cast<int>(points.size()), terms, CV_64F);
    cv::Mat observations(static_cast<int>(points.size()), 1, CV_64F);
    const double y_scale = 1.0 / static_cast<double>(image_height - 1);
    for (std::size_t row = 0; row < points.size(); ++row) {
      const double normalized_y = static_cast<double>(points[row].y) * y_scale;
      double power = 1.0;
      for (int column = 0; column < terms; ++column) {
        design.at<double>(static_cast<int>(row), column) = power;
        power *= normalized_y;
      }
      observations.at<double>(static_cast<int>(row), 0) = points[row].x;
    }

    cv::Mat solution;
    if (!cv::solve(design, observations, solution, cv::DECOMP_SVD)) {
      return false;
    }
    coefficients.resize(static_cast<std::size_t>(terms));
    for (int index = 0; index < terms; ++index) {
      const double value = solution.at<double>(index, 0);
      if (!std::isfinite(value)) {
        return false;
      }
      coefficients[static_cast<std::size_t>(index)] = value;
    }
    return true;
  }

  static double evaluate_polynomial(
    const std::vector<double> & coefficients, double normalized_y)
  {
    double value = 0.0;
    for (auto iterator = coefficients.rbegin(); iterator != coefficients.rend(); ++iterator) {
      value = value * normalized_y + *iterator;
    }
    return value;
  }

  double effective_lookahead_distance_m() const
  {
    if (!control_.adaptive_speed_enabled) {
      return path_detection_.lookahead_distance_m;
    }
    const double reference_speed = control_.enabled ?
      std::max(last_speed_command_mps_, control_.minimum_speed_mps) :
      control_.test_speed_mps;
    return std::clamp(
      control_.minimum_lookahead_m + control_.lookahead_speed_gain * reference_speed,
      control_.minimum_lookahead_m, control_.maximum_lookahead_m);
  }

  bool build_component_spline(LaneFitResult & result, const cv::Size & image_size) const
  {
    if (result.component_centers.size() < 3U || image_size.width < 2 || image_size.height < 2) {
      return false;
    }

    const double lane_width_ratio = 0.5 * (
      perspective_.dst_top_right_x_ratio - perspective_.dst_top_left_x_ratio +
      perspective_.dst_bottom_right_x_ratio - perspective_.dst_bottom_left_x_ratio);
    const double lane_width_px = lane_width_ratio * static_cast<double>(image_size.width - 1);
    if (lane_width_px <= 1.0) {
      return false;
    }
    const double meters_per_pixel_x = path_detection_.lane_width_m / lane_width_px;
    const double meters_per_pixel_y = path_detection_.bev_forward_range_m /
      static_cast<double>(image_size.height - 1);
    const cv::Point2f vehicle_pixel(
      static_cast<float>(image_size.width - 1) * 0.5F,
      static_cast<float>(image_size.height - 1));
    const auto to_metric = [&](const cv::Point2f & point) {
        return cv::Point2d(
          (point.x - vehicle_pixel.x) * meters_per_pixel_x,
          (vehicle_pixel.y - point.y) * meters_per_pixel_y);
      };
    const auto metric_distance = [&](const cv::Point2f & first, const cv::Point2f & second) {
        const cv::Point2d delta = to_metric(second) - to_metric(first);
        return std::hypot(delta.x, delta.y);
      };

    std::vector<bool> used(result.component_centers.size(), false);
    std::vector<cv::Point2f> ordered;
    ordered.reserve(result.component_centers.size());
    std::size_t start_index = 0U;
    double start_distance = std::numeric_limits<double>::max();
    for (std::size_t index = 0; index < result.component_centers.size(); ++index) {
      const cv::Point2d metric = to_metric(result.component_centers[index]);
      const double distance = std::hypot(metric.x, metric.y);
      if (distance < start_distance) {
        start_distance = distance;
        start_index = index;
      }
    }
    if (start_distance > path_detection_.component_chain_max_gap_m) {
      return false;
    }

    used[start_index] = true;
    ordered.push_back(result.component_centers[start_index]);
    cv::Point2d heading(0.0, 1.0);
    double maximum_gap = 0.0;
    while (ordered.size() < result.component_centers.size()) {
      const cv::Point2d current_metric = to_metric(ordered.back());
      std::size_t best_index = result.component_centers.size();
      double best_score = std::numeric_limits<double>::max();
      double best_distance = 0.0;
      cv::Point2d best_direction;
      for (std::size_t index = 0; index < result.component_centers.size(); ++index) {
        if (used[index]) {
          continue;
        }
        const cv::Point2d delta = to_metric(result.component_centers[index]) - current_metric;
        const double distance = std::hypot(delta.x, delta.y);
        if (distance < 1.0e-6 || distance > path_detection_.component_chain_max_gap_m) {
          continue;
        }
        const cv::Point2d direction(delta.x / distance, delta.y / distance);
        const double alignment = heading.dot(direction);
        if (alignment < -0.20) {
          continue;
        }
        const double score = distance * (1.0 + 0.75 * (1.0 - alignment));
        if (score < best_score) {
          best_score = score;
          best_index = index;
          best_distance = distance;
          best_direction = direction;
        }
      }
      if (best_index == result.component_centers.size()) {
        break;
      }
      used[best_index] = true;
      ordered.push_back(result.component_centers[best_index]);
      maximum_gap = std::max(maximum_gap, best_distance);
      heading = cv::Point2d(
        0.35 * heading.x + 0.65 * best_direction.x,
        0.35 * heading.y + 0.65 * best_direction.y);
      const double heading_length = std::hypot(heading.x, heading.y);
      if (heading_length > 1.0e-6) {
        heading.x /= heading_length;
        heading.y /= heading_length;
      }
    }

    if (static_cast<int>(ordered.size()) < path_detection_.min_detection_points) {
      return false;
    }

    std::vector<cv::Point2f> spline_pixels;
    for (std::size_t segment = 0; segment + 1U < ordered.size(); ++segment) {
      const cv::Point2f & p0 = ordered[segment == 0U ? 0U : segment - 1U];
      const cv::Point2f & p1 = ordered[segment];
      const cv::Point2f & p2 = ordered[segment + 1U];
      const cv::Point2f & p3 = ordered[std::min(segment + 2U, ordered.size() - 1U)];
      const int samples = std::max(
        2, static_cast<int>(std::ceil(
          metric_distance(p1, p2) / path_detection_.spline_sample_spacing_m)));
      for (int sample = 0; sample < samples; ++sample) {
        const float t = static_cast<float>(sample) / static_cast<float>(samples);
        const float t2 = t * t;
        const float t3 = t2 * t;
        const cv::Point2f point = 0.5F * (
          (2.0F * p1) + (-p0 + p2) * t +
          (2.0F * p0 - 5.0F * p1 + 4.0F * p2 - p3) * t2 +
          (-p0 + 3.0F * p1 - 3.0F * p2 + p3) * t3);
        if (point.x >= 0.0F && point.x < image_size.width &&
          point.y >= 0.0F && point.y < image_size.height)
        {
          spline_pixels.push_back(point);
        }
      }
    }
    spline_pixels.push_back(ordered.back());
    if (spline_pixels.size() < 2U) {
      return false;
    }

    const double lookahead_distance_m = effective_lookahead_distance_m();
    double accumulated_distance = metric_distance(vehicle_pixel, spline_pixels.front());
    double total_distance = accumulated_distance;
    cv::Point2f lookahead = spline_pixels.back();
    bool selected_lookahead = accumulated_distance >= lookahead_distance_m;
    if (selected_lookahead) {
      lookahead = spline_pixels.front();
    }
    for (std::size_t index = 0; index < spline_pixels.size(); ++index) {
      const cv::Point rounded(
        static_cast<int>(std::lround(spline_pixels[index].x)),
        static_cast<int>(std::lround(spline_pixels[index].y)));
      if (result.sampled_path.empty() || result.sampled_path.back() != rounded) {
        result.sampled_path.push_back(rounded);
      }
      if (index == 0U) {
        continue;
      }
      const double segment_distance = metric_distance(
        spline_pixels[index - 1U], spline_pixels[index]);
      total_distance += segment_distance;
      if (!selected_lookahead && total_distance >= lookahead_distance_m) {
        lookahead = spline_pixels[index];
        selected_lookahead = true;
      }
    }
    if (total_distance < std::max(0.15, 0.35 * lookahead_distance_m) ||
      result.sampled_path.size() < 2U)
    {
      result.sampled_path.clear();
      return false;
    }

    float minimum_y = static_cast<float>(image_size.height - 1);
    float maximum_y = 0.0F;
    for (const auto & point : ordered) {
      minimum_y = std::min(minimum_y, point.y);
      maximum_y = std::max(maximum_y, point.y);
    }
    result.vertical_coverage = std::clamp(
      static_cast<double>(maximum_y - minimum_y) /
      static_cast<double>(image_size.height - 1), 0.0, 1.0);
    const double length_score = std::min(
      1.0, total_distance / std::max(0.1, 1.25 * lookahead_distance_m));
    const double ordered_score = std::min(1.0, static_cast<double>(ordered.size()) / 8.0);
    const double start_score = std::clamp(
      1.0 - start_distance / path_detection_.component_chain_max_gap_m, 0.0, 1.0);
    const double continuity_score = std::clamp(
      1.0 - maximum_gap / path_detection_.component_chain_max_gap_m, 0.0, 1.0);
    result.confidence = std::clamp(
      0.40 * length_score + 0.25 * ordered_score +
      0.20 * start_score + 0.15 * continuity_score,
      0.0, 1.0);
    result.detection_points = ordered;
    result.lookahead_point = cv::Point(
      static_cast<int>(std::lround(lookahead.x)),
      static_cast<int>(std::lround(lookahead.y)));
    result.model = "SPLINE";
    result.valid = true;
    return true;
  }

  LaneFitResult detect_orange_path(const cv::Mat & binary_mask)
  {
    LaneFitResult result;
    if (binary_mask.empty() || binary_mask.type() != CV_8UC1) {
      return result;
    }

    cv::Mat statistics;
    cv::Mat centroids;
    const int label_count = cv::connectedComponentsWithStats(
      binary_mask, component_labels_, statistics, centroids, 8, CV_32S);
    std::vector<uint8_t> valid_label(static_cast<std::size_t>(label_count), 0U);
    for (int label = 1; label < label_count; ++label) {
      const int area = statistics.at<int>(label, cv::CC_STAT_AREA);
      if (area < path_detection_.min_component_area_px ||
        area > path_detection_.max_component_area_px)
      {
        continue;
      }
      valid_label[static_cast<std::size_t>(label)] = 1U;
      result.component_centers.emplace_back(
        static_cast<float>(centroids.at<double>(label, 0)),
        static_cast<float>(centroids.at<double>(label, 1)));
      ++result.component_count;
    }

    filtered_component_mask_.create(binary_mask.size(), CV_8UC1);
    filtered_component_mask_.setTo(cv::Scalar(0));
    for (int y = 0; y < binary_mask.rows; ++y) {
      const int * label_row = component_labels_.ptr<int>(y);
      uint8_t * filtered_row = filtered_component_mask_.ptr<uint8_t>(y);
      for (int x = 0; x < binary_mask.cols; ++x) {
        const int label = label_row[x];
        if (label > 0 && valid_label[static_cast<std::size_t>(label)] != 0U) {
          filtered_row[x] = 255U;
        }
      }
    }

    // A 2D parametric spline remains valid when a sharp corner becomes nearly
    // horizontal in BEV. The x(y) polynomial below is retained as a fallback
    // for frames that contain too few separately connected dash components.
    if (build_component_spline(result, binary_mask.size())) {
      return result;
    }

    // Each horizontal bin contributes one center. This preserves the shape of
    // long dash segments while naturally bridging the empty spaces between them.
    for (int y_begin = 0; y_begin < binary_mask.rows;
      y_begin += path_detection_.row_bin_height_px)
    {
      const int y_end = std::min(
        binary_mask.rows, y_begin + path_detection_.row_bin_height_px);
      int64_t sum_x = 0;
      int64_t sum_y = 0;
      int count = 0;
      for (int y = y_begin; y < y_end; ++y) {
        const uint8_t * row = filtered_component_mask_.ptr<uint8_t>(y);
        for (int x = 0; x < binary_mask.cols; ++x) {
          if (row[x] == 0U) {
            continue;
          }
          sum_x += x;
          sum_y += y;
          ++count;
        }
      }
      if (count > 0) {
        result.detection_points.emplace_back(
          static_cast<float>(static_cast<double>(sum_x) / count),
          static_cast<float>(static_cast<double>(sum_y) / count));
      }
    }

    if (static_cast<int>(result.detection_points.size()) <
      path_detection_.min_detection_points)
    {
      return result;
    }

    std::vector<double> coefficients;
    if (!fit_polynomial(result.detection_points, binary_mask.rows, coefficients)) {
      return result;
    }

    // One robust rejection pass prevents a small false orange blob from
    // pulling the path abruptly toward the edge of the image.
    std::vector<cv::Point2f> inliers;
    inliers.reserve(result.detection_points.size());
    const double y_scale = 1.0 / static_cast<double>(binary_mask.rows - 1);
    for (const auto & point : result.detection_points) {
      const double predicted_x = evaluate_polynomial(coefficients, point.y * y_scale);
      if (std::abs(predicted_x - point.x) <= path_detection_.fit_max_residual_px) {
        inliers.push_back(point);
      }
    }
    if (static_cast<int>(inliers.size()) < path_detection_.min_detection_points) {
      return result;
    }
    if (inliers.size() != result.detection_points.size()) {
      if (!fit_polynomial(inliers, binary_mask.rows, coefficients)) {
        return result;
      }
      result.detection_points = inliers;
    }

    double squared_error_sum = 0.0;
    float minimum_y = static_cast<float>(binary_mask.rows - 1);
    float maximum_y = 0.0F;
    for (const auto & point : result.detection_points) {
      const double predicted_x = evaluate_polynomial(coefficients, point.y * y_scale);
      const double residual = predicted_x - point.x;
      squared_error_sum += residual * residual;
      minimum_y = std::min(minimum_y, point.y);
      maximum_y = std::max(maximum_y, point.y);
    }
    result.residual_rms_px = std::sqrt(
      squared_error_sum / static_cast<double>(result.detection_points.size()));
    result.vertical_coverage = std::clamp(
      static_cast<double>(maximum_y - minimum_y) /
      static_cast<double>(binary_mask.rows - 1), 0.0, 1.0);

    const double point_score = std::min(
      1.0, static_cast<double>(result.detection_points.size()) / 18.0);
    const double residual_score = std::clamp(
      1.0 - result.residual_rms_px / path_detection_.fit_max_residual_px,
      0.0, 1.0);
    const double component_score = std::min(
      1.0, static_cast<double>(result.component_count) / 3.0);
    result.confidence = std::clamp(
      0.40 * result.vertical_coverage + 0.25 * point_score +
      0.20 * residual_score + 0.15 * component_score,
      0.0, 1.0);
    result.coefficients = coefficients;
    result.model = "POLY";

    for (int y = binary_mask.rows - 1; y >= 0;
      y -= path_detection_.path_sample_step_px)
    {
      const double x = evaluate_polynomial(coefficients, y * y_scale);
      if (!std::isfinite(x) || x < 0.0 || x >= binary_mask.cols) {
        continue;
      }
      result.sampled_path.emplace_back(static_cast<int>(std::lround(x)), y);
    }

    const double lookahead_ratio = std::clamp(
      effective_lookahead_distance_m() /
      path_detection_.bev_forward_range_m, 0.0, 1.0);
    const int lookahead_y = static_cast<int>(std::lround(
      static_cast<double>(binary_mask.rows - 1) * (1.0 - lookahead_ratio)));
    const double lookahead_x = evaluate_polynomial(coefficients, lookahead_y * y_scale);
    if (!std::isfinite(lookahead_x) || lookahead_x < 0.0 ||
      lookahead_x >= binary_mask.cols || result.sampled_path.empty())
    {
      return result;
    }
    result.lookahead_point = cv::Point(
      static_cast<int>(std::lround(lookahead_x)), lookahead_y);
    result.valid = true;
    return result;
  }

  LaneFitResult detect_white_path(const cv::Mat & binary_mask)
  {
    LaneFitResult result;
    if (binary_mask.empty() || binary_mask.type() != CV_8UC1 || binary_mask.rows < 2) {
      return result;
    }
    const double lane_width_ratio = 0.5 * (
      perspective_.dst_top_right_x_ratio - perspective_.dst_top_left_x_ratio +
      perspective_.dst_bottom_right_x_ratio - perspective_.dst_bottom_left_x_ratio);
    const double expected_lane_width_px = lane_width_ratio *
      static_cast<double>(binary_mask.cols - 1);
    const double minimum_lane_width_px =
      expected_lane_width_px * white_mask_.minimum_lane_width_ratio;
    const double maximum_lane_width_px =
      expected_lane_width_px * white_mask_.maximum_lane_width_ratio;
    const int image_center_x = binary_mask.cols / 2;
    std::vector<double> measured_widths;
    int both_boundary_bins = 0;

    for (int y_begin = 0; y_begin < binary_mask.rows;
      y_begin += path_detection_.row_bin_height_px)
    {
      const int y_end = std::min(
        binary_mask.rows, y_begin + path_detection_.row_bin_height_px);
      std::vector<int> left_pixels;
      std::vector<int> right_pixels;
      for (int y = y_begin; y < y_end; ++y) {
        const uint8_t * row = binary_mask.ptr<uint8_t>(y);
        for (int x = 0; x < binary_mask.cols; ++x) {
          if (row[x] == 0U) {
            continue;
          }
          if (x < image_center_x) {
            left_pixels.push_back(x);
          } else {
            right_pixels.push_back(x);
          }
        }
      }
      const bool has_left = static_cast<int>(left_pixels.size()) >=
        white_mask_.minimum_pixels_per_bin;
      const bool has_right = static_cast<int>(right_pixels.size()) >=
        white_mask_.minimum_pixels_per_bin;
      if (!has_left && !has_right) {
        continue;
      }

      double left_x = 0.0;
      double right_x = 0.0;
      if (has_left) {
        const std::size_t index = static_cast<std::size_t>(
          0.80 * static_cast<double>(left_pixels.size() - 1U));
        std::nth_element(
          left_pixels.begin(), left_pixels.begin() + static_cast<std::ptrdiff_t>(index),
          left_pixels.end());
        left_x = left_pixels[index];
      }
      if (has_right) {
        const std::size_t index = static_cast<std::size_t>(
          0.20 * static_cast<double>(right_pixels.size() - 1U));
        std::nth_element(
          right_pixels.begin(), right_pixels.begin() + static_cast<std::ptrdiff_t>(index),
          right_pixels.end());
        right_x = right_pixels[index];
      }
      const float point_y = static_cast<float>(0.5 * (y_begin + y_end - 1));
      if (has_left) {
        result.left_boundary_points.emplace_back(static_cast<float>(left_x), point_y);
      }
      if (has_right) {
        result.right_boundary_points.emplace_back(static_cast<float>(right_x), point_y);
      }

      double center_x = 0.0;
      if (has_left && has_right) {
        const double width = right_x - left_x;
        if (width < minimum_lane_width_px || width > maximum_lane_width_px) {
          continue;
        }
        center_x = 0.5 * (left_x + right_x);
        measured_widths.push_back(width);
        ++both_boundary_bins;
      } else if (has_left) {
        center_x = left_x + 0.5 * expected_lane_width_px;
      } else {
        center_x = right_x - 0.5 * expected_lane_width_px;
      }
      if (center_x >= 0.0 && center_x < binary_mask.cols) {
        result.detection_points.emplace_back(static_cast<float>(center_x), point_y);
      }
    }

    if (static_cast<int>(result.detection_points.size()) <
      path_detection_.min_detection_points)
    {
      return result;
    }
    std::vector<double> coefficients;
    if (!fit_polynomial(result.detection_points, binary_mask.rows, coefficients)) {
      return result;
    }
    const double y_scale = 1.0 / static_cast<double>(binary_mask.rows - 1);
    std::vector<cv::Point2f> inliers;
    for (const auto & point : result.detection_points) {
      const double predicted_x = evaluate_polynomial(coefficients, point.y * y_scale);
      if (std::abs(predicted_x - point.x) <= path_detection_.fit_max_residual_px) {
        inliers.push_back(point);
      }
    }
    if (static_cast<int>(inliers.size()) < path_detection_.min_detection_points) {
      return result;
    }
    if (inliers.size() != result.detection_points.size()) {
      if (!fit_polynomial(inliers, binary_mask.rows, coefficients)) {
        return result;
      }
      result.detection_points = inliers;
    }

    double squared_error_sum = 0.0;
    float minimum_y = static_cast<float>(binary_mask.rows - 1);
    float maximum_y = 0.0F;
    for (const auto & point : result.detection_points) {
      const double predicted_x = evaluate_polynomial(coefficients, point.y * y_scale);
      const double residual = predicted_x - point.x;
      squared_error_sum += residual * residual;
      minimum_y = std::min(minimum_y, point.y);
      maximum_y = std::max(maximum_y, point.y);
    }
    result.residual_rms_px = std::sqrt(
      squared_error_sum / static_cast<double>(result.detection_points.size()));
    result.vertical_coverage = std::clamp(
      static_cast<double>(maximum_y - minimum_y) /
      static_cast<double>(binary_mask.rows - 1), 0.0, 1.0);
    const double point_score = std::min(
      1.0, static_cast<double>(result.detection_points.size()) / 18.0);
    const double residual_score = std::clamp(
      1.0 - result.residual_rms_px / path_detection_.fit_max_residual_px, 0.0, 1.0);
    const double both_score = std::clamp(
      static_cast<double>(both_boundary_bins) /
      static_cast<double>(result.detection_points.size()), 0.0, 1.0);
    double width_score = both_score > 0.0 ? 1.0 : 0.4;
    if (measured_widths.size() >= 2U) {
      const double mean_width = std::accumulate(
        measured_widths.begin(), measured_widths.end(), 0.0) /
        static_cast<double>(measured_widths.size());
      double variance = 0.0;
      for (const double width : measured_widths) {
        const double difference = width - mean_width;
        variance += difference * difference;
      }
      const double normalized_stddev = std::sqrt(
        variance / static_cast<double>(measured_widths.size())) /
        std::max(1.0, expected_lane_width_px);
      width_score = std::clamp(1.0 - 4.0 * normalized_stddev, 0.0, 1.0);
    }
    result.confidence = std::clamp(
      0.35 * result.vertical_coverage + 0.20 * point_score +
      0.15 * residual_score + 0.20 * both_score + 0.10 * width_score,
      0.0, 1.0);

    for (int y = binary_mask.rows - 1; y >= 0;
      y -= path_detection_.path_sample_step_px)
    {
      const double x = evaluate_polynomial(coefficients, y * y_scale);
      if (std::isfinite(x) && x >= 0.0 && x < binary_mask.cols) {
        result.sampled_path.emplace_back(static_cast<int>(std::lround(x)), y);
      }
    }
    const double lookahead_ratio = std::clamp(
      effective_lookahead_distance_m() / path_detection_.bev_forward_range_m,
      0.0, 1.0);
    const int lookahead_y = static_cast<int>(std::lround(
      static_cast<double>(binary_mask.rows - 1) * (1.0 - lookahead_ratio)));
    const double lookahead_x = evaluate_polynomial(coefficients, lookahead_y * y_scale);
    if (!std::isfinite(lookahead_x) || lookahead_x < 0.0 ||
      lookahead_x >= binary_mask.cols || result.sampled_path.empty())
    {
      return result;
    }
    result.coefficients = coefficients;
    result.lookahead_point = cv::Point(
      static_cast<int>(std::lround(lookahead_x)), lookahead_y);
    result.model = both_score >= 0.60 ? "WHITE_BOTH" : "WHITE_SINGLE";
    result.valid = true;
    return result;
  }

  void on_obstacles(
    const physicar_interfaces::msg::ObstacleArray::ConstSharedPtr message)
  {
    latest_obstacles_ = message->obstacles;
    last_obstacle_message_at_ = std::chrono::steady_clock::now();
    received_obstacles_ = true;
  }

  double path_clearance_m(const LaneFitResult & lane_fit, double lateral_offset_m) const
  {
    if (lane_fit.sampled_path.empty()) {
      return std::numeric_limits<double>::infinity();
    }
    const double lane_width_ratio = 0.5 * (
      perspective_.dst_top_right_x_ratio - perspective_.dst_top_left_x_ratio +
      perspective_.dst_bottom_right_x_ratio - perspective_.dst_bottom_left_x_ratio);
    const double meters_per_pixel_x = path_detection_.lane_width_m /
      std::max(1.0, lane_width_ratio * (perspective_.output_width - 1));
    const double meters_per_pixel_y = path_detection_.bev_forward_range_m /
      std::max(1, perspective_.output_height - 1);
    const double vehicle_x_px = 0.5 * static_cast<double>(perspective_.output_width - 1);
    double minimum_clearance = std::numeric_limits<double>::infinity();
    for (const auto & obstacle : latest_obstacles_) {
      if (obstacle.centroid.x < avoidance_.minimum_forward_m ||
        obstacle.centroid.x > avoidance_.detection_distance_m)
      {
        continue;
      }
      for (const auto & point : lane_fit.sampled_path) {
        const double path_forward =
          (perspective_.output_height - 1 - point.y) * meters_per_pixel_y;
        const double path_lateral_left =
          (vehicle_x_px - point.x) * meters_per_pixel_x + lateral_offset_m;
        const double clearance = std::hypot(
          obstacle.centroid.x - path_forward,
          obstacle.centroid.y - path_lateral_left) - 0.5 * obstacle.width;
        minimum_clearance = std::min(minimum_clearance, clearance);
      }
    }
    return minimum_clearance;
  }

  double obstacle_path_clearance_m(
    const LaneFitResult & lane_fit,
    const physicar_interfaces::msg::Obstacle & obstacle,
    double lateral_offset_m) const
  {
    if (lane_fit.sampled_path.empty()) {
      return std::numeric_limits<double>::infinity();
    }
    const double lane_width_ratio = 0.5 * (
      perspective_.dst_top_right_x_ratio - perspective_.dst_top_left_x_ratio +
      perspective_.dst_bottom_right_x_ratio - perspective_.dst_bottom_left_x_ratio);
    const double meters_per_pixel_x = path_detection_.lane_width_m /
      std::max(1.0, lane_width_ratio * (perspective_.output_width - 1));
    const double meters_per_pixel_y = path_detection_.bev_forward_range_m /
      std::max(1, perspective_.output_height - 1);
    const double vehicle_x_px = 0.5 * static_cast<double>(perspective_.output_width - 1);
    double minimum_clearance = std::numeric_limits<double>::infinity();
    for (const auto & point : lane_fit.sampled_path) {
      const double path_forward =
        (perspective_.output_height - 1 - point.y) * meters_per_pixel_y;
      const double path_lateral_left =
        (vehicle_x_px - point.x) * meters_per_pixel_x + lateral_offset_m;
      const double clearance = std::hypot(
        obstacle.centroid.x - path_forward,
        obstacle.centroid.y - path_lateral_left) - 0.5 * obstacle.width;
      minimum_clearance = std::min(minimum_clearance, clearance);
    }
    return minimum_clearance;
  }

  const physicar_interfaces::msg::Obstacle * nearest_base_path_collision(
    const LaneFitResult & lane_fit) const
  {
    const physicar_interfaces::msg::Obstacle * nearest = nullptr;
    double nearest_forward = std::numeric_limits<double>::infinity();
    for (const auto & obstacle : latest_obstacles_) {
      if (obstacle.centroid.x < avoidance_.minimum_forward_m ||
        obstacle.centroid.x > avoidance_.detection_distance_m)
      {
        continue;
      }
      const double clearance = obstacle_path_clearance_m(lane_fit, obstacle, 0.0);
      if (clearance < avoidance_.safety_distance_m &&
        obstacle.centroid.x < nearest_forward)
      {
        nearest = &obstacle;
        nearest_forward = obstacle.centroid.x;
      }
    }
    return nearest;
  }

  const physicar_interfaces::msg::Obstacle * associated_active_obstacle() const
  {
    const physicar_interfaces::msg::Obstacle * associated = nullptr;
    double best_distance = avoidance_.association_distance_m;
    for (const auto & obstacle : latest_obstacles_) {
      if (obstacle.centroid.x < avoidance_.passed_forward_m ||
        obstacle.centroid.x > avoidance_.detection_distance_m)
      {
        continue;
      }
      const double distance = std::hypot(
        obstacle.centroid.x - active_obstacle_forward_m_,
        obstacle.centroid.y - active_obstacle_lateral_m_);
      if (distance <= best_distance) {
        associated = &obstacle;
        best_distance = distance;
      }
    }
    return associated;
  }

  void apply_obstacle_avoidance(
    LaneFitResult & lane_fit,
    const std::chrono::steady_clock::time_point & now)
  {
    lane_fit.base_sampled_path = lane_fit.sampled_path;
    const double obstacle_age_sec = received_obstacles_ ?
      std::chrono::duration<double>(now - last_obstacle_message_at_).count() :
      std::numeric_limits<double>::infinity();
    obstacle_data_stale_ = avoidance_.enabled &&
      obstacle_age_sec > avoidance_.obstacle_timeout_sec;
    if (!avoidance_.enabled) {
      current_path_offset_m_ = 0.0;
      active_obstacle_valid_ = false;
      planner_state_ = "LANE_FOLLOW";
      last_planner_at_ = now;
      return;
    }
    if (obstacle_data_stale_) {
      active_obstacle_valid_ = false;
      planner_state_ = "LIDAR_STALE";
      last_planner_at_ = now;
      return;
    }

    const double base_clearance = path_clearance_m(lane_fit, 0.0);
    const auto * collision_obstacle = nearest_base_path_collision(lane_fit);
    const bool collision = collision_obstacle != nullptr;

    if (active_obstacle_valid_) {
      const auto * associated = associated_active_obstacle();
      if (associated != nullptr) {
        active_obstacle_forward_m_ = associated->centroid.x;
        active_obstacle_lateral_m_ = associated->centroid.y;
        active_obstacle_last_seen_at_ = now;
      } else {
        const double unseen_age_sec = std::chrono::duration<double>(
          now - active_obstacle_last_seen_at_).count();
        if (unseen_age_sec > avoidance_.active_lost_timeout_sec) {
          active_obstacle_valid_ = false;
        }
      }
    }

    double target_offset_m = current_path_offset_m_;
    if (active_obstacle_valid_) {
      obstacle_path_blocked_ = false;
      target_offset_m = avoidance_direction_sign_ * avoidance_.lateral_offset_m;
      planner_state_ = avoidance_direction_sign_ > 0.0 ? "AVOID_LEFT" : "AVOID_RIGHT";
      last_collision_at_ = now;
    } else if (collision) {
      last_collision_at_ = now;
      const double left_clearance = path_clearance_m(
        lane_fit, avoidance_.lateral_offset_m);
      const double right_clearance = path_clearance_m(
        lane_fit, -avoidance_.lateral_offset_m);
      if (std::max(left_clearance, right_clearance) < avoidance_.safety_distance_m) {
        obstacle_path_blocked_ = true;
        target_offset_m = current_path_offset_m_;
        planner_state_ = "OBSTACLE_BLOCKED";
      } else {
        obstacle_path_blocked_ = false;
        if (std::abs(left_clearance - right_clearance) < 0.02) {
          avoidance_direction_sign_ = avoidance_.preferred_side == "left" ? 1.0 : -1.0;
        } else {
          avoidance_direction_sign_ = left_clearance > right_clearance ? 1.0 : -1.0;
        }
        active_obstacle_valid_ = true;
        active_obstacle_forward_m_ = collision_obstacle->centroid.x;
        active_obstacle_lateral_m_ = collision_obstacle->centroid.y;
        active_obstacle_last_seen_at_ = now;
        target_offset_m = avoidance_direction_sign_ * avoidance_.lateral_offset_m;
        planner_state_ = avoidance_direction_sign_ > 0.0 ? "AVOID_LEFT" : "AVOID_RIGHT";
      }
    } else {
      obstacle_path_blocked_ = false;
      const double clear_age_sec = std::chrono::duration<double>(
        now - last_collision_at_).count();
      if ((planner_state_ == "AVOID_LEFT" || planner_state_ == "AVOID_RIGHT") &&
        clear_age_sec <= avoidance_.clear_time_sec)
      {
        target_offset_m = avoidance_direction_sign_ * avoidance_.lateral_offset_m;
      } else if (std::abs(current_path_offset_m_) > 0.005) {
        target_offset_m = 0.0;
        planner_state_ = "RETURN_TO_LANE";
      } else {
        target_offset_m = 0.0;
        current_path_offset_m_ = 0.0;
        planner_state_ = "LANE_FOLLOW";
      }
    }

    double elapsed_sec = std::chrono::duration<double>(now - last_planner_at_).count();
    elapsed_sec = std::clamp(elapsed_sec, 0.0, 0.25);
    double offset_rate = avoidance_.offset_rate_mps;
    if (planner_state_ == "RETURN_TO_LANE") {
      const double reference_speed = std::max(0.20, last_speed_command_mps_);
      offset_rate = std::max(
        0.05, avoidance_.lateral_offset_m * reference_speed /
        avoidance_.return_distance_m);
    }
    const double maximum_step = offset_rate * elapsed_sec;
    if (current_path_offset_m_ < target_offset_m) {
      current_path_offset_m_ = std::min(
        current_path_offset_m_ + maximum_step, target_offset_m);
    } else {
      current_path_offset_m_ = std::max(
        current_path_offset_m_ - maximum_step, target_offset_m);
    }
    last_planner_at_ = now;

    const double lane_width_ratio = 0.5 * (
      perspective_.dst_top_right_x_ratio - perspective_.dst_top_left_x_ratio +
      perspective_.dst_bottom_right_x_ratio - perspective_.dst_bottom_left_x_ratio);
    const double meters_per_pixel_x = path_detection_.lane_width_m /
      std::max(1.0, lane_width_ratio * (perspective_.output_width - 1));
    const int offset_pixels = static_cast<int>(std::lround(
      current_path_offset_m_ / meters_per_pixel_x));
    for (auto & point : lane_fit.sampled_path) {
      point.x = std::clamp(point.x - offset_pixels, 0, perspective_.output_width - 1);
    }
    lane_fit.lookahead_point.x = std::clamp(
      lane_fit.lookahead_point.x - offset_pixels, 0, perspective_.output_width - 1);

    if (planner_state_ != last_logged_planner_state_) {
      RCLCPP_INFO(
        get_logger(), "Local planner state: %s (base clearance %.2f m, offset %.2f m)",
        planner_state_.c_str(), base_clearance, current_path_offset_m_);
      last_logged_planner_state_ = planner_state_;
    }
  }

  void publish_control(double speed_mps, double steering_rad)
  {
    if (!std::isfinite(speed_mps) || !std::isfinite(steering_rad)) {
      speed_mps = 0.0;
      steering_rad = 0.0;
    }
    speed_mps = std::clamp(speed_mps, 0.0, 3.0);
    steering_rad = std::clamp(
      steering_rad, -control_.max_steering_rad, control_.max_steering_rad);
    std_msgs::msg::Float64 speed_message;
    std_msgs::msg::Float64 steering_message;
    speed_message.data = speed_mps;
    steering_message.data = steering_rad;

    // While moving, settle steering first. During a stop, remove propulsion
    // first and then center the steering command.
    if (speed_mps > 1.0e-3) {
      steering_pub_->publish(steering_message);
      speed_pub_->publish(speed_message);
    } else {
      speed_pub_->publish(speed_message);
      steering_pub_->publish(steering_message);
    }
    last_speed_command_mps_ = speed_mps;
    last_steering_command_rad_ = steering_rad;
  }

  void update_controller(
    const LaneFitResult & lane_fit,
    const std::chrono::steady_clock::time_point & now)
  {
    if (!control_.enabled) {
      controller_state_ = "DISABLED";
      last_speed_command_mps_ = 0.0;
      last_steering_command_rad_ = 0.0;
      last_control_at_ = now;
      has_reliable_control_ = false;
      lost_frame_count_ = 0;
      return;
    }
    if (avoidance_.enabled && obstacle_data_stale_) {
      publish_control(0.0, 0.0);
      controller_state_ = "LOST_LIDAR_STALE";
      last_control_at_ = now;
      return;
    }
    if (avoidance_.enabled && obstacle_path_blocked_) {
      publish_control(0.0, 0.0);
      controller_state_ = "OBSTACLE_BLOCKED";
      last_control_at_ = now;
      return;
    }
    const bool lane_usable = lane_fit.valid &&
      lane_fit.confidence >= path_detection_.minimum_lane_confidence;
    const bool lane_reliable = lane_usable &&
      lane_fit.confidence >= control_.tracking_lane_confidence;
    if (!lane_usable) {
      ++lost_frame_count_;
      const double reliable_age_sec = has_reliable_control_ ?
        std::chrono::duration<double>(now - last_reliable_control_at_).count() :
        std::numeric_limits<double>::infinity();
      if (has_reliable_control_ &&
        lost_frame_count_ <= control_.lost_frame_threshold &&
        reliable_age_sec <= control_.previous_path_hold_sec)
      {
        publish_control(
          control_.low_confidence_speed_mps, last_reliable_steering_rad_);
        controller_state_ = "LOW_CONFIDENCE_HOLD";
      } else {
        publish_control(0.0, 0.0);
        controller_state_ = "LOST";
      }
      last_control_at_ = now;
      return;
    }

    const double lane_width_ratio = 0.5 * (
      perspective_.dst_top_right_x_ratio - perspective_.dst_top_left_x_ratio +
      perspective_.dst_bottom_right_x_ratio - perspective_.dst_bottom_left_x_ratio);
    const double lane_width_px = lane_width_ratio *
      static_cast<double>(perspective_.output_width - 1);
    if (lane_width_px <= 1.0 || perspective_.output_height < 2) {
      publish_control(0.0, 0.0);
      controller_state_ = "LOST_INVALID_GEOMETRY";
      last_control_at_ = now;
      return;
    }
    const double meters_per_pixel_x = path_detection_.lane_width_m / lane_width_px;
    const double meters_per_pixel_y = path_detection_.bev_forward_range_m /
      static_cast<double>(perspective_.output_height - 1);
    const double vehicle_x_px = static_cast<double>(perspective_.output_width - 1) * 0.5;
    const double lateral_left_m =
      (vehicle_x_px - lane_fit.lookahead_point.x) * meters_per_pixel_x;
    const double forward_m =
      (static_cast<double>(perspective_.output_height - 1) - lane_fit.lookahead_point.y) *
      meters_per_pixel_y;
    const double target_distance_squared =
      forward_m * forward_m + lateral_left_m * lateral_left_m;
    if (!std::isfinite(target_distance_squared) ||
      target_distance_squared <
      control_.minimum_target_distance_m * control_.minimum_target_distance_m)
    {
      publish_control(0.0, 0.0);
      controller_state_ = "LOST_INVALID_TARGET";
      last_control_at_ = now;
      return;
    }

    double steering_rad = std::atan2(
      2.0 * control_.wheelbase_m * lateral_left_m,
      target_distance_squared);
    steering_rad = std::clamp(
      steering_rad, -control_.max_steering_rad, control_.max_steering_rad);
    double elapsed_sec = std::chrono::duration<double>(now - last_control_at_).count();
    elapsed_sec = std::clamp(elapsed_sec, 0.0, 0.25);
    const double maximum_change = control_.max_steering_rate_rad_s * elapsed_sec;
    steering_rad = std::clamp(
      steering_rad,
      last_steering_command_rad_ - maximum_change,
      last_steering_command_rad_ + maximum_change);

    double target_speed_mps = control_.test_speed_mps;
    if (control_.adaptive_speed_enabled) {
      const double absolute_steering = std::abs(steering_rad);
      const double steering_ratio = std::clamp(
        (absolute_steering - control_.steering_slowdown_threshold_rad) /
        std::max(
          1.0e-6,
          control_.max_steering_rad - control_.steering_slowdown_threshold_rad),
        0.0, 1.0);
      const double steering_speed = control_.maximum_speed_mps -
        steering_ratio * (control_.maximum_speed_mps - control_.minimum_speed_mps);
      const double curvature = std::abs(std::tan(steering_rad) / control_.wheelbase_m);
      const double curvature_speed = control_.maximum_speed_mps /
        (1.0 + control_.curvature_slowdown_gain * curvature);
      target_speed_mps = std::clamp(
        std::min(steering_speed, curvature_speed),
        control_.minimum_speed_mps, control_.maximum_speed_mps);
    }
    if (avoidance_.enabled &&
      (active_obstacle_valid_ || std::abs(current_path_offset_m_) > 0.005))
    {
      target_speed_mps = std::min(target_speed_mps, avoidance_.avoidance_speed_mps);
    }

    if (lane_reliable) {
      lost_frame_count_ = 0;
      has_reliable_control_ = true;
      last_reliable_control_at_ = now;
      last_reliable_steering_rad_ = steering_rad;
      controller_state_ = "TRACKING";
    } else {
      const double reliable_age_sec = has_reliable_control_ ?
        std::chrono::duration<double>(now - last_reliable_control_at_).count() :
        std::numeric_limits<double>::infinity();
      if (!has_reliable_control_ || reliable_age_sec > control_.previous_path_hold_sec) {
        publish_control(0.0, 0.0);
        controller_state_ = "LOST_LOW_CONFIDENCE";
        last_control_at_ = now;
        return;
      }
      ++lost_frame_count_;
      steering_rad = 0.65 * last_reliable_steering_rad_ + 0.35 * steering_rad;
      target_speed_mps = std::min(target_speed_mps, control_.low_confidence_speed_mps);
      controller_state_ = "LOW_CONFIDENCE";
    }
    publish_control(target_speed_mps, steering_rad);
    last_control_at_ = now;
  }

  void render_path_debug(const LaneFitResult & lane_fit)
  {
    birdseye_image_.copyTo(path_debug_image_);
    if (std::abs(current_path_offset_m_) > 0.005 && lane_fit.base_sampled_path.size() >= 2U) {
      cv::polylines(
        path_debug_image_, lane_fit.base_sampled_path, false,
        cv::Scalar(255, 120, 40), 2, cv::LINE_AA);
    }
    for (const auto & center : lane_fit.component_centers) {
      cv::circle(path_debug_image_, center, 5, cv::Scalar(255, 80, 0), 1, cv::LINE_AA);
    }
    for (const auto & point : lane_fit.left_boundary_points) {
      cv::circle(path_debug_image_, point, 3, cv::Scalar(255, 0, 255), -1, cv::LINE_AA);
    }
    for (const auto & point : lane_fit.right_boundary_points) {
      cv::circle(path_debug_image_, point, 3, cv::Scalar(255, 255, 0), -1, cv::LINE_AA);
    }
    for (const auto & point : lane_fit.detection_points) {
      cv::circle(path_debug_image_, point, 3, cv::Scalar(0, 255, 255), -1, cv::LINE_AA);
    }
    const double lane_width_ratio = 0.5 * (
      perspective_.dst_top_right_x_ratio - perspective_.dst_top_left_x_ratio +
      perspective_.dst_bottom_right_x_ratio - perspective_.dst_bottom_left_x_ratio);
    const double meters_per_pixel_x = path_detection_.lane_width_m /
      std::max(1.0, lane_width_ratio * (path_debug_image_.cols - 1));
    const double meters_per_pixel_y = path_detection_.bev_forward_range_m /
      std::max(1, path_debug_image_.rows - 1);
    for (const auto & obstacle : latest_obstacles_) {
      if (obstacle.centroid.x < 0.0 ||
        obstacle.centroid.x > path_detection_.bev_forward_range_m)
      {
        continue;
      }
      const cv::Point center(
        static_cast<int>(std::lround(
          0.5 * (path_debug_image_.cols - 1) - obstacle.centroid.y / meters_per_pixel_x)),
        static_cast<int>(std::lround(
          (path_debug_image_.rows - 1) - obstacle.centroid.x / meters_per_pixel_y)));
      if (center.x < 0 || center.x >= path_debug_image_.cols ||
        center.y < 0 || center.y >= path_debug_image_.rows)
      {
        continue;
      }
      cv::ellipse(
        path_debug_image_, center,
        cv::Size(
          static_cast<int>(std::lround(
            (avoidance_.safety_distance_m + 0.5 * obstacle.width) / meters_per_pixel_x)),
          static_cast<int>(std::lround(
            (avoidance_.safety_distance_m + 0.5 * obstacle.width) / meters_per_pixel_y))),
        0.0, 0.0, 360.0, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
      cv::circle(path_debug_image_, center, 5, cv::Scalar(0, 0, 255), -1, cv::LINE_AA);
    }
    if (lane_fit.valid && lane_fit.sampled_path.size() >= 2U) {
      cv::polylines(
        path_debug_image_, lane_fit.sampled_path, false,
        cv::Scalar(0, 255, 0), 3, cv::LINE_AA);
      cv::circle(
        path_debug_image_, lane_fit.lookahead_point, 9,
        cv::Scalar(0, 0, 255), 3, cv::LINE_AA);
    }
    const cv::Point vehicle_reference(
      path_debug_image_.cols / 2, path_debug_image_.rows - 1);
    cv::drawMarker(
      path_debug_image_, vehicle_reference, cv::Scalar(255, 255, 0),
      cv::MARKER_TRIANGLE_UP, 18, 2, cv::LINE_AA);

    std::string state = "LOST";
    cv::Scalar state_color(0, 0, 255);
    if (lane_fit.valid && lane_fit.confidence >= control_.tracking_lane_confidence) {
      state = "TRACKING";
      state_color = cv::Scalar(0, 255, 0);
    } else if (lane_fit.valid &&
      lane_fit.confidence >= path_detection_.minimum_lane_confidence)
    {
      state = "LOW_CONFIDENCE";
      state_color = cv::Scalar(0, 200, 255);
    }
    const std::string state_signature = state + "/" + lane_fit.model;
    if (state_signature != last_lane_state_) {
      RCLCPP_INFO(
        get_logger(), "Lane detection state: %s/%s (confidence %.2f, points %zu, components %d)",
        state.c_str(), lane_fit.model.c_str(), lane_fit.confidence, lane_fit.detection_points.size(),
        lane_fit.component_count);
      last_lane_state_ = state_signature;
    }

    std::ostringstream status;
    status << state << "/" << lane_fit.model << "  conf=" << std::fixed << std::setprecision(2)
           << lane_fit.confidence << "  points=" << lane_fit.detection_points.size()
           << "  rms=" << std::setprecision(1) << lane_fit.residual_rms_px << "px";
    cv::rectangle(
      path_debug_image_, cv::Rect(0, 0, path_debug_image_.cols, 68),
      cv::Scalar(0, 0, 0), -1);
    cv::putText(
      path_debug_image_, status.str(), cv::Point(7, 20), cv::FONT_HERSHEY_SIMPLEX,
      0.48, state_color, 1, cv::LINE_AA);
    std::ostringstream control_status;
    control_status << controller_state_ << "  steer=" << std::fixed << std::setprecision(1)
                   << (last_steering_command_rad_ * 180.0 / 3.14159265358979323846)
                   << "deg  speed="
                   << std::setprecision(2) << last_speed_command_mps_ << "m/s";
    cv::putText(
      path_debug_image_, control_status.str(), cv::Point(7, 40),
      cv::FONT_HERSHEY_SIMPLEX, 0.46, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
    std::ostringstream planner_status;
    planner_status << planner_state_ << "  offset=" << std::fixed << std::setprecision(2)
                   << current_path_offset_m_ << "m  obstacles=" << latest_obstacles_.size();
    cv::putText(
      path_debug_image_, planner_status.str(), cv::Point(7, 60),
      cv::FONT_HERSHEY_SIMPLEX, 0.46, cv::Scalar(160, 220, 255), 1, cv::LINE_AA);
  }

  void on_image(const sensor_msgs::msg::Image::ConstSharedPtr msg)
  {
    const auto callback_started = std::chrono::steady_clock::now();

    try {
      // Keep the source image untouched for the original debug topic. The BGR
      // copy becomes the working image for ROI and later OpenCV phases.
      const auto cv_image = cv_bridge::toCvCopy(
        msg, sensor_msgs::image_encodings::BGR8);
      if (cv_image->image.empty()) {
        RCLCPP_WARN_THROTTLE(
          get_logger(), steady_clock_, 2000, "Received an empty camera frame");
        return;
      }

      last_frame_at_ = callback_started;
      received_frame_ = true;
      stale_stop_sent_ = false;

      if (!logged_first_frame_) {
        logged_first_frame_ = true;
        RCLCPP_INFO(
          get_logger(), "First camera frame: %ux%u, encoding=%s, frame_id=%s",
          msg->width, msg->height, msg->encoding.c_str(), msg->header.frame_id.c_str());
      }

      // Preserve the source encoding and timestamp. Later phases will publish
      // additional ROI, bird's-eye and mask images on separate debug topics.
      debug_original_pub_->publish(*msg);

      update_perspective_transform(msg->width, msg->height);
      const auto bev_started = std::chrono::steady_clock::now();
      cv::warpPerspective(
        cv_image->image,
        birdseye_image_,
        perspective_matrix_,
        cv::Size(perspective_.output_width, perspective_.output_height),
        cv::INTER_LINEAR,
        cv::BORDER_CONSTANT,
        cv::Scalar(0, 0, 0));
      const auto bev_finished = std::chrono::steady_clock::now();
      const auto birdseye_msg = cv_bridge::CvImage(
        msg->header, sensor_msgs::image_encodings::BGR8, birdseye_image_).toImageMsg();
      debug_birdseye_pub_->publish(*birdseye_msg);

      const auto threshold_started = std::chrono::steady_clock::now();
      cv::cvtColor(birdseye_image_, hsv_image_, cv::COLOR_BGR2HSV);
      cv::inRange(
        hsv_image_,
        cv::Scalar(orange_mask_.h_min, orange_mask_.s_min, orange_mask_.v_min),
        cv::Scalar(orange_mask_.h_max, orange_mask_.s_max, orange_mask_.v_max),
        orange_binary_mask_);
      if (morphology_kernels_dirty_) {
        update_morphology_kernels();
      }
      cv::morphologyEx(
        orange_binary_mask_, orange_binary_mask_, cv::MORPH_OPEN, open_kernel_,
        cv::Point(-1, -1), orange_mask_.morphology_iterations);
      cv::morphologyEx(
        orange_binary_mask_, orange_binary_mask_, cv::MORPH_CLOSE, close_kernel_,
        cv::Point(-1, -1), orange_mask_.morphology_iterations);

      cv::inRange(
        hsv_image_, cv::Scalar(0, 0, white_mask_.value_min),
        cv::Scalar(179, white_mask_.saturation_max, 255), white_binary_mask_);
      if (white_morphology_kernels_dirty_) {
        update_white_morphology_kernels();
      }
      cv::morphologyEx(
        white_binary_mask_, white_binary_mask_, cv::MORPH_OPEN, white_open_kernel_,
        cv::Point(-1, -1), white_mask_.morphology_iterations);
      cv::morphologyEx(
        white_binary_mask_, white_binary_mask_, cv::MORPH_CLOSE, white_close_kernel_,
        cv::Point(-1, -1), white_mask_.morphology_iterations);
      const auto threshold_finished = std::chrono::steady_clock::now();
      const auto mask_msg = cv_bridge::CvImage(
        msg->header, sensor_msgs::image_encodings::MONO8,
        orange_binary_mask_).toImageMsg();
      debug_mask_pub_->publish(*mask_msg);
      const auto white_mask_msg = cv_bridge::CvImage(
        msg->header, sensor_msgs::image_encodings::MONO8,
        white_binary_mask_).toImageMsg();
      debug_white_mask_pub_->publish(*white_mask_msg);

      const auto path_started = std::chrono::steady_clock::now();
      const auto orange_lane_fit = detect_orange_path(orange_binary_mask_);
      const auto white_lane_fit = detect_white_path(white_binary_mask_);
      LaneFitResult lane_fit = orange_lane_fit;
      if (white_lane_fit.valid &&
        (!orange_lane_fit.valid ||
        orange_lane_fit.confidence < path_detection_.minimum_lane_confidence ||
        (orange_lane_fit.confidence < control_.tracking_lane_confidence &&
        white_lane_fit.confidence > orange_lane_fit.confidence)))
      {
        lane_fit = white_lane_fit;
      } else {
        lane_fit.left_boundary_points = white_lane_fit.left_boundary_points;
        lane_fit.right_boundary_points = white_lane_fit.right_boundary_points;
      }
      apply_obstacle_avoidance(lane_fit, callback_started);
      update_controller(lane_fit, callback_started);
      render_path_debug(lane_fit);
      const auto path_finished = std::chrono::steady_clock::now();
      const auto path_msg = cv_bridge::CvImage(
        msg->header, sensor_msgs::image_encodings::BGR8,
        path_debug_image_).toImageMsg();
      debug_path_pub_->publish(*path_msg);

      const auto points = roi_points(msg->width, msg->height);
      const std::vector<cv::Point> polygon(points.begin(), points.end());
      cv::polylines(
        cv_image->image,
        std::vector<std::vector<cv::Point>>{polygon},
        true,
        cv::Scalar(0, 255, 0),
        roi_.line_thickness,
        cv::LINE_AA);
      debug_roi_pub_->publish(*cv_image->toImageMsg());

      last_bev_duration_us_ = std::chrono::duration_cast<std::chrono::microseconds>(
        bev_finished - bev_started).count();
      last_threshold_duration_us_ = std::chrono::duration_cast<std::chrono::microseconds>(
        threshold_finished - threshold_started).count();
      last_path_duration_us_ = std::chrono::duration_cast<std::chrono::microseconds>(
        path_finished - path_started).count();
    } catch (const cv_bridge::Exception & error) {
      RCLCPP_ERROR_THROTTLE(
        get_logger(), steady_clock_, 2000,
        "cv_bridge rejected camera frame: %s", error.what());
      return;
    } catch (const std::exception & error) {
      RCLCPP_ERROR_THROTTLE(
        get_logger(), steady_clock_, 2000,
        "Camera callback failed safely: %s", error.what());
      return;
    }

    if (timing_log_enabled_) {
      const auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now() - callback_started).count();
      RCLCPP_INFO_THROTTLE(
        get_logger(), steady_clock_, timing_log_period_ms_,
        "camera callback: %.3f ms (BEV: %.3f ms, threshold: %.3f ms, path: %.3f ms)",
        static_cast<double>(elapsed_us) / 1000.0,
        static_cast<double>(last_bev_duration_us_) / 1000.0,
        static_cast<double>(last_threshold_duration_us_) / 1000.0,
        static_cast<double>(last_path_duration_us_) / 1000.0);
    }
  }

  void check_frame_timeout()
  {
    const auto now = std::chrono::steady_clock::now();
    const auto reference = received_frame_ ? last_frame_at_ : started_at_;
    const double age_sec = std::chrono::duration<double>(now - reference).count();
    if (age_sec <= frame_timeout_sec_) {
      return;
    }

    if (received_frame_) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), steady_clock_, 2000,
        "Camera frame is stale (last frame %.2f s ago)", age_sec);
    } else {
      RCLCPP_WARN_THROTTLE(
        get_logger(), steady_clock_, 2000,
        "Waiting for the first camera frame (%.2f s)", age_sec);
    }
    if (control_.enabled && !stale_stop_sent_) {
      publish_control(0.0, 0.0);
      controller_state_ = "CAMERA_TIMEOUT";
      stale_stop_sent_ = true;
      RCLCPP_ERROR(get_logger(), "Camera watchdog published a safe stop");
    }
  }

  rclcpp::Clock steady_clock_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  rclcpp::Subscription<physicar_interfaces::msg::ObstacleArray>::SharedPtr obstacle_sub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr debug_original_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr debug_roi_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr debug_birdseye_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr debug_mask_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr debug_white_mask_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr debug_path_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr speed_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr steering_pub_;
  rclcpp::TimerBase::SharedPtr frame_watchdog_;
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr parameter_callback_;

  std::chrono::steady_clock::time_point started_at_;
  std::chrono::steady_clock::time_point last_frame_at_;
  double frame_timeout_sec_{1.0};
  int64_t timing_log_period_ms_{2000};
  int64_t last_bev_duration_us_{0};
  int64_t last_threshold_duration_us_{0};
  int64_t last_path_duration_us_{0};
  bool timing_log_enabled_{false};
  bool received_frame_{false};
  bool logged_first_frame_{false};
  RoiConfig roi_;
  PerspectiveConfig perspective_;
  bool perspective_dirty_{true};
  uint32_t transform_source_width_{0};
  uint32_t transform_source_height_{0};
  cv::Mat perspective_matrix_;
  cv::Mat birdseye_image_;
  OrangeMaskConfig orange_mask_;
  bool morphology_kernels_dirty_{true};
  cv::Mat hsv_image_;
  cv::Mat orange_binary_mask_;
  cv::Mat open_kernel_;
  cv::Mat close_kernel_;
  WhiteMaskConfig white_mask_;
  bool white_morphology_kernels_dirty_{true};
  cv::Mat white_binary_mask_;
  cv::Mat white_open_kernel_;
  cv::Mat white_close_kernel_;
  PathDetectionConfig path_detection_;
  cv::Mat component_labels_;
  cv::Mat filtered_component_mask_;
  cv::Mat path_debug_image_;
  std::string last_lane_state_;
  ControlConfig control_;
  std::chrono::steady_clock::time_point last_control_at_{};
  double last_speed_command_mps_{0.0};
  double last_steering_command_rad_{0.0};
  std::chrono::steady_clock::time_point last_reliable_control_at_{};
  double last_reliable_steering_rad_{0.0};
  int lost_frame_count_{0};
  bool has_reliable_control_{false};
  std::string controller_state_{"DISABLED"};
  bool stale_stop_sent_{false};
  AvoidanceConfig avoidance_;
  std::vector<physicar_interfaces::msg::Obstacle> latest_obstacles_;
  std::chrono::steady_clock::time_point last_obstacle_message_at_{};
  std::chrono::steady_clock::time_point last_collision_at_{};
  std::chrono::steady_clock::time_point active_obstacle_last_seen_at_{};
  std::chrono::steady_clock::time_point last_planner_at_{};
  double current_path_offset_m_{0.0};
  double avoidance_direction_sign_{1.0};
  double active_obstacle_forward_m_{0.0};
  double active_obstacle_lateral_m_{0.0};
  std::string planner_state_{"LANE_FOLLOW"};
  std::string last_logged_planner_state_;
  bool received_obstacles_{false};
  bool obstacle_data_stale_{true};
  bool obstacle_path_blocked_{false};
  bool active_obstacle_valid_{false};
};

}  // namespace physicar_autonomy

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<physicar_autonomy::LaneFollowNode>());
  rclcpp::shutdown();
  return 0;
}
