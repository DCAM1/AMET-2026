// Copyright 2026 AICASTLE Inc.
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <cmath>
#include <optional>

#include <opencv2/core.hpp>

namespace physicar_autonomy
{

struct VehiclePoint
{
  double forward_m{0.0};
  double lateral_left_m{0.0};
};

class BevGeometry
{
public:
  BevGeometry(
    int width, int height, double lane_width_pixels,
    double lane_width_m, double forward_range_m)
  : width_(width),
    height_(height),
    meters_per_pixel_x_(lane_width_m / lane_width_pixels),
    meters_per_pixel_y_(forward_range_m / static_cast<double>(height - 1)),
    vehicle_x_px_(0.5 * static_cast<double>(width - 1))
  {
  }

  bool valid() const
  {
    return width_ >= 2 && height_ >= 2 &&
           std::isfinite(meters_per_pixel_x_) && meters_per_pixel_x_ > 0.0 &&
           std::isfinite(meters_per_pixel_y_) && meters_per_pixel_y_ > 0.0;
  }

  std::optional<cv::Point2d> vehicleToBevPixel(
    double forward_m, double lateral_left_m) const
  {
    if (!valid() || !std::isfinite(forward_m) || !std::isfinite(lateral_left_m)) {
      return std::nullopt;
    }
    return cv::Point2d(
      vehicle_x_px_ - lateral_left_m / meters_per_pixel_x_,
      static_cast<double>(height_ - 1) - forward_m / meters_per_pixel_y_);
  }

  std::optional<VehiclePoint> bevPixelToVehicle(double pixel_x, double pixel_y) const
  {
    if (!valid() || !std::isfinite(pixel_x) || !std::isfinite(pixel_y)) {
      return std::nullopt;
    }
    return VehiclePoint{
      (static_cast<double>(height_ - 1) - pixel_y) * meters_per_pixel_y_,
      (vehicle_x_px_ - pixel_x) * meters_per_pixel_x_};
  }

  double metersPerPixelX() const {return meters_per_pixel_x_;}
  double metersPerPixelY() const {return meters_per_pixel_y_;}
  int width() const {return width_;}
  int height() const {return height_;}

private:
  int width_;
  int height_;
  double meters_per_pixel_x_;
  double meters_per_pixel_y_;
  double vehicle_x_px_;
};

}  // namespace physicar_autonomy
