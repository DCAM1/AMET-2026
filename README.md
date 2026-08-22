# AME2026 PhysiCar Autonomy

Raspberry Pi 5와 ROS 2 Jazzy를 대상으로 만든 OpenCV 기반 PhysiCar 대회용
자율주행 패키지입니다. SLAM, 전역 지도 localization, 딥러닝 차선 검출은
사용하지 않습니다.

현재 구현 범위는 다음과 같습니다.

- 주황색 중앙 점선 우선 검출과 흰색 양쪽 경계선 fallback
- 사다리꼴 ROI와 Bird's-Eye View
- 점선 컴포넌트 연결, spline 및 polynomial 경로 fitting
- 속도 연동 lookahead와 Pure Pursuit 조향
- 차선 신뢰도에 따른 감속, 짧은 이전 경로 유지, 안전 정지
- LiDAR 전방 클러스터링
- 라바콘 충돌 판정, 좌·우 local path offset, 중앙 차선 복귀
- 빨간불 정지 및 시작점 초록불 확인 후 출발
- RViz/rqt용 카메라·경로·장애물 디버그 출력

## 주행 파이프라인

```text
Camera 480x360
  -> Trapezoid ROI
  -> Bird's-Eye View
  -> Orange center line / White boundary fallback
  -> Local center path
  -> LiDAR obstacle path offset
  -> Speed-dependent lookahead
  -> Pure Pursuit
  -> /speed, /steering
```

차선 경로와 LiDAR 장애물은 현재 프레임의 로컬 정보만 사용합니다. 장애물을
지도에 누적하거나 SLAM에 사용하지 않습니다.

## 시작 신호등 동작

신호등 게이트는 기본적으로 닫혀 있습니다.

1. `RED`, `YELLOW`, `UNKNOWN`에서는 속도 `0`을 유지합니다.
2. 카메라 오른쪽 ROI에서 `GREEN`이 기본 5프레임 연속 검출되어야 출발합니다.
3. 차량이 한 번 출발하면 상태가 `RELEASED`로 고정됩니다.
4. 주행 중 다른 빨간색이나 초록색 물체가 검출돼도 차량을 다시 정지시키지
   않습니다.
5. `control_enabled`를 `false`로 바꾸면 게이트가 초기화됩니다. 다음 활성화
   때는 초록불을 다시 확인해야 합니다.

이 로직은 `lane_follow_node`가 발행하는 명령만 제한합니다. 웹 앱, 키보드,
게임패드 등 다른 노드가 `/speed`를 동시에 발행하면 신호등이 빨간불이어도 그
외부 명령으로 차량이 움직일 수 있습니다. 자율주행 중에는 수동 스로틀을
중립으로 두고 아래 명령으로 발행자 수를 반드시 확인하십시오.

```bash
ros2 topic info /speed --verbose
ros2 topic info /steering --verbose
```

## 확인된 ROS 2 인터페이스

| 용도 | 토픽 | 메시지 타입 |
|---|---|---|
| 카메라 입력 | `/camera/image_raw` | `sensor_msgs/msg/Image` |
| LiDAR 입력 | `/scan` | `sensor_msgs/msg/LaserScan` |
| 속도 명령 | `/speed` | `std_msgs/msg/Float64` (m/s) |
| 조향 명령 | `/steering` | `std_msgs/msg/Float64` (rad) |
| 장애물 목록 | `/obstacles` | `physicar_interfaces/msg/ObstacleArray` |
| 장애물 마커 | `/obstacle/debug/markers` | `visualization_msgs/msg/MarkerArray` |

조향각은 `-0.349066`~`+0.349066 rad`, 즉 약 ±20도로 제한됩니다. 실제 차량의
카메라 드라이버는 640x480을 보정·축소하여 `/camera/image_raw`에 480x360 영상을
발행하는 기존 PhysiCar bringup 구성을 전제로 합니다.

## 저장소 구조

```text
physicar_autonomy/
├── CMakeLists.txt
├── package.xml
├── config/
│   ├── lane_follow.yaml
│   └── lidar_obstacle.yaml
├── launch/lane_follow.launch.py
├── rviz/lane_follow.rviz
└── src/
    ├── lane_follow_node.cpp
    └── lidar_obstacle_node.cpp
```

## 실차 설치 전 필수 조건

- Ubuntu 24.04
- ROS 2 Jazzy
- 기존 PhysiCar ROS 2 workspace와 하드웨어 bringup
- `cv_bridge`, OpenCV, RViz2 및 이 패키지의 `package.xml` 의존성
- `physicar_interfaces`에 아래 두 메시지가 등록되어 있어야 함
  - `msg/Obstacle.msg`
  - `msg/ObstacleArray.msg`

이 GitHub 저장소는 이전 요청에 따라 `physicar_autonomy` 패키지만 담고
있습니다. 따라서 기존 PhysiCar workspace의 `physicar_interfaces`가 위 두
메시지를 제공하지 않으면 이 저장소만 clone해서는 빌드되지 않습니다. 개발
컴퓨터의 다음 파일과 변경 내용을 실차에도 먼저 동기화해야 합니다.

```text
src/physicar-ros/physicar_interfaces/msg/Obstacle.msg
src/physicar-ros/physicar_interfaces/msg/ObstacleArray.msg
src/physicar-ros/physicar_interfaces/CMakeLists.txt
src/physicar-ros/physicar_interfaces/package.xml
```

빌드 전에 다음 두 명령이 모두 성공하는지 확인하십시오.

```bash
source /opt/ros/jazzy/setup.bash
source /opt/physicar/install/setup.bash
ros2 interface show physicar_interfaces/msg/Obstacle
ros2 interface show physicar_interfaces/msg/ObstacleArray
```

`Could not find the interface`가 나오면 `physicar_interfaces` 소스와 빌드를 먼저
업데이트해야 합니다.

## Raspberry Pi 5에 설치

아래 예시는 기존 실차 workspace가 `/opt/physicar`에 있다고 가정합니다. 실제
설치 경로가 다르면 해당 경로만 바꾸십시오. 기존 폴더를 삭제하거나 덮어쓰기
전에 반드시 별도로 백업하십시오.

처음 설치하는 경우:

```bash
cd /opt/physicar/src/physicar-ros
git clone https://github.com/YJS906/AME2026_Hackathon.git physicar_autonomy
```

이미 이 저장소를 clone한 경우:

```bash
cd /opt/physicar/src/physicar-ros/physicar_autonomy
git pull --ff-only
```

의존성을 설치하고 패키지를 빌드합니다.

```bash
cd /opt/physicar
source /opt/ros/jazzy/setup.bash
rosdep install --from-paths src/physicar-ros/physicar_autonomy \
  --ignore-src --rosdistro jazzy -r -y
colcon build --symlink-install --packages-up-to physicar_autonomy
source /opt/physicar/install/setup.bash
```

빌드 확인:

```bash
ros2 pkg prefix physicar_autonomy
ros2 pkg executables physicar_autonomy
```

정상이라면 `lane_follow_node`와 `lidar_obstacle_node`가 표시됩니다.

## 실차 센서 확인

먼저 기존 PhysiCar 하드웨어 bringup을 실행합니다. 실차에서
`physicar.service`가 자동으로 bringup을 실행 중이라면 같은 launch를 중복으로
실행하지 마십시오.

```bash
sudo systemctl status physicar.service
```

서비스를 사용하지 않는 개발 환경에서는 기존 프로젝트의 실차 launch를 사용할
수 있습니다.

```bash
source /opt/ros/jazzy/setup.bash
source /opt/physicar/install/setup.bash
ros2 launch physicar_bringup real.launch.py
```

새 터미널에서 센서 계약을 확인합니다.

```bash
source /opt/ros/jazzy/setup.bash
source /opt/physicar/install/setup.bash

ros2 topic type /camera/image_raw
ros2 topic type /scan
ros2 topic hz /camera/image_raw
ros2 topic hz /scan
ros2 topic echo /camera/camera_info --once
```

카메라는 `sensor_msgs/msg/Image`, LiDAR는 `sensor_msgs/msg/LaserScan`이어야
합니다. 카메라가 480x360이 아니거나 장착 각도·노출이 시뮬레이터와 다르면 ROI,
Bird's-Eye View, HSV를 실차 영상에 맞춰 다시 보정해야 합니다.

## 첫 실행: 제어 비활성화

처음에는 바퀴가 바닥에서 뜬 상태 또는 차량이 움직일 수 없는 안전한 장소에서
`control_enabled:=false`로 perception만 실행하십시오.

```bash
source /opt/ros/jazzy/setup.bash
source /opt/physicar/install/setup.bash
ros2 launch physicar_autonomy lane_follow.launch.py \
  use_rviz:=false use_lidar:=true control_enabled:=false
```

라즈베리파이에 모니터가 연결되어 있으면 `use_rviz:=true`로 바꿀 수 있습니다.
원격 PC에서 같은 ROS_DOMAIN_ID와 DDS 설정을 사용하는 경우 RViz 또는
`rqt_image_view`에서 다음 토픽을 확인할 수 있습니다.

```bash
rqt_image_view /lane/debug/path
```

디버그 영상 토픽:

- `/lane/debug/original`: 원본 영상, 신호등 ROI·검출 박스·상태
- `/lane/debug/roi`: 사다리꼴 ROI overlay
- `/lane/debug/birdseye`: Bird's-Eye View
- `/lane/debug/mask`: 주황색 binary mask
- `/lane/debug/white_mask`: 흰색 경계 binary mask
- `/lane/debug/path`: 검출점, 경로, lookahead, 조향, 속도, 상태

정상 확인 기준:

- 차선 상태가 `TRACKING` 또는 안정적인 `LOW_CONFIDENCE`
- 주황색 점선 중심 또는 흰색 경계 fallback이 실제 차선과 일치
- 초록 fitted path가 차선 중앙을 따라감
- lookahead point가 경로 위에 있음
- 시작점 빨간불에서 `signal=RED` 및 속도 `0`
- 초록불 연속 확인 후 `signal=RELEASED`
- RViz에서 라바콘 cluster가 실제 LiDAR 위치와 일치

## 제어 활성화와 긴급 비활성화

제어 활성화:

```bash
ros2 param set /lane_follow control_enabled true
```

즉시 제어 비활성화:

```bash
ros2 param set /lane_follow control_enabled false
```

비활성화하면 이 노드는 속도와 조향을 `0`으로 발행하고 시작 신호등 게이트를
초기화합니다. 다른 `/speed` 발행자가 존재하면 그 발행자도 별도로 중지해야
합니다.

현재 값 확인:

```bash
ros2 param get /lane_follow max_speed_mps
ros2 param get /lane_follow min_speed_mps
ros2 param get /lane_follow min_lookahead_distance_m
ros2 param get /lane_follow max_lookahead_distance_m
ros2 param get /lane_follow traffic_light_gate_enabled
```

## 실차 첫 주행 권장 설정

현재 `config/lane_follow.yaml`의 속도는 시뮬레이터에서 조정한 값입니다.

```yaml
test_speed_mps: 1.10
min_speed_mps: 0.60
max_speed_mps: 1.30
avoidance_speed_mps: 0.60
```

이 값으로 실차 첫 주행을 시작하지 마십시오. 타이어 마찰, 카메라 지연, 조향
유격과 모터 응답이 시뮬레이터와 다릅니다. 첫 실차 시험은 예를 들어 다음처럼
0.4~0.6 m/s 범위에서 시작하고, 매 단계 완주가 안정적일 때만 조금씩 올리는
것을 권장합니다.

```yaml
test_speed_mps: 0.40
min_speed_mps: 0.25
max_speed_mps: 0.50
avoidance_speed_mps: 0.25
```

설치된 YAML이 아니라 source의
`config/lane_follow.yaml`을 수정했다면 다시 `colcon build` 후 launch를
재시작해야 합니다. `--symlink-install` 환경에서도 안전하게 재빌드하는 것을
권장합니다.

## 주요 튜닝 파라미터

모든 기본값은 `config/lane_follow.yaml`과 `config/lidar_obstacle.yaml`에
있습니다.

| 영역 | 주요 파라미터 |
|---|---|
| ROI | `roi_*_ratio` |
| Perspective | `perspective_dst_*`, `bev_forward_range_m` |
| 주황색 | `orange_h_*`, `orange_s_*`, `orange_v_*` |
| 흰색 fallback | `white_s_max`, `white_v_min`, `lane_width_m` |
| 경로 | `minimum_lane_confidence`, `fit_max_residual_px` |
| Lookahead | `min_lookahead_distance_m`, `max_lookahead_distance_m`, `lookahead_speed_gain` |
| 조향 | `wheelbase_m`, `max_steering_rad`, `max_steering_rate_rad_s` |
| 속도 | `min_speed_mps`, `max_speed_mps`, `curvature_slowdown_gain` |
| 신호등 | `traffic_light_roi_*`, `traffic_light_*_hue_*`, `traffic_light_green_confirm_frames` |
| LiDAR | `angle_*_deg`, `range_*_m`, `cluster_distance_m` |
| 회피 | `obstacle_detection_distance_m`, `obstacle_safety_distance_m`, `avoidance_lateral_offset_m` |

- `max_obstacle_fusion_age_sec`: 카메라와 장애물 메시지 timestamp 차이 허용값
- `lidar_to_bev_forward_offset_m`: LiDAR obstacle centroid에 더하는 전방 고정 offset
- `lidar_to_bev_lateral_offset_m`: LiDAR obstacle centroid에 더하는 좌우 고정 offset

### 신호등 실차 보정

기본 신호등 ROI는 시뮬레이터의 480x360 영상에 맞춰져 있습니다.

```yaml
traffic_light_roi_left_ratio: 0.68
traffic_light_roi_right_ratio: 0.95
traffic_light_roi_top_ratio: 0.35
traffic_light_roi_bottom_ratio: 0.75
traffic_light_green_confirm_frames: 5
```

실차 카메라 위치, tilt, 자동 노출과 대회 조명에 따라 ROI와 HSV가 달라질 수
있습니다. `/lane/debug/original`에서 사각형 안에 신호등 전체가 들어오는지 먼저
확인하고, 빨간불 사진과 초록불 사진을 각각 확보한 뒤 threshold를 조정하십시오.
확실하지 않은 프레임은 `UNKNOWN`으로 처리되어 정지하는 것이 정상입니다.

### 차선과 Lookahead 보정 순서

1. 차량을 움직이지 않고 ROI가 양쪽 흰 경계와 중앙 점선을 포함하는지 확인
2. Bird's-Eye View에서 차선 폭이 지나치게 좁거나 잘리지 않는지 확인
3. 주황색 mask가 노면이나 라바콘을 과검출하지 않는지 확인
4. 흰색 fallback이 양쪽 경계를 올바르게 구분하는지 확인
5. 0.4 m/s에서 path와 lookahead가 급커브 안쪽을 과도하게 자르지 않는지 확인
6. 필요하면 lookahead를 줄이고 steering rate를 낮춰 단계적으로 시험
7. 차선주행이 안정된 뒤 LiDAR 회피를 활성화하여 라바콘 간격 확인

## 상태와 안전 정지

주요 controller/planner 상태:

- `WAIT_TRAFFIC_RED`, `WAIT_TRAFFIC_GREEN`: 초록불 출발 허가 대기
- `TRACKING`: 정상 차선 추종
- `LOW_CONFIDENCE`: 감속 추종
- `LOW_CONFIDENCE_HOLD`: 짧은 시간 이전 경로 유지
- `LOST`, `CAMERA_TIMEOUT`: 안전 정지
- `AVOID_LEFT`, `AVOID_RIGHT`: 라바콘 회피
- `RETURN_TO_LANE`: 중앙 경로 복귀
- `LIDAR_STALE`, `OBSTACLE_BLOCKED`: 안전 정지

카메라 frame 없음, fitting 실패, NaN/Inf, 유효 target 없음, LiDAR timeout 또는
회피 공간 부족이 발생하면 노드는 속도와 조향을 0으로 보냅니다. 이전 조향을
무한정 유지하지 않습니다.

## 문제 해결

### `Package 'physicar_autonomy' not found`

현재 터미널에서 build 결과를 source했는지 확인합니다.

```bash
source /opt/ros/jazzy/setup.bash
source /opt/physicar/install/setup.bash
```

### `physicar_interfaces/msg/ObstacleArray` 관련 build 오류

실차의 `physicar_interfaces`에 `Obstacle.msg`, `ObstacleArray.msg`가 없거나
CMake에 등록되지 않은 상태입니다. 위의 **실차 설치 전 필수 조건**을 먼저
완료하고 `physicar_interfaces`부터 다시 빌드하십시오.

### 노드가 `Invalid obstacle avoidance parameters`로 종료됨

`obstacle_detection_distance_m`가 `bev_forward_range_m`보다 크지 않은지
확인합니다.

```text
obstacle_detection_distance_m <= bev_forward_range_m
```

### 빨간불인데 차량이 움직임

먼저 `/speed`와 `/steering`의 발행자를 확인합니다. 웹 앱이나 수동 제어 노드가
동시에 직접 명령을 발행하면 해당 발행자를 중지하거나 중립으로 바꿔야 합니다.

```bash
ros2 topic info /speed --verbose
ros2 topic echo /speed
```

### 초록불인데 출발하지 않음

`/lane/debug/original`의 신호등 ROI와 상태를 확인합니다. ROI, HSV, 밝기 또는
검은 하우징 비율 검사가 실차 환경과 맞지 않을 수 있습니다. 또한 차선이
`LOST`이거나 LiDAR가 stale이면 초록불이어도 안전 정지합니다.

### Debug 영상은 보이지만 제어되지 않음

`control_enabled`와 출력 토픽 타입을 확인합니다.

```bash
ros2 param get /lane_follow control_enabled
ros2 topic type /speed
ros2 topic type /steering
```

## 시뮬레이터 실행 예시

호스트에서 simulator를 시작한 뒤 다음 한 줄로 perception과 RViz를 실행합니다.

```bash
docker exec -it physicar-sim bash -lc 'source /opt/ros/jazzy/setup.bash && source /opt/physicar/install/setup.bash && export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp && export CYCLONEDDS_URI=file:///opt/physicar/src/physicar-ros/deploy/cyclonedds.xml && ros2 launch physicar_autonomy lane_follow.launch.py use_rviz:=true use_lidar:=true control_enabled:=false'
```

호스트에서 `config/lane_follow.yaml`을 수정했다면 현재 Docker 구성이 source를
image 안에 bake하는지 bind mount하는지 확인하십시오. 설치 공간의 YAML을
사용하는 launch에서는 보통 재빌드 후 재실행해야 변경이 확실히 반영됩니다.

## License

GPL-3.0-or-later
