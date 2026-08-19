# PhysiCar autonomy

Raspberry Pi 5에서 실행할 수 있는 OpenCV 기반 대회용 자율주행 패키지입니다.
SLAM, 전역 지도 localization, 딥러닝 차선 검출은 사용하지 않습니다.

## Phase 1: camera debug

시뮬레이터를 실행한 뒤 컨테이너 셸에서 다음 명령을 실행합니다.

```bash
source /opt/ros/jazzy/setup.bash
source /opt/physicar/install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=file:///opt/physicar/src/physicar-ros/deploy/cyclonedds.xml
ros2 launch physicar_autonomy lane_follow.launch.py use_rviz:=true
```

현재 단계는 `/camera/image_raw`을 구독하여 `/lane/debug/original`, ROI가
그려진 `/lane/debug/roi`, 변환된 `/lane/debug/birdseye`, HSV 주황색 마스크
`/lane/debug/mask`, 점선 검출점·2D 스플라인 경로·룩어헤드·신뢰도를 합성한
`/lane/debug/path`를 발행하며 아직 차량 제어 명령은 발행하지 않습니다.
`use_rviz:=true`이면 RViz의 `Lane Debug Image`에 최종 경로가 표시됩니다.
RViz 없이 rqt로 확인하려면 별도
셸에서 다음 명령을 실행합니다.

```bash
source /opt/ros/jazzy/setup.bash
source /opt/physicar/install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export CYCLONEDDS_URI=file:///opt/physicar/src/physicar-ros/deploy/cyclonedds.xml
ros2 run rqt_image_view rqt_image_view /lane/debug/path
```

정상이라면 노드 로그에 `480x360`, `rgb8` 프레임 정보가 한 번 표시되고,
RViz 또는 rqt 창에서 노란 검출점, 초록 피팅 경로, 빨간 룩어헤드 점과
`TRACKING` 상태가 보여야 합니다. 파란 원은 유효한 점선 컴포넌트 중심입니다.
급커브는 점선 컴포넌트를 차량 앞부터 순서대로 연결한 파라메트릭 스플라인으로
표현하며, 컴포넌트가 부족한 프레임에만 `x(y)` 다항식을 폴백으로 사용합니다.

## Phase 6: Pure Pursuit 저속 시험

기본 설정은 `control_enabled: false`라서 위 명령만으로 차량이 움직이지 않습니다.
경로가 `TRACKING`인지 먼저 확인한 뒤 다음처럼 0.4 m/s 저속 제어를 켭니다.

```bash
ros2 launch physicar_autonomy lane_follow.launch.py use_rviz:=true control_enabled:=true
```

노드는 기존 계약 그대로 `/speed`와 `/steering`에 각각
`std_msgs/msg/Float64`를 발행합니다. 조향은 라디안이며 ±20도로 포화되고,
변화율도 제한됩니다. 차선이 유효하지 않거나 카메라 프레임이 1초 이상
끊기면 속도와 조향을 0으로 발행합니다.

## Phase 7: white-boundary fallback

주황 점선 신뢰도가 기준보다 낮아지면 `/lane/debug/white_mask`에서 양쪽 흰색
경계를 행 단위로 찾고 중앙을 복원합니다. 양쪽 경계가 있으면 정확한 중점을,
한쪽만 있으면 저장된 트랙 폭 0.70 m를 이용한 임시 중심을 사용합니다.
최종 화면에서 왼쪽 경계 검출점은 자홍색, 오른쪽은 청록색이며 상태의 모델명이
`WHITE_BOTH` 또는 `WHITE_SINGLE`로 바뀝니다.

## Phase 8: adaptive speed and confidence states

`adaptive_speed_enabled`가 켜지면 직선은 `max_speed_mps`(기본 0.50 m/s),
조향·곡률이 큰 구간은 `min_speed_mps`(기본 0.25 m/s)까지 자동 감속합니다.
속도에 따라 룩어헤드도 0.35~0.65 m 안에서 바뀝니다. 상태는
`TRACKING`, `LOW_CONFIDENCE`, `LOST`로 구분하며, 완전한 검출 실패 시에는
최대 5프레임·0.35초 동안만 이전의 신뢰 가능한 조향을 0.20 m/s로 유지한 뒤
반드시 정지합니다. 흰 경계 폴백이 정상이라면 점선 공백은 대부분 이 유지 상태에
들어가기 전에 복구됩니다.

## Phase 9: LiDAR obstacle clustering

같은 launch가 기본으로 `lidar_obstacle_node`도 실행합니다. 이 노드는 실제
`/scan`(`sensor_msgs/msg/LaserScan`)에서 전방 -75~+75도, 0.15~3.0 m만 골라
스캔 순서를 이용한 순차 유클리드 클러스터링을 수행합니다. 결과는
`/obstacles`(`physicar_interfaces/msg/ObstacleArray`)에 centroid, width,
distance, angle, point count로 발행하고 `/obstacle/debug/markers`에는 빨간
원기둥으로 표시합니다. RViz에는 원본 LiDAR와 클러스터 마커가 함께 보입니다.
`use_lidar:=false`로 이 노드만 끌 수 있습니다.

```bash
ros2 topic echo /obstacles
ros2 param set /lidar_obstacle timing_log_enabled true
```

## Phase 10: local obstacle avoidance

차선 노드는 `/obstacles`를 받아 기본 차선 경로가 장애물 반폭+0.14 m 이내로
충돌하는 전방 0.25~1.50 m 클러스터만 고려합니다. 좌·우 0.27 m 후보 중 여유가 큰
방향을 택해 `AVOID_LEFT` 또는 `AVOID_RIGHT`로 전환하고, 회피 중에는 0.25 m/s로
감속합니다. 활성 라바콘을 scan 좌표에서 프레임 간 연계해 추적하므로, 그것을
통과한 뒤 다음 라바콘에 대해서는 좌·우 여유를 다시 계산합니다. 장애물이 사라지면
`RETURN_TO_LANE`을 거쳐 약 0.80 m에 걸쳐 중앙으로 복귀합니다. 최종 debug
영상에서 파란 선은 원래 차선 경로, 초록 선은 제어에 쓰는 offset 경로,
빨간 점·타원은 LiDAR 장애물과 안전 반경입니다. LiDAR 정보가 0.60초 이상
끊기면 장애물 회피가 켜진 상태에서는 안전 정지합니다.
두 후보 모두 필요한 여유를 만들 수 없을 때도 `OBSTACLE_BLOCKED`로 정지합니다.

현재 기본 배치에서는 `cone2`를 왼쪽으로 회피한 직후 `cone4`를 오른쪽으로
회피하는 연속 방향 전환과 전체 388-point 코스 1바퀴를 0.25~0.50 m/s에서
검증했습니다. 검증 후에도 `control_enabled`는 안전을 위해 기본값 `false`입니다.
