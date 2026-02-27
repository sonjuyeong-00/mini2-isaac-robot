# E0509 Lift (`joint_pos_env_cfg_1.py`) 가이드

이 문서는 [`joint_pos_env_cfg_1.py`](/home/jiwoo/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/config/e0509/joint_pos_env_cfg_1.py)를 이해하고,  
`reach -> grip -> lift` 순서로 단계 학습하는 방법을 쉽게 정리한 문서입니다.

## 1. 이 파일이 하는 일

- 기본 Lift 환경(`LiftEnvCfg`)을 상속해 E0509 로봇용 환경으로 커스터마이즈합니다.
- 로봇/큐브/테이블/센서/리셋 범위/보상 임계값을 오버라이드합니다.
- 보상은 `E0509_TRAIN_STAGE`(환경변수)로 단계별 전환됩니다.

## 2. 주요 구성 요소

- 클래스
  - `E0509CubeLiftEnvCfg`: 학습용 메인 설정
  - `E0509CubeLiftEnvCfg_PLAY`: 테스트/플레이용 축소 설정
- 로봇
  - `E0509_CFG` 사용
  - 베이스 위치: `(-0.45, 0.0, -0.225)`
  - 팔 제어: `JointPositionActionCfg(joint_[1-6], scale=0.5)`
  - 그리퍼 제어: `BinaryJointPositionActionCfg(rh_l1/rh_r1/rh_l2/rh_r2)`
- 오브젝트/테이블
  - 큐브: DexCube USD (`scale=(0.8, 0.8, 0.8)`)
  - 테이블: 고정 Cuboid (`size=(1.2, 0.8, 0.8)`)
- 센서
  - `FrameTransformerCfg`로 `link_6` 위치 추적
  - `ContactSensorCfg`로 로봇-테이블 접촉 추적 (`contact_forces`)
  - `ContactSensorCfg`로 그리퍼-오브젝트 접촉 추적 (`contact_grasp`)

## 3. 학습 안정화를 위한 고정값

- 큐브 리셋 범위(작게 제한)
  - `x: (-0.01, 0.01)`
  - `y: (-0.02, 0.02)`
  - `z: (0.0, 0.0)`
- 종료/보상 높이 보정
  - `object_dropping.minimum_height = -0.35`
  - `lifting_object.minimal_height = -0.195`
  - `object_goal_tracking.minimal_height = -0.195`
  - `object_goal_tracking_fine_grained.minimal_height = -0.195`
- 테이블 접촉 즉시 종료
  - `terminations.table_contact = illegal_contact(...)`
  - 의미: 로봇이 테이블에 접촉하면 바로 episode 종료
  - 현재 운영값:
    - Stage2: 임계값 완화 (`threshold=1.2`)
    - Stage3: 임계값 엄격 (`threshold=0.3`)

## 4. 단계별 보상 체계

`E0509_TRAIN_STAGE` 값으로 단계 선택:

- `1` (Reach 단계)
  - `reaching_object` 보상을 크게
  - `lifting/object_goal_tracking/fine_grained` 보상은 0
  - 그리퍼는 열림 상태로 고정(접근 학습 집중)

- `2` (Grip 단계)
  - `reaching_object`는 보조로 유지
  - `object_goal_tracking`을 `grasp_proximity_and_closure` 함수로 교체
  - 물체 근처에서 그리퍼를 닫는 행동을 강하게 보상
  - `object_goal_tracking_fine_grained`를 `grasp_contact_reward`로 교체
  - 즉, 실제 그리퍼-물체 접촉(`contact_grasp`)이 있어야 추가 보상을 받음
  - `lifting` 보상은 0

- `3` (Lift 단계)
  - `lifting_object` 보상을 크게
  - `object_goal_tracking`은 기본 `object_goal_distance`로 복원
  - reach/grip 신호는 보조로 유지

잘못된 stage 값(1/2/3 외)은 에러로 중단됩니다.

## 5. 학습 코드 (Train)

작업 경로: `/home/jiwoo/IsaacLab`

권장값:
- Stage 1: `num_envs=2048`, `max_iterations=400`
- Stage 2: `num_envs=2048`, `max_iterations=500`
- Stage 3: `num_envs=3072`(부담되면 2048), `max_iterations=800`

1) Stage 1 시작:

```bash
E0509_TRAIN_STAGE=1 ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Lift-Cube-E0509-v1 \
  --num_envs 2048 \
  --max_iterations 400 \
  --headless
```

2) Stage 2 (Stage 1 체크포인트 이어학습):

```bash
E0509_TRAIN_STAGE=2 ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --num_envs 2048 \
  --max_iterations 500 \
  --task Isaac-Lift-Cube-E0509-v1 --resume \
  --load_run <stage1_run_folder> --checkpoint <stage1_model>.pt --headless
```

예시 (실제 값 사용):

```bash
E0509_TRAIN_STAGE=2 ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Lift-Cube-E0509-v1 \
  --num_envs 2048 \
  --max_iterations 500 \
  --resume \
  --load_run 2026-02-25_11-28-09 \
  --checkpoint model_399.pt \
  --headless
```

3) Stage 3 (Stage 2 체크포인트 이어학습):

```bash
E0509_TRAIN_STAGE=3 ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --num_envs 3072 \
  --max_iterations 800 \
  --task Isaac-Lift-Cube-E0509-v1 --resume \
  --load_run <stage2_run_folder> --checkpoint <stage2_model>.pt --headless
```

## 6. 실행 코드 (Play)

1) 특정 체크포인트 실행(권장: 절대경로 사용):

```bash
E0509_TRAIN_STAGE=1 ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Lift-Cube-E0509-Play-v1 \
  --num_envs 1 \
  --checkpoint /home/jiwoo/IsaacLab/logs/rsl_rl/e0509_lift/<run_folder>/<model_xxx.pt>
```

2) 예시:

```bash
E0509_TRAIN_STAGE=1 ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Isaac-Lift-Cube-E0509-Play-v1 \
  --num_envs 1 \
  --checkpoint /home/jiwoo/IsaacLab/logs/rsl_rl/e0509_lift/2026-02-25_11-28-09/model_399.pt
```

주의:
- `play.py`에서 `--checkpoint`를 주면 해당 파일 경로를 직접 찾습니다.
- 그래서 `--checkpoint model_399.pt`처럼 파일명만 주면 현재 작업 디렉터리에서 찾다가 실패할 수 있습니다.
- `E0509CubeLiftEnvCfg_PLAY`에서는 시각화 확인을 위해 `table_contact` 종료를 비활성화했습니다.

## 7. 빠른 체크 포인트

- `Episode_Reward/reaching_object`가 오르지 않으면 Stage 1 보상을 더 키웁니다.
- `Episode_Reward/object_goal_tracking_fine_grained`가 0에서 상승하면 실제 파지 접촉 보상이 작동 중입니다.
- `Episode_Termination/object_dropping`이 높아지면 높이 임계값/초기 z를 다시 확인합니다.
- `Episode_Termination/table_contact`가 높으면(자주 종료되면) 접촉 임계치 또는 접근 높이(ee offset)를 조정합니다.
- Stage 전환 시에는 반드시 `E0509_TRAIN_STAGE`를 바꿔서 실행합니다.
