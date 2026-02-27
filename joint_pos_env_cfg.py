import isaaclab.sim as sim_utils
import torch

from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass
from isaaclab.sim import (
    CollisionPropertiesCfg,
    MassPropertiesCfg,
    PreviewSurfaceCfg,
    RigidBodyPropertiesCfg,
)
from isaaclab.managers import EventTermCfg as EventTerm, SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm

from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg

from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab_assets.robots.e0509 import E0509_CFG



# stage1 => reach object
def grasp_proximity_and_closure(
    env,
    proximity_std: float,
    activate_dist: float,
    z_tolerance: float,
    far_close_penalty_scale: float,
    open_targets: tuple[float, float, float, float],
    close_targets: tuple[float, float, float, float],
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
):
    """Dense grasp reward: close gripper while staying near the object."""
    robot = env.scene[robot_cfg.name]
    obj = env.scene[object_cfg.name]
    ee_frame = env.scene[ee_frame_cfg.name]

    ee_pos = ee_frame.data.target_pos_w[..., 0, :]
    obj_pos = obj.data.root_pos_w
    # Use XY distance for approach shaping; Z is handled by a separate gate.
    xy_dist = torch.norm((obj_pos - ee_pos)[:, :2], dim=1)
    proximity = 1.0 - torch.tanh(xy_dist / proximity_std)
    proximity_gate = torch.sigmoid((activate_dist - xy_dist) / 0.06)
    z_err = torch.abs(obj_pos[:, 2] - ee_pos[:, 2])
    # Sharper Z-axis gate to prevent farming grasp reward at wrong height.
    z_gate = torch.sigmoid((z_tolerance - z_err) / 0.02)

    q = robot.data.joint_pos[:, robot_cfg.joint_ids]
    q_open = torch.tensor(open_targets, device=q.device).unsqueeze(0)
    q_close = torch.tensor(close_targets, device=q.device).unsqueeze(0)
    # Robust to mixed joint directions (some close by increasing angle, others by decreasing).
    denom = q_close - q_open
    close_progress = torch.where(
        denom > 0.0,
        (q - q_open) / (denom + 1e-6),
        (q_open - q) / ((q_open - q_close) + 1e-6),
    )
    close_progress = torch.clamp(close_progress, min=0.0, max=1.0).mean(dim=1)

    # Balance XY and Z so the policy must both approach and descend.
    approach_score = 0.5 * (proximity * proximity_gate) + 0.5 * z_gate
    # Penalize closing while still far in XY to break reward-farming loops.
    far_gate = torch.sigmoid((xy_dist - activate_dist) / 0.04)
    return approach_score * close_progress - far_close_penalty_scale * far_gate * close_progress


def gripper_close_progress_only(
    env,
    open_targets: tuple[float, float, float, float],
    close_targets: tuple[float, float, float, float],
    robot_cfg: SceneEntityCfg,
):
    """Dense reward for learning close motion itself (independent of distance gate)."""
    robot = env.scene[robot_cfg.name]
    q = robot.data.joint_pos[:, robot_cfg.joint_ids]
    q_open = torch.tensor(open_targets, device=q.device).unsqueeze(0)
    q_close = torch.tensor(close_targets, device=q.device).unsqueeze(0)
    denom = q_close - q_open
    close_progress = torch.where(
        denom > 0.0,
        (q - q_open) / (denom + 1e-6),
        (q_open - q) / ((q_open - q_close) + 1e-6),
    )
    return torch.clamp(close_progress, min=0.0, max=1.0).mean(dim=1)


def close_far_penalty(
    env,
    activate_dist: float,
    open_targets: tuple[float, float, float, float],
    close_targets: tuple[float, float, float, float],
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
):
    """Penalize closing the gripper while still far from the object."""
    robot = env.scene[robot_cfg.name]
    obj = env.scene[object_cfg.name]
    ee_frame = env.scene[ee_frame_cfg.name]

    ee_pos = ee_frame.data.target_pos_w[..., 0, :]
    obj_pos = obj.data.root_pos_w
    dist = torch.norm(obj_pos - ee_pos, dim=1)
    far_gate = (dist >= activate_dist).float()

    q = robot.data.joint_pos[:, robot_cfg.joint_ids]
    q_open = torch.tensor(open_targets, device=q.device).unsqueeze(0)
    q_close = torch.tensor(close_targets, device=q.device).unsqueeze(0)
    denom = q_close - q_open
    close_progress = torch.where(
        denom > 0.0,
        (q - q_open) / (denom + 1e-6),
        (q_open - q) / ((q_open - q_close) + 1e-6),
    )
    close_progress = torch.clamp(close_progress, min=0.0, max=1.0).mean(dim=1)

    return far_gate * close_progress


def grasp_proximity_reward(
    env,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
):
    """Dense proximity reward for grasp curriculum (approach before closure)."""
    obj = env.scene[object_cfg.name]
    ee_frame = env.scene[ee_frame_cfg.name]
    ee_pos = ee_frame.data.target_pos_w[..., 0, :]
    obj_pos = obj.data.root_pos_w
    dist = torch.norm(obj_pos - ee_pos, dim=1)
    return 1.0 - torch.tanh(dist / std)


def ee_object_z_alignment_reward(
    env,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
):
    """Dense reward for aligning end-effector height with object height."""
    obj = env.scene[object_cfg.name]
    ee_frame = env.scene[ee_frame_cfg.name]
    ee_pos = ee_frame.data.target_pos_w[..., 0, :]
    obj_pos = obj.data.root_pos_w
    z_err = torch.abs(obj_pos[:, 2] - ee_pos[:, 2])
    return 1.0 - torch.tanh(z_err / std)


def dense_lift_progress(
    env,
    start_height: float,
    max_lift: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
):
    """Dense lift reward from initial table height toward upward displacement."""
    obj = env.scene[object_cfg.name]
    z = obj.data.root_pos_w[:, 2]
    return torch.clamp((z - start_height) / (max_lift + 1e-6), min=0.0, max=1.0)


def object_reached_height_goal(
    env,
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
):
    """Success when the object is lifted above the target Z threshold."""
    return mdp.object_is_lifted(env=env, minimal_height=minimal_height, object_cfg=object_cfg) > 0.5


def ee_reached_object_goal(
    env,
    xy_threshold: float,
    z_threshold: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
):
    """Success when end-effector is close enough to object in XY and Z."""
    obj = env.scene[object_cfg.name]
    ee_frame = env.scene[ee_frame_cfg.name]
    ee_pos = ee_frame.data.target_pos_w[..., 0, :]
    obj_pos = obj.data.root_pos_w
    xy_dist = torch.norm((obj_pos - ee_pos)[:, :2], dim=1)
    z_err = torch.abs(obj_pos[:, 2] - ee_pos[:, 2])
    return (xy_dist < xy_threshold) & (z_err < z_threshold)


def object_out_of_table_xy(
    env,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
):
    """Terminate when the object center leaves the tabletop XY bounds."""
    obj = env.scene[object_cfg.name]
    x = obj.data.root_pos_w[:, 0]
    y = obj.data.root_pos_w[:, 1]
    return (x < x_min) | (x > x_max) | (y < y_min) | (y > y_max)


@configclass
class E0509CubeLiftEnvCfg(LiftEnvCfg):
    """Stable Lift Task with E0509 + 5cm cube"""

    def __post_init__(self):
        super().__post_init__()
        # Training curriculum stage:
        # 1 = reach only, 2 = reach + grasp, 3 = full lift task
        stage = 1
        # Keep task timing close to default lift setup.
        self.decimation = 2
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physics_material.static_friction = 0.8
        self.sim.physics_material.dynamic_friction = 0.7
        self.sim.physics_material.restitution = 0.0
        self.sim.physics_material.friction_combine_mode = "min"
        self.episode_length_s = 5.0

        # ==============================
        # Robot
        # ==============================
        self.scene.robot = E0509_CFG.replace(
            prim_path="{ENV_REGEX_NS}/e0509"
        )
        self.commands.object_pose.body_name = "link_6"
        self.commands.object_pose.debug_vis = False

        # Fixed base 확실히 적용
        if self.scene.robot.spawn.articulation_props is None:
            self.scene.robot.spawn.articulation_props = sim_utils.ArticulationRootPropertiesCfg()

        self.scene.robot.spawn.articulation_props.fix_root_link = True
        self.scene.robot.spawn.articulation_props.enabled_self_collisions = False
        self.scene.robot.actuators["arm"].stiffness = 800.0
        self.scene.robot.actuators["arm"].damping = 60.0
        self.scene.robot.actuators["arm"].velocity_limit = 0.8
        self.scene.robot.actuators["gripper"].effort_limit = 300.0
        self.scene.robot.actuators["gripper"].velocity_limit = 1.0
        self.scene.robot.actuators["gripper"].stiffness = 2800.0
        self.scene.robot.actuators["gripper"].damping = 220.0

        # 로봇 초기 위치
        self.scene.robot.init_state.pos = (-0.45, 0.0, -0.225)
        self.scene.robot.init_state.rot = (1.0, 0.0, 0.0, 0.0)
        
        # 관절 초기값 (안 눕는 안전 자세)
        self.scene.robot.init_state.joint_pos = {
            "joint_1": 0.0,
            "joint_2": 0.65,
            "joint_3": 0.75,
            "joint_4": 0.0,
            "joint_5": 1.3,
            "joint_6": 1.5708,
            "rh_l1": 0.14835,
            "rh_r1": 0.14835,
            "rh_l2": 0.00873,
            "rh_r2": 0.00873,
        }

        # Arm Action
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["joint_[1-6]"],
            scale=0.08,
            use_default_offset=True,
        )

        # Gripper Action
        self.actions.gripper_action = mdp.AbsBinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["rh_l1", "rh_r1", "rh_l2", "rh_r2"],
            open_command_expr={
                "rh_l1": 0.14835,
                "rh_r1": 0.14835,
                "rh_l2": 0.00873,
                "rh_r2": 0.00873,
            },
            close_command_expr={
                "rh_l1": 1.25,
                "rh_r1": 1.25,
                "rh_l2": -1.00,
                "rh_r2": -1.00,
            },
            threshold=0.0,
            positive_threshold=False,
        )
        gripper_open_pose = {
            "rh_l1": 0.14835,
            "rh_r1": 0.14835,
            "rh_l2": 0.00873,
            "rh_r2": 0.00873,
        }
        # -----------------------------
        # Table (static)
        # -----------------------------
        self.scene.table = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/table",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.0, 0.0, -0.65], rot=[1.0, 0.0, 0.0, 0.0]),
            spawn=sim_utils.CuboidCfg(
                size=(1.2, 0.8, 0.8),
                collision_props=CollisionPropertiesCfg(
                    contact_offset=0.005,
                    rest_offset=0.0,
                ),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=1.0,
                    dynamic_friction=1.0,
                    restitution=0.0,
                    friction_combine_mode="min",
                ),
                visual_material=PreviewSurfaceCfg(),
                mass_props=MassPropertiesCfg(mass=100.0),
                rigid_props=RigidBodyPropertiesCfg(
                    kinematic_enabled=True,
                    disable_gravity=True,
                    max_depenetration_velocity=1.0,
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=4,
                ),
            ),
        )

        # ==============================
        # Cube (5cm)
        # ==============================
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",   
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[0.2, 0.0, -0.225],
                rot=[1, 0, 0, 0],
            ),
            spawn=sim_utils.CuboidCfg(
                size=(0.05, 0.05, 0.05),
                collision_props=CollisionPropertiesCfg(
                    contact_offset=0.005,
                    rest_offset=0.0,
                ),
                visual_material=PreviewSurfaceCfg(diffuse_color=(0.2, 0.6, 0.95)),
                mass_props=MassPropertiesCfg(mass=0.02),
                rigid_props=RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    enable_gyroscopic_forces=True,
                    linear_damping=0.2,
                    angular_damping=0.2,
                    max_depenetration_velocity=1.0,
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=4,
                    sleep_threshold=0.005,
                    stabilization_threshold=0.0025,
                ),
            ),
        )
        
        # Limit joint_1 to [-90 deg, +90 deg] = [-pi/2, +pi/2].
        self.events.joint1_limits = EventTerm(
            func=mdp.randomize_joint_parameters,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=["joint_1"]),
                "operation": "abs",
                "distribution": "uniform",
                "lower_limit_distribution_params": (-1.5708, -1.5708),
                "upper_limit_distribution_params": (1.5708, 1.5708),
            },
        )
        self.events.reset_object_position.params["pose_range"] = {
            "x": (-0.01, 0.01),
            "y": (-0.02, 0.02),
            "z": (0.0, 0.0),
        }
        self.rewards.reaching_object.weight = 12.0
        self.rewards.reaching_object.params["std"] = 0.20
        self.rewards.lifting_object.func = dense_lift_progress
        self.rewards.lifting_object.params = {
            "start_height": -0.225,
            "max_lift": 0.03,
            "object_cfg": SceneEntityCfg("object"),
        }
        self.rewards.lifting_object.weight = 30.0
        self.rewards.object_goal_tracking.func = grasp_proximity_and_closure
        self.rewards.object_goal_tracking.params = {
            "proximity_std": 0.20,
            "activate_dist": 0.20,
            "z_tolerance": 0.10,
            "far_close_penalty_scale": 0.5,
            "open_targets": (0.14835, 0.14835, 0.00873, 0.00873),
            "close_targets": (1.25, 1.25, -1.00, -1.00),
            "robot_cfg": SceneEntityCfg("robot", joint_names=["rh_l1", "rh_r1", "rh_l2", "rh_r2"]),
            "object_cfg": SceneEntityCfg("object"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
        }
        self.rewards.object_goal_tracking.weight = 6.0
        # Strongly encourage approaching at the correct object height (z-axis alignment).
        self.rewards.object_goal_tracking_fine_grained.func = ee_object_z_alignment_reward
        self.rewards.object_goal_tracking_fine_grained.params = {
            "std": 0.08,
            "object_cfg": SceneEntityCfg("object"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
        }
        self.rewards.object_goal_tracking_fine_grained.weight = 14.0
        if stage == 1:
            # Reach-only stage: remove grasp/lift incentives to stabilize approach first.
            self.rewards.lifting_object.weight = 0.0
            self.rewards.object_goal_tracking.weight = 0.0
            # Keep a pure Z-alignment dense reward so stage-1 learns descending behavior.
            self.rewards.object_goal_tracking_fine_grained.weight = 6.0
            # Keep gripper open during approach curriculum.
            self.actions.gripper_action.open_command_expr = gripper_open_pose
            self.actions.gripper_action.close_command_expr = gripper_open_pose
            # Remove goal-command observation noise for pure reach curriculum.
            self.observations.policy.target_object_position = None
        self.rewards.action_rate.weight = -5e-5
        self.rewards.joint_vel.weight = -5e-5
        self.curriculum.action_rate = None
        self.curriculum.joint_vel = None
        self.terminations.object_dropping.params["minimum_height"] = -0.35
        self.terminations.object_out_of_table = DoneTerm(
            func=object_out_of_table_xy,
            params={
                # Table top size is (1.2, 0.8), keep a safety margin from edges.
                "x_min": -0.50,
                "x_max": 0.50,
                "y_min": -0.30,
                "y_max": 0.30,
                "object_cfg": SceneEntityCfg("object"),
            },
        )
        self.terminations.object_reached_goal = DoneTerm(
            func=object_reached_height_goal,
            params={
                "minimal_height": -0.195,
                "object_cfg": SceneEntityCfg("object"),
            },
        )
        if stage == 1:
            # Stage-1 success: reaching object vicinity (not lifting).
            self.terminations.object_reached_goal = DoneTerm(
                func=ee_reached_object_goal,
                params={
                    "xy_threshold": 0.14,
                    "z_threshold": 0.06,
                    "object_cfg": SceneEntityCfg("object"),
                    "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                },
            )

        # ==============================
        # Frame Transformer
        # ==============================
        
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"

        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/e0509/e0509/base_link",
            debug_vis=True,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/e0509/e0509/link_6",
                    name="end_effector",
                    offset=OffsetCfg(pos=[0.0, 0.0, 0.0]),
                ),
            ],
        )


@configclass
class E0509CubeLiftEnvCfg_PLAY(E0509CubeLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # 디버깅용 1개 환경
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.viewer.eye = (1.45, 0.0, -0.12)
        self.viewer.lookat = (-0.25, 0.0, -0.12)
        self.observations.policy.enable_corruption = False
