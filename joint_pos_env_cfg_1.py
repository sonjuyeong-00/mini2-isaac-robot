# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

import os
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim import CollisionPropertiesCfg, MassPropertiesCfg, PreviewSurfaceCfg, RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab_assets.robots.e0509 import E0509_CFG

from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg


def finger_midpoint_object_frame_distance(
    env,
    std: float,
    finger_frame_cfg: SceneEntityCfg = SceneEntityCfg("finger_frame"),
    object_frame_cfg: SceneEntityCfg = SceneEntityCfg("object_frame"),
):
    """Dense reach reward using averaged gripper/finger frame center to object frame."""
    finger_frame = env.scene[finger_frame_cfg.name]
    object_frame = env.scene[object_frame_cfg.name]

    # If multiple finger/gripper links match, average them to a single contact-center proxy.
    finger_mid = finger_frame.data.target_pos_w.mean(dim=1)
    obj_pos = object_frame.data.target_pos_w[..., 0, :]
    dist = torch.norm(obj_pos - finger_mid, dim=1)
    return 1.0 - torch.tanh(dist / std)


def finger_object_xy_z_alignment(
    env,
    xy_std: float,
    z_std: float,
    finger_frame_cfg: SceneEntityCfg = SceneEntityCfg("finger_frame"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
):
    """Reward when object center is inside finger midpoint in XY and matched in Z."""
    finger_frame = env.scene[finger_frame_cfg.name]
    obj = env.scene[object_cfg.name]

    finger_mid = finger_frame.data.target_pos_w.mean(dim=1)
    obj_pos = obj.data.root_pos_w

    xy_dist = torch.norm((obj_pos - finger_mid)[:, :2], dim=1)
    z_err = torch.abs(obj_pos[:, 2] - finger_mid[:, 2])

    xy_score = 1.0 - torch.tanh(xy_dist / xy_std)
    z_score = 1.0 - torch.tanh(z_err / z_std)
    return xy_score * z_score


def gripper_close_progress_only(
    env,
    open_targets: tuple[float, float, float, float],
    close_targets: tuple[float, float, float, float],
    robot_cfg: SceneEntityCfg,
):
    """Dense reward for closing motion itself (independent from lift)."""
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


def gated_gripper_close_reward(
    env,
    open_targets: tuple[float, float, float, float],
    close_targets: tuple[float, float, float, float],
    robot_cfg: SceneEntityCfg,
    finger_frame_cfg: SceneEntityCfg = SceneEntityCfg("finger_frame"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    xy_std: float = 0.12,
    z_std: float = 0.05,
):
    """Close reward gated by proximity/alignment so far-away closing is not rewarded."""
    close_progress = gripper_close_progress_only(
        env=env,
        open_targets=open_targets,
        close_targets=close_targets,
        robot_cfg=robot_cfg,
    )
    align_score = finger_object_xy_z_alignment(
        env=env,
        xy_std=xy_std,
        z_std=z_std,
        finger_frame_cfg=finger_frame_cfg,
        object_cfg=object_cfg,
    )
    return close_progress * align_score


def grasp_success_condition(
    env,
    close_threshold: float,
    align_threshold: float,
    contact_force_threshold: float,
    hold_steps: int,
    open_targets: tuple[float, float, float, float],
    close_targets: tuple[float, float, float, float],
    robot_cfg: SceneEntityCfg,
    finger_frame_cfg: SceneEntityCfg = SceneEntityCfg("finger_frame"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    contact_cfg: SceneEntityCfg = SceneEntityCfg("gripper_object_contact"),
    left_body_pattern: str = "rh_p12_rn_l.*",
    right_body_pattern: str = "rh_p12_rn_r.*",
    align_xy_std: float = 0.12,
    align_z_std: float = 0.05,
):
    """Binary grasp success: close + align + both-side contact sustained for hold_steps."""
    close_progress = gripper_close_progress_only(
        env=env,
        open_targets=open_targets,
        close_targets=close_targets,
        robot_cfg=robot_cfg,
    )
    align_score = finger_object_xy_z_alignment(
        env=env,
        xy_std=align_xy_std,
        z_std=align_z_std,
        finger_frame_cfg=finger_frame_cfg,
        object_cfg=object_cfg,
    )

    contact_sensor = env.scene[contact_cfg.name]
    left_ids, _ = contact_sensor.find_bodies(left_body_pattern)
    right_ids, _ = contact_sensor.find_bodies(right_body_pattern)
    if len(left_ids) == 0 or len(right_ids) == 0:
        return torch.zeros_like(close_progress, dtype=torch.bool)

    force_hist = torch.norm(contact_sensor.data.net_forces_w_history, dim=-1)  # (N, H, B)
    hold_steps = max(1, min(hold_steps, force_hist.shape[1]))
    left_contact_hold = (force_hist[:, :hold_steps, left_ids] > contact_force_threshold).any(dim=2).all(dim=1)
    right_contact_hold = (force_hist[:, :hold_steps, right_ids] > contact_force_threshold).any(dim=2).all(dim=1)

    return (close_progress > close_threshold) & (align_score > align_threshold) & left_contact_hold & right_contact_hold


def grasp_success_bonus(
    env,
    close_threshold: float,
    align_threshold: float,
    contact_force_threshold: float,
    hold_steps: int,
    open_targets: tuple[float, float, float, float],
    close_targets: tuple[float, float, float, float],
    robot_cfg: SceneEntityCfg,
    finger_frame_cfg: SceneEntityCfg = SceneEntityCfg("finger_frame"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    contact_cfg: SceneEntityCfg = SceneEntityCfg("gripper_object_contact"),
    left_body_pattern: str = "rh_p12_rn_l.*",
    right_body_pattern: str = "rh_p12_rn_r.*",
    align_xy_std: float = 0.12,
    align_z_std: float = 0.05,
):
    """Sparse bonus when grasp_success_condition is satisfied."""
    return grasp_success_condition(
        env=env,
        close_threshold=close_threshold,
        align_threshold=align_threshold,
        contact_force_threshold=contact_force_threshold,
        hold_steps=hold_steps,
        open_targets=open_targets,
        close_targets=close_targets,
        robot_cfg=robot_cfg,
        finger_frame_cfg=finger_frame_cfg,
        object_cfg=object_cfg,
        contact_cfg=contact_cfg,
        left_body_pattern=left_body_pattern,
        right_body_pattern=right_body_pattern,
        align_xy_std=align_xy_std,
        align_z_std=align_z_std,
    ).float()


@configclass
class E0509CubeLiftEnvCfg(LiftEnvCfg):
    """Simplified E0509 lift curriculum.

    Stage 1: reach object only (gripper fixed open)
    Stage 2: reach + lift (start grasp behavior)
    Stage 3: full lift + goal tracking
    """

    def __post_init__(self):
        super().__post_init__()

        stage = int(os.getenv("E0509_TRAIN_STAGE", "1"))
        if stage not in (1, 2, 3):
            raise ValueError(f"E0509_TRAIN_STAGE must be 1, 2, or 3. Got: {stage}")

        # Keep simulation close to default Lift settings.
        self.decimation = 2
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation
        self.episode_length_s = 5.0

        # Robot
        self.scene.robot = E0509_CFG.replace(prim_path="{ENV_REGEX_NS}/e0509")
        if hasattr(self.scene.robot, "spawn") and self.scene.robot.spawn is not None:
            self.scene.robot.spawn.activate_contact_sensors = True
        if self.scene.robot.spawn.articulation_props is None:
            self.scene.robot.spawn.articulation_props = sim_utils.ArticulationRootPropertiesCfg()

        self.scene.robot.spawn.articulation_props.fix_root_link = True
        self.scene.robot.spawn.articulation_props.enabled_self_collisions = False

        self.scene.robot.actuators["arm"].stiffness = 800.0
        self.scene.robot.actuators["arm"].damping = 60.0
        # Increase arm joint velocity ceiling to avoid overly sluggish motion.
        self.scene.robot.actuators["arm"].velocity_limit = 0.9
        self.scene.robot.actuators["gripper"].effort_limit = 300.0
        self.scene.robot.actuators["gripper"].velocity_limit = 0.06
        self.scene.robot.actuators["gripper"].stiffness = 2000.0
        self.scene.robot.actuators["gripper"].damping = 220.0

        self.scene.robot.init_state.pos = (-0.45, 0.0, -0.225)
        self.scene.robot.init_state.rot = (1.0, 0.0, 0.0, 0.0)
        self.scene.robot.init_state.joint_pos = {
            "joint_1": 0.0,
            "joint_2": 0.65,
            "joint_3": 0.75,
            "joint_4": 0.0,
            "joint_5": 1.3,
            "joint_6": 1.5708,
            "rh_l1": 0.02,
            "rh_r1": 0.02,
            "rh_l2": 0.02,
            "rh_r2": 0.02,
        }

        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["joint_[1-6]"], scale=0.22, use_default_offset=True
        )
        self.actions.gripper_action = mdp.AbsBinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["rh_l1", "rh_r1", "rh_l2", "rh_r2"],
            open_command_expr={
                "rh_l1": 0.02,
                "rh_r1": 0.02,
                "rh_l2": 0.02,
                "rh_r2": 0.02,
            },
            close_command_expr={
                "rh_l1": 0.95993,  # 55 deg
                "rh_r1": 0.95993,  # 55 deg
                "rh_l2": 0.78540,  # 45 deg
                "rh_r2": 0.78540,  # 45 deg
            },
            threshold=0.0,
            positive_threshold=True,
        )

        self.commands.object_pose.body_name = "link_6"

        # Object
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.2, 0.0, -0.225], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.8, 0.8, 0.8),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=4,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=2.0,
                    disable_gravity=False,
                ),
            ),
        )

        # Table (static)
        self.scene.table = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/table",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.0, 0.0, -0.65], rot=[1.0, 0.0, 0.0, 0.0]),
            spawn=sim_utils.CuboidCfg(
                size=(1.2, 0.8, 0.8),
                collision_props=CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
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

        # End-effector frame for distance reward (single, fixed offset)
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/e0509/e0509/base_link",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/e0509/e0509/link_6",
                    name="end_effector",
                    offset=OffsetCfg(pos=[0.0, 0.0, 0.0]),
                ),
            ],
        )
        self.scene.finger_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/e0509/e0509/base_link",
            debug_vis=False,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/e0509/e0509/rh_.*",
                    name="finger_set",
                    offset=OffsetCfg(pos=[0.0, 0.0, 0.0]),
                ),
            ],
        )
        self.scene.object_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            debug_vis=False,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Object",
                    name="object_center",
                    # Slightly below center to bias the gripper midpoint to descend more.
                    offset=OffsetCfg(pos=[0.0, 0.0, -0.01]),
                ),
            ],
        )
        self.rewards.reaching_object.func = finger_midpoint_object_frame_distance
        self.rewards.reaching_object.params = {
            "std": 0.14,
            "finger_frame_cfg": SceneEntityCfg("finger_frame"),
            "object_frame_cfg": SceneEntityCfg("object_frame"),
        }

        # Optional table-contact termination sensor
        self.scene.contact_forces = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/e0509/e0509/.*",
            update_period=self.sim.dt,
            history_length=3,
            debug_vis=False,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/table"],
        )
        self.scene.gripper_object_contact = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/e0509/e0509/.*",
            update_period=self.sim.dt,
            history_length=5,
            debug_vis=False,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        )

        self.events.reset_object_position.params["pose_range"] = {
            "x": (-0.02, 0.02),
            "y": (-0.02, 0.02),
            "z": (0.0, 0.0),
        }

        # Height thresholds adjusted for this table/object setup.
        self.terminations.object_dropping.params["minimum_height"] = -0.35
        self.rewards.lifting_object.params["minimal_height"] = -0.195
        self.rewards.object_goal_tracking.params["minimal_height"] = -0.195
        self.rewards.object_goal_tracking_fine_grained.params["minimal_height"] = -0.195

        # Disallow table contact by robot links and gripper in all stages.
        self.terminations.table_contact = DoneTerm(
            func=mdp.illegal_contact,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["link_[1-6]", "rh_.*"]),
                "threshold": 0.3,
            },
        )

        # Stage-specific simplification
        if stage == 1:
            # Reach only: keep gripper open/fixed.
            gripper_open = {
                "rh_l1": 0.02,
                "rh_r1": 0.02,
                "rh_l2": 0.02,
                "rh_r2": 0.02,
            }
            self.actions.gripper_action.open_command_expr = gripper_open
            self.actions.gripper_action.close_command_expr = gripper_open
            self.actions.gripper_action.threshold = 1.0
            self.scene.robot.actuators["gripper"].velocity_limit = 1e-3

            self.actions.arm_action.scale = 0.20
            self.rewards.reaching_object.weight = 24.0
            self.rewards.reaching_object.params["std"] = 0.10
            self.rewards.lifting_object.weight = 0.0
            self.rewards.object_goal_tracking.weight = 0.0
            self.rewards.object_goal_tracking_fine_grained.func = finger_object_xy_z_alignment
            self.rewards.object_goal_tracking_fine_grained.params = {
                "xy_std": 0.05,
                "z_std": 0.015,
                "finger_frame_cfg": SceneEntityCfg("finger_frame"),
                "object_cfg": SceneEntityCfg("object"),
            }
            self.rewards.object_goal_tracking_fine_grained.weight = 16.0
            self.rewards.action_rate.weight = -1e-5
            self.rewards.joint_vel.weight = -1e-5
            self.curriculum.action_rate = None
            self.curriculum.joint_vel = None

        elif stage == 2:
            # Grasp + hold stage: focus on insertion/alignment/closure, not lift.
            # Allow larger exploration in Stage-2 grasping motions.
            self.actions.arm_action.scale = 0.22
            # Stage-1 resumed policies often output gripper actions near/below 0.
            # For AbsBinaryJointPositionAction:
            # - positive_threshold=True  -> action > threshold => open, else close
            # This makes near-zero outputs map to CLOSE, so gripper motion appears.
            self.actions.gripper_action.positive_threshold = True
            # Stage-2 bootstrap: keep gripper mostly in close mode.
            # (action range is usually [-1, 1], so threshold=1.0 means almost-always close)
            self.actions.gripper_action.threshold = 1.0
            self.scene.robot.actuators["gripper"].velocity_limit = 0.12
            self.rewards.reaching_object.weight = 5.0
            self.rewards.reaching_object.params["std"] = 0.16
            self.rewards.lifting_object.weight = 0.0
            self.rewards.object_goal_tracking.weight = 3.0
            self.rewards.object_goal_tracking.func = gated_gripper_close_reward
            self.rewards.object_goal_tracking.params = {
                "open_targets": (0.02, 0.02, 0.02, 0.02),
                "close_targets": (0.95993, 0.95993, 0.78540, 0.78540),
                "robot_cfg": SceneEntityCfg("robot", joint_names=["rh_l1", "rh_r1", "rh_l2", "rh_r2"]),
                "finger_frame_cfg": SceneEntityCfg("finger_frame"),
                "object_cfg": SceneEntityCfg("object"),
                "xy_std": 0.12,
                "z_std": 0.05,
            }
            self.rewards.object_goal_tracking_fine_grained.func = finger_object_xy_z_alignment
            self.rewards.object_goal_tracking_fine_grained.params = {
                "xy_std": 0.12,
                "z_std": 0.05,
                "finger_frame_cfg": SceneEntityCfg("finger_frame"),
                "object_cfg": SceneEntityCfg("object"),
            }
            self.rewards.object_goal_tracking_fine_grained.weight = 10.0
            self.rewards.grasp_success = RewTerm(
                func=grasp_success_bonus,
                params={
                    "close_threshold": 0.55,
                    "align_threshold": 0.15,
                    "contact_force_threshold": 0.1,
                    "hold_steps": 1,
                    "open_targets": (0.02, 0.02, 0.02, 0.02),
                    "close_targets": (0.95993, 0.95993, 0.78540, 0.78540),
                    "robot_cfg": SceneEntityCfg("robot", joint_names=["rh_l1", "rh_r1", "rh_l2", "rh_r2"]),
                    "finger_frame_cfg": SceneEntityCfg("finger_frame"),
                    "object_cfg": SceneEntityCfg("object"),
                    "contact_cfg": SceneEntityCfg("gripper_object_contact"),
                    "left_body_pattern": "rh_p12_rn_l.*",
                    "right_body_pattern": "rh_p12_rn_r.*",
                    "align_xy_std": 0.12,
                    "align_z_std": 0.05,
                },
                weight=12.0,
            )
            # Stage-2 needs frequent near-table finger motion for grasp attempts.
            # Relax table-contact termination by excluding gripper links and using
            # a higher force threshold for arm links.
            self.terminations.table_contact.params["sensor_cfg"] = SceneEntityCfg(
                "contact_forces", body_names=["link_[1-6]"]
            )
            self.terminations.table_contact.params["threshold"] = 1.0
            self.terminations.grasp_success = DoneTerm(
                func=grasp_success_condition,
                params={
                    "close_threshold": 0.55,
                    "align_threshold": 0.15,
                    "contact_force_threshold": 0.1,
                    "hold_steps": 1,
                    "open_targets": (0.02, 0.02, 0.02, 0.02),
                    "close_targets": (0.95993, 0.95993, 0.78540, 0.78540),
                    "robot_cfg": SceneEntityCfg("robot", joint_names=["rh_l1", "rh_r1", "rh_l2", "rh_r2"]),
                    "finger_frame_cfg": SceneEntityCfg("finger_frame"),
                    "object_cfg": SceneEntityCfg("object"),
                    "contact_cfg": SceneEntityCfg("gripper_object_contact"),
                    "left_body_pattern": "rh_p12_rn_l.*",
                    "right_body_pattern": "rh_p12_rn_r.*",
                    "align_xy_std": 0.12,
                    "align_z_std": 0.05,
                },
            )
            # Relax motion penalties so arm can move more freely in Stage-2.
            self.rewards.action_rate.weight = -1e-5
            self.rewards.joint_vel.weight = -1e-5
            self.curriculum.action_rate = None
            self.curriculum.joint_vel = None

        else:
            # Full lift.
            self.actions.arm_action.scale = 0.14
            self.rewards.reaching_object.weight = 4.0
            self.rewards.reaching_object.params["std"] = 0.14
            self.rewards.lifting_object.weight = 20.0
            self.rewards.object_goal_tracking.weight = 10.0
            self.rewards.object_goal_tracking_fine_grained.weight = 3.0
            self.rewards.action_rate.weight = -1e-4
            self.rewards.joint_vel.weight = -1e-4


@configclass
class E0509CubeLiftEnvCfg_PLAY(E0509CubeLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.viewer.eye = (1.45, 0.0, -0.12)
        self.viewer.lookat = (-0.25, 0.0, -0.12)
        self.observations.policy.enable_corruption = False
