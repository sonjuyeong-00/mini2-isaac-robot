# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import SceneEntityCfg
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
        self.scene.robot.actuators["arm"].velocity_limit = 0.5
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
            "rh_l1": 0.14835,
            "rh_r1": 0.14835,
            "rh_l2": 0.00873,
            "rh_r2": 0.00873,
        }

        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["joint_[1-6]"], scale=0.2, use_default_offset=True
        )
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
            threshold=0.2,
            positive_threshold=False,
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

        # Optional table-contact termination sensor
        self.scene.contact_forces = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/e0509/e0509/.*",
            update_period=self.sim.dt,
            history_length=3,
            debug_vis=False,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/table"],
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
                "rh_l1": 0.14835,
                "rh_r1": 0.14835,
                "rh_l2": 0.00873,
                "rh_r2": 0.00873,
            }
            self.actions.gripper_action.open_command_expr = gripper_open
            self.actions.gripper_action.close_command_expr = gripper_open
            self.actions.gripper_action.threshold = 1.0
            self.scene.robot.actuators["gripper"].velocity_limit = 1e-3

            self.actions.arm_action.scale = 0.15
            self.rewards.reaching_object.weight = 20.0
            self.rewards.reaching_object.params["std"] = 0.07
            self.rewards.lifting_object.weight = 0.0
            self.rewards.object_goal_tracking.weight = 0.0
            self.rewards.object_goal_tracking_fine_grained.weight = 0.0
            self.rewards.action_rate.weight = -1e-5
            self.rewards.joint_vel.weight = -1e-5
            self.curriculum.action_rate = None
            self.curriculum.joint_vel = None

        elif stage == 2:
            # Grasp/lift start: simple reward mix without custom terms.
            self.actions.arm_action.scale = 0.12
            self.rewards.reaching_object.weight = 8.0
            self.rewards.reaching_object.params["std"] = 0.12
            self.rewards.lifting_object.weight = 18.0
            self.rewards.object_goal_tracking.weight = 0.0
            self.rewards.object_goal_tracking_fine_grained.weight = 0.0
            self.rewards.action_rate.weight = -5e-5
            self.rewards.joint_vel.weight = -5e-5
            self.curriculum.action_rate = None
            self.curriculum.joint_vel = None

        else:
            # Full lift.
            self.actions.arm_action.scale = 0.12
            self.rewards.reaching_object.weight = 4.0
            self.rewards.reaching_object.params["std"] = 0.10
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
        self.viewer.eye = (-0.46, 0.0, -0.16)
        self.viewer.lookat = (-0.32, 0.0, -0.16)
        self.observations.policy.enable_corruption = False
