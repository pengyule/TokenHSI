"""
This script demonstrates how to load a custom .obj asset and stack multiple
instances of it in a stable tower using Isaac Gym. It incorporates best
practices for physics tuning to ensure a stable simulation, based on the
provided instructions.
"""

from isaacgym import gymapi, gymutil
import numpy as np
import os
from env.tasks.base_task import BaseTask


class ObjectLib_JerryTesting():
    def __init__(self, mode, dataset_root, dataset_categories, category_specified, num_envs, device):
        self.device = device
        self.mode = mode

        # load basic info
        dataset_categories_count = []
        obj_urdfs = []
        obj_cateIds = []
        obj_bbox_centers = []
        obj_bbox_lengths = []
        obj_facings = []
        obj_up_facings = []
        obj_on_platform_trans = []
        self.platform_thickness = 0.001 # this is the thickness of the platform, which is used to ensure the object is on the platform
        self.platform_width = 1.0
        for cat in dataset_categories:
            
            if cat == category_specified: # we only use the specified object category

                obj_list = os.listdir(os.path.join(dataset_root, mode, cat))
                dataset_categories_count.append(len(obj_list))
                for obj_name in obj_list:
                    curr_dir = os.path.join(dataset_root, mode, cat, obj_name)
                    obj_urdfs.append(os.path.join(curr_dir, "asset.urdf"))
                    obj_cateIds.append(ObjectCategoryId[cat].value)

                    with open(os.path.join(os.getcwd(), curr_dir, "config.json"), "r") as f:
                        object_cfg = json.load(f)
                        assert not np.sum(np.abs(object_cfg["center"])) > 0.0 
                        obj_platform_centers.append(object_cfg["platform_center"])
                        obj_bbox_lengths.append(object_cfg["bbox"])
                        obj_facings.append(object_cfg["facing"])
                        obj_up_facings.append(object_cfg["up_facing"])
                        obj_bbox_center = obj_platform_centers[-1]
                        obj_bbox_center[2] += (obj_bbox_lengths[-1][2] + self.platform_thickness) / 2  # ensure the object is on the platform
                        obj_on_platform_trans.append(-1 * (obj_bbox_centers[-1][2] - obj_bbox_lengths[-1][2] / 2))  

        assert len(dataset_categories_count) != 0, "You must specify one type of object!!!"
        assert len(dataset_categories_count) == 1, "You can only specify one type of object, no more!!!"

        # randomly sample a fixed object for each simulation env, due to the limitation of IsaacGym
        num_objs_loaded = len(obj_urdfs)
        weights = torch.ones(num_objs_loaded, device=self.device) * (1.0 / num_objs_loaded)
        self._every_env_object_ids = torch.multinomial(weights, num_samples=num_envs, replacement=True).squeeze(-1)
        if num_envs == 1:
            self._every_env_object_ids = self._every_env_object_ids.unsqueeze(0)
        self._every_env_object_cateIds = to_torch(obj_cateIds, dtype=torch.long, device=self.device)[self._every_env_object_ids]
        self._every_env_object_platform_centers = to_torch(obj_platform_centers, dtype=torch.float, device=self.device)[self._every_env_object_ids]
        self._every_env_object_bbox_lengths = to_torch(obj_bbox_lengths, dtype=torch.float, device=self.device)[self._every_env_object_ids]
        self._every_env_object_facings = to_torch(obj_facings, dtype=torch.float, device=self.device)[self._every_env_object_ids]
        self._every_env_object_up_facings = to_torch(obj_up_facings, dtype=torch.float, device=self.device)[self._every_env_object_ids]
        self._every_env_object_on_platform_trans = to_torch(obj_on_platform_trans, dtype=torch.float, device=self.device)[self._every_env_object_ids]

        self._obj_urdfs = obj_urdfs

        self._build_object_bps()

        return
    
    def _build_object_bps(self):

        bps_0 = torch.cat([     self._every_env_object_bbox_lengths[:, 0].unsqueeze(-1) / 2,      self._every_env_object_bbox_lengths[:, 1].unsqueeze(-1) / 2, -1 * self._every_env_object_bbox_lengths[:, 2].unsqueeze(-1) / 2], dim=-1)
        bps_1 = torch.cat([-1 * self._every_env_object_bbox_lengths[:, 0].unsqueeze(-1) / 2,      self._every_env_object_bbox_lengths[:, 1].unsqueeze(-1) / 2, -1 * self._every_env_object_bbox_lengths[:, 2].unsqueeze(-1) / 2], dim=-1)
        bps_2 = torch.cat([-1 * self._every_env_object_bbox_lengths[:, 0].unsqueeze(-1) / 2, -1 * self._every_env_object_bbox_lengths[:, 1].unsqueeze(-1) / 2, -1 * self._every_env_object_bbox_lengths[:, 2].unsqueeze(-1) / 2], dim=-1)
        bps_3 = torch.cat([     self._every_env_object_bbox_lengths[:, 0].unsqueeze(-1) / 2, -1 * self._every_env_object_bbox_lengths[:, 1].unsqueeze(-1) / 2, -1 * self._every_env_object_bbox_lengths[:, 2].unsqueeze(-1) / 2], dim=-1)
        bps_4 = torch.cat([     self._every_env_object_bbox_lengths[:, 0].unsqueeze(-1) / 2,      self._every_env_object_bbox_lengths[:, 1].unsqueeze(-1) / 2,      self._every_env_object_bbox_lengths[:, 2].unsqueeze(-1) / 2], dim=-1)
        bps_5 = torch.cat([-1 * self._every_env_object_bbox_lengths[:, 0].unsqueeze(-1) / 2,      self._every_env_object_bbox_lengths[:, 1].unsqueeze(-1) / 2,      self._every_env_object_bbox_lengths[:, 2].unsqueeze(-1) / 2], dim=-1)
        bps_6 = torch.cat([-1 * self._every_env_object_bbox_lengths[:, 0].unsqueeze(-1) / 2, -1 * self._every_env_object_bbox_lengths[:, 1].unsqueeze(-1) / 2,      self._every_env_object_bbox_lengths[:, 2].unsqueeze(-1) / 2], dim=-1)
        bps_7 = torch.cat([     self._every_env_object_bbox_lengths[:, 0].unsqueeze(-1) / 2, -1 * self._every_env_object_bbox_lengths[:, 1].unsqueeze(-1) / 2,      self._every_env_object_bbox_lengths[:, 2].unsqueeze(-1) / 2], dim=-1)
        
        self._every_env_object_bps = torch.cat([
            bps_0.unsqueeze(1),
            bps_1.unsqueeze(1),
            bps_2.unsqueeze(1),
            bps_3.unsqueeze(1),
            bps_4.unsqueeze(1),
            bps_5.unsqueeze(1),
            bps_6.unsqueeze(1),
            bps_7.unsqueeze(1)]
        , dim=1)
            
        self._every_env_object_bps += self._every_env_object_bbox_centers.unsqueeze(1) # (num_envs, 8, 3)

        return

class ObjectCategoryId(Enum):
    Plate = 2

class WithoutHumanoidFirst(BaseTask):
    """
    This class is a placeholder for any future functionality that might be added
    to handle scenarios without humanoid interactions.
    """

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):

        # manage multi task obs
        self._num_tasks = 2
        task_obs_size_carry = 3 + 3 + 6 + 3 + 3 + 3 * 8 # bps
        self._each_subtask_obs_size = [
            task_obs_size_carry, # new carry
            task_obs_size_carry, # old carry
        ]
        self._multiple_task_names = ["new_carry", "old_carry"]
        self._enable_task_mask_obs = False

        # configs for task
        self._enable_task_obs = cfg["env"]["enableTaskObs"]
        self._only_vel_reward = cfg["env"]["onlyVelReward"]
        self._only_height_handheld_reward = cfg["env"]["onlyHeightHandHeldReward"]

        self._enable_upperbody_penalty = cfg["env"]["enableStraightUpperBodyPenalty"]
        self._upperbody_coeff = cfg["env"]["upperbodyPenaltyCoeff"]

        self._box_vel_penalty = cfg["env"]["box_vel_penalty"]
        self._box_vel_pen_coeff = cfg["env"]["box_vel_pen_coeff"]
        self._box_vel_pen_thre = cfg["env"]["box_vel_pen_threshold"]

        self._mode = cfg["env"]["mode"] # determine which set of objects to use (train or test)
        assert self._mode in ["train", "test"]

        if cfg["args"].eval:
            self._mode = "test"

        # configs for box
        box_cfg = cfg["env"]["box"]

        self._reset_random_rot = box_cfg["reset"]["randomRot"]
        self._reset_random_height = box_cfg["reset"]["randomHeight"]
        self._reset_random_height_prob = box_cfg["reset"]["randomHeightProb"]
        self._reset_maxTopSurfaceHeight = box_cfg["reset"]["maxTopSurfaceHeight"]
        self._reset_minBottomSurfaceHeight = box_cfg["reset"]["maxTopSurfaceHeight"]

        self._enable_bbox_obs = box_cfg["obs"]["enableBboxObs"]

        self._obj_fall_allow_dist = box_cfg["objFallAllowDist"]
        self._enable_obj_fall_termination = box_cfg["enableObjFallTermination"]

        self._enable_leave_init_pos_rwd = box_cfg["enableLeaveInitPosRwd"]
        self._leave_coeff = box_cfg["leaveCoeff"]

        self._enable_walk_rwd = box_cfg["enable_walk_rwd"]

        self._disable_random_height_cateIds = [
            ObjectCategoryId["ArmChair_Normal"].value,
            ObjectCategoryId["Table_Circle"].value,
        ]

        self._is_eval = cfg["args"].eval
        self._eval_task = cfg["args"].eval_task
        if self._is_eval:
            cfg["env"]["box"]["build"]["objSpecified"] = self._eval_task

        # configs for amp
        state_init = cfg["env"]["stateInit"]
        self._state_init = HumanoidAdaptCarryBox2Objs.StateInit[state_init]
        self._hybrid_init_prob = cfg["env"]["hybridInitProb"]
        self._num_amp_obs_steps = cfg["env"]["numAMPObsSteps"]
        assert(self._num_amp_obs_steps >= 2)

        self._reset_default_env_ids = []
        self._reset_ref_env_ids = {} # to enable multi-skill reference init, use dict instead of list
        self._reset_ref_motion_ids = {}
        self._reset_ref_motion_times = {}

        self._power_reward = cfg["env"]["power_reward"]
        self._power_coefficient = cfg["env"]["power_coefficient"]

        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)
        
        self._skill = cfg["env"]["skill"]
        self._skill_init_prob = torch.tensor(cfg["env"]["skillInitProb"], device=self.device, dtype=torch.float) # probs for state init
        self._skill_disc_prob = torch.tensor(cfg["env"]["skillDiscProb"], device=self.device, dtype=torch.float) # probs for amp obs demo fetch

        motion_file = cfg['env']['motion_file']
        self._load_motion(motion_file)

        self._amp_obs_buf = torch.zeros((self.num_envs, self._num_amp_obs_steps, self._num_amp_obs_per_step), device=self.device, dtype=torch.float)
        self._curr_amp_obs_buf = self._amp_obs_buf[:, 0]
        self._hist_amp_obs_buf = self._amp_obs_buf[:, 1:]
        
        self._amp_obs_demo_buf = None

        # tensors for task
        self._prev_root_pos = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float)
        self._prev_root_rot = torch.zeros([self.num_envs, 4], device=self.device, dtype=torch.float)
        self._prev_box_pos = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float)
        self._tar_pos = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float) # target location of the box, 3d xyz

        spacing = cfg["env"]["envSpacing"]
        if spacing <= 0.5:
            self._tar_pos_dist = torch.distributions.uniform.Uniform(
                torch.tensor([-4.5, -4.5, 0.5], device=self.device),
                torch.tensor([4.5, 4.5, 1.0], device=self.device))
        else:
            self._tar_pos_dist = torch.distributions.uniform.Uniform(
                torch.tensor([-(spacing - 0.5), -(spacing - 0.5), 0.5], device=self.device),
                torch.tensor([(spacing - 0.5), (spacing - 0.5), 1.0], device=self.device))

        if (not self.headless):
            self._build_marker_state_tensors()

        # tensors for box
        self._build_box_tensors()

        # tensors for platforms
        if self._reset_random_height:
            self._build_platforms_state_tensors()

        # tensors for enableTrackInitState
        self._every_env_init_dof_pos = torch.zeros((self.num_envs, self.num_dof), device=self.device, dtype=torch.float)

        # tensors for fixing obs bug
        self._kinematic_humanoid_rigid_body_states = torch.zeros((self.num_envs, self.num_bodies, 13), device=self.device, dtype=torch.float)

        ###### evaluation!!!
        self._is_eval = cfg["args"].eval
        if self._is_eval:

            self._success_buf = torch.zeros((self.num_envs), device=self.device, dtype=torch.long)
            self._precision_buf = torch.zeros((self.num_envs), device=self.device, dtype=torch.float)

            self._success_threshold = cfg["env"]["eval"]["successThreshold"]

            self._skill = cfg["env"]["eval"]["skill"]
            self._skill_init_prob = torch.tensor(cfg["env"]["eval"]["skillInitProb"], device=self.device, dtype=torch.float) # probs for state init

        return

    def _create_envs(self, num_envs, spacing, num_per_row):
        if (not self.headless):
            self._marker_handles = []
            self._load_marker_asset()

        if self._reset_random_height:
            self._platform_handles = []
            self._tar_platform_handles = []
            self._load_platform_asset()
        
        # load objects
        self._obj_lib = ObjectLib_JerryTesting(
            mode=self._mode,
            dataset_root=os.path.join(os.path.dirname(self.cfg['env']['motion_file']), self.cfg["env"]["box"]["build"]["objRoot"]),
            dataset_categories=self.cfg["env"]["box"]["build"]["objCategories"],
            category_specified=self.cfg["env"]["box"]["build"]["objSpecified"],
            num_envs=self.num_envs,
            device=self.device,
        )

        # load physical assets
        self._object_handles = []
        self._object_assets = self._load_object_asset(self._obj_lib._obj_urdfs)

        super()._create_envs(num_envs, spacing, num_per_row)
        return

    def _load_platform_asset(self):
        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.linear_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        asset_options.density = 1.0
        asset_options.fix_base_link = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        self._platform_height = 2.0
        self._platform_asset = self.gym.create_box(self.sim, 0.4, 0.4, self._platform_height, asset_options)

        return

    def _build_platforms(self, env_id, env_ptr):
        col_group = env_id
        col_filter = 0
        segmentation_id = 0
        default_pose = gymapi.Transform()
        
        default_pose.p = self._obj_lib.every_env_object_platform_centers[env_id] # place under the ground
        platform_handle = self.gym.create_actor(env_ptr, self._platform_asset, default_pose, "platform", col_group, col_filter, segmentation_id)
        self.gym.set_rigid_body_color(env_ptr, platform_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.5, 0.235, 0.6)) 

        # default_pose.p.z = -5 - self._platform_height
        # tar_platform_handle = self.gym.create_actor(env_ptr, self._platform_asset, default_pose, "tar_platform", col_group, col_filter, segmentation_id)
        # self.gym.set_rigid_body_color(env_ptr, tar_platform_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.0, 0.0, 0.8))

        self._platform_handles.append(platform_handle)
        # self._tar_platform_handles.append(tar_platform_handle)

        return
    
    def _load_object_asset(self, object_urdfs):
        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.linear_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        asset_options.fix_base_link = False # obj can move!!
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        # Load materials from meshes
        asset_options.use_mesh_materials = True
        asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX

        # use default convex decomposition params
        asset_options.vhacd_enabled = True
        asset_options.vhacd_params = gymapi.VhacdParams()
        asset_options.vhacd_params.resolution = 100000
        asset_options.vhacd_params.max_convex_hulls = 128
        asset_options.vhacd_params.max_num_vertices_per_ch = 64

        asset_options.replace_cylinder_with_capsule = False # support cylinder

        asset_root = "tokenhsi/data/dataset_dynamic_objects/train" # assuming the urdf files are in the same directory as this script
        object_assets = []
        for urdf in object_urdfs:
            object_assets.append(self.gym.load_asset(self.sim, asset_root, urdf, asset_options))

        return object_assets
    
    def _build_object(self, env_id, env_ptr):
        # However we should actually build objects on platforms, does default_pose control that?
        # in fact, 
        col_group = env_id
        col_filter = 0
        segmentation_id = 0

        default_pose = gymapi.Transform()
        default_pose.p = self._obj_lib._every_env_object_on_platform_trans[env_id] # ensure no penetration between object and ground plane
        
        object_handle = self.gym.create_actor(env_ptr, self._object_assets[self._obj_lib._every_env_object_ids[env_id]], default_pose, "object", col_group, col_filter, segmentation_id)
        self._object_handles.append(object_handle)

        return
    
    def _build_env(self, env_id, env_ptr, humanoid_asset):
        super()._build_env(env_id, env_ptr, humanoid_asset)
        
        self._build_platforms(env_id, env_ptr)

        self._build_object(env_id, env_ptr)

        return


# set up environment using WithoutHumanoidFirst class
def main():
    # using lines 1~359 to set up the environment
    gym = gymapi.acquire_gym()

    # Configure simulation parameters
    sim_params = gymapi.SimParams()
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    
    # Configure PhysX parameters for stability
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 12  # Higher = more stable
    sim_params.physx.num_velocity_iterations = 4
    # sim_params.physx.use_ccd = True  # Prevent tunneling
    sim_params.dt = 1.0 / 300.0  # Smaller timestep for stability

    # Create simulation
    sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
    if sim is None:
        print("*** Failed to create sim")
        quit()

    # Load the WithoutHumanoidFirst task
    cfg = {
        "env": {
            "motion_file": "path/to/motion_file.json",
            "box": {
                "build": {
                    "objRoot": "path/to/obj_root",
                    "objCategories": ["Plate"],
                    "objSpecified": "Plate"
                },
                "reset": {
                    "randomRot": True,
                    "randomHeight": True,
                    "randomHeightProb": 0.5,
                }
            }
        }
    }
    sim_params = gymapi.SimParams()
    physics_engine = gymapi.SIM_PHYSX
    device_type = gymapi.GPU
    device_id = 0
    headless = False

    task = WithoutHumanoidFirst(cfg, sim_params, physics_engine, device_type, device_id, headless)

    # Create environments
    num_envs = 4
    spacing = 2.0
    num_per_row = 2
    task._create_envs(num_envs, spacing, num_per_row)

    # Run the simulation (this part is not implemented in the original code)
    # You would typically have a loop here to step the simulation and render the environments

    # Clean up
    gym.destroy_sim(sim)
    print("Simulation closed.")



# # --- 1. Environment Setup ---

# # Initialize Gym
# gym = gymapi.acquire_gym()

# # Configure simulation parameters
# sim_params = gymapi.SimParams()
# sim_params.up_axis = gymapi.UP_AXIS_Z
# sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)

# # Configure PhysX parameters for stability
# sim_params.physx.solver_type = 1  # TGS solver
# sim_params.physx.num_position_iterations = 12 # Higher = more stable
# sim_params.physx.num_velocity_iterations = 4
# # sim_params.physx.use_ccd = True # Prevent tunneling
# sim_params.dt = 1.0 / 300.0 # Smaller timestep for stability

# # Create simulation
# sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
# if sim is None:
#     print("*** Failed to create sim")
#     quit()

# # --- 2. Load OBJ Asset ---

# # Use the 'cone.urdf' which wraps the obj file
# asset_root = os.path.dirname(os.path.abspath(__file__))
# asset_file = "cone.urdf"

# # Configure asset options for better collision handling
# asset_options = gymapi.AssetOptions()
# asset_options.fix_base_link = False
# asset_options.thickness = 0.001  # Thin collision margin
# asset_options.convex_decomposition_from_submeshes = True

# # Load the asset
# print(f"Loading asset '{asset_file}' from '{asset_root}'")
# asset = gym.load_asset(sim, asset_root, asset_file, asset_options)
# if asset is None:
#     print(f"*** Failed to load asset {asset_file}")
#     quit()

# # --- Key Physics Tuning: Increase Friction ---
# rigid_props = gym.get_asset_rigid_shape_properties(asset)
# for prop in rigid_props:
#     prop.friction = 1.5  # Increase friction for better grip
# gym.set_asset_rigid_shape_properties(asset, rigid_props)


# # --- 3. Create Stacking Scene ---

# # Create a ground plane
# plane_params = gymapi.PlaneParams()
# plane_params.normal = gymapi.Vec3(0, 0, 1)
# gym.add_ground(sim, plane_params)

# # Set up the environment
# env_spacing = 2.0
# env_lower = gymapi.Vec3(-env_spacing, -env_spacing, 0.0)
# env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)
# env = gym.create_env(sim, env_lower, env_upper, 1)

# # --- 4. Load Platform Asset ---
# platform_asset_options = gymapi.AssetOptions()
# platform_asset_options.fix_base_link = True
# platform_height = 0.1
# platform_width = 0.5
# platform_asset = gym.create_box(sim, platform_width, platform_width, platform_height, platform_asset_options)


# # --- 5. Get Plate Dimensions & Create Actors ---

# # Get object dimensions by creating a temporary actor
# temp_pose = gymapi.Transform()
# temp_pose.p = gymapi.Vec3(0, 0, -10) # Place it far away
# temp_actor = gym.create_actor(env, asset, temp_pose, "temp", 0, 0)
# body_shape_props = gym.get_actor_rigid_body_shape_properties(env, temp_actor)
# plate_height = body_shape_props[0].box.size.z
# gym.remove_actor(env, temp_actor) # Clean up the temporary actor

# # Actor Creation Loop
# num_platforms = 2
# for i in range(num_platforms):
#     # Create platform
#     platform_pose = gymapi.Transform()
#     platform_pose.p = gymapi.Vec3(i * 1.0, 0, platform_height / 2)
#     gym.create_actor(env, platform_asset, platform_pose, f"platform_{i}", 0, 0)

#     # Create plate on top of the platform
#     plate_pose = gymapi.Transform()
#     plate_pose.p = gymapi.Vec3(i * 1.0, 0, platform_height + plate_height / 2 + 0.01)
#     gym.create_actor(env, asset, plate_pose, f"plate_{i}", 0, 0)


# # --- 6. Create Viewer and Run Simulation ---
# viewer = gym.create_viewer(sim, gymapi.CameraProperties())
# if viewer is None:
#     print("*** Failed to create viewer")
#     quit()

# # Position the camera
# cam_pos = gymapi.Vec3(1, -2, 1)
# cam_target = gymapi.Vec3(1, 0, 0.5)
# gym.viewer_camera_look_at(viewer, env, cam_pos, cam_target)

# # Simulation loop
# while not gym.query_viewer_has_closed(viewer):
#     # Step the physics
#     gym.simulate(sim)
#     gym.fetch_results(sim, True)

#     # Update the viewer
#     gym.step_graphics(sim)
#     gym.draw_viewer(viewer, sim, True)

#     # Synchronize simulation frame rate
#     gym.sync_frame_time(sim)

# print("Closing simulation.")
# gym.destroy_viewer(viewer)
# gym.destroy_sim(sim)
# # This returns a list of bounds for each rigid body in the actor.
# bounds = gym.get_actor_rigid_body_shape_bounds(env, temp_actor)

# # For a single-body asset, we access the first element.
# # The bounds object has 'min' and 'max' attributes which are Vec3.
# obj_height = bounds[0].max.z - bounds[0].min.z

# # Clean up the temporary actor
# gym.remove_actor(env, temp_actor)
# # --- End of dimension calculation ---


# # Initial poses for the stacked tower
# num_objects_to_stack = 10
# spacing_z = obj_height * 1.05  # 5% gap to avoid initial intersection

# print(f"Stacking {num_objects_to_stack} objects...")
# for i in range(num_objects_to_stack):
#     # Define the pose for each actor in the stack
#     pose = gymapi.Transform()
#     pose.p = gymapi.Vec3(0, 0, (obj_height / 2.0) + i * spacing_z)
    
#     actor_handle = gym.create_actor(env, asset, pose, f"obj_{i}", 0, 0)
    
#     # --- Troubleshooting: Increase Damping per-actor ---
#     actor_props = gym.get_actor_rigid_body_properties(env, actor_handle)
#     for prop in actor_props:
#         prop.angular_damping = 0.5  # Add damping to reduce wobbling
#     gym.set_actor_rigid_body_properties(env, actor_handle, actor_props)


# # --- 4. Run Simulation ---

# # Setup the viewer
# viewer = gym.create_viewer(sim, gymapi.CameraProperties())
# if viewer is None:
#     print("*** Failed to create viewer")
#     quit()

# # Position the camera to look at the middle of the tower
# cam_pos = gymapi.Vec3(3, 3, obj_height * num_objects_to_stack / 2)
# cam_target = gymapi.Vec3(0, 0, obj_height * num_objects_to_stack / 2)
# gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

# # Main simulation loop
# print("--- Simulation running. Close the viewer to exit. ---")
# while not gym.query_viewer_has_closed(viewer):
#     # Step the physics
#     gym.simulate(sim)
#     gym.fetch_results(sim, True)

#     # Update the viewer
#     gym.step_graphics(sim)
#     gym.draw_viewer(viewer, sim, True)

#     # Synchronize simulation frame rate
#     gym.sync_frame_time(sim)

# # --- Cleanup ---
# print("--- Simulation finished. ---")
# gym.destroy_viewer(viewer)
# gym.destroy_sim(sim)