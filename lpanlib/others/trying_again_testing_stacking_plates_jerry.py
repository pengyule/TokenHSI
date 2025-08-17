from isaacgym import gymapi, gymutil
import sys, os, json
import numpy as np

def main():
    # --- parse type argument ---
    t = 1
    if "--type" in sys.argv:
        idx = sys.argv.index("--type")
        if idx + 1 < len(sys.argv):
            t = int(sys.argv[idx + 1])
            sys.argv.pop(idx)
            sys.argv.pop(idx)

    # --- Isaac Gym args ---
    args = gymutil.parse_arguments(description="Plate spawn test")

    # --- asset paths ---
    asset_root = "/home/jerrypyl/TokenHSI/tokenhsi/data/dataset_dynamic_objects"
    mode = "train"
    category = "Plate"
    object_name = "0000"
    urdf_path = os.path.join(asset_root, mode, category, object_name, "asset.urdf")
    json_path = os.path.join(asset_root, mode, category, object_name, "config.json")

    # --- load platform metadata ---
    with open(json_path, "r") as f:
        meta = json.load(f)
    platform_height = meta.get("platform_height", 3.0)
    platform_width = meta.get("platform_width", 1.0)
    platform_thickness = meta.get("platform_thickness", 0.01)
    object_bbox = meta.get("bbox")

    # --- init gym ---
    gym = gymapi.acquire_gym()
    sim_params = gymapi.SimParams()
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.dt = 1/60.0
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
    sim = gym.create_sim(args.compute_device_id, args.graphics_device_id,
                         args.physics_engine, sim_params)
    if sim is None:
        print("Failed to create sim")
        return

    # --- add ground ---
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0,0,1)
    gym.add_ground(sim, plane_params)

    # --- viewer ---
    cam_props = gymapi.CameraProperties()
    viewer = gym.create_viewer(sim, cam_props)
    cam_pos = gymapi.Vec3(2.0,-2.0,3.5)
    cam_target = gymapi.Vec3(0,0,3.0)
    gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

    # --- environment ---
    env_lower = gymapi.Vec3(-2.0,-2.0,0.0)
    env_upper = gymapi.Vec3(2.0,2.0,2.0)
    env = gym.create_env(sim, env_lower, env_upper, 1)

    # --- platform (static) ---
    platform_options = gymapi.AssetOptions()
    platform_options.fix_base_link = True  # makes it truly immovable
    platform_asset = gym.create_box(sim, platform_width/2, platform_width/2, platform_thickness/2, platform_options)
    platform_pose = gymapi.Transform()
    platform_pose.p = gymapi.Vec3(0,0,platform_height)
    gym.create_actor(env, platform_asset, platform_pose, "platform", 0, 0)

    # --- plate asset ---
    plate_options = gymapi.AssetOptions()
    plate_options.fix_base_link = False  # dynamic
    plate_asset = gym.load_asset(sim, os.path.dirname(urdf_path), os.path.basename(urdf_path), plate_options)

    # --- first plate placement ---
    # initiate a list of plate poses
    num_plates = 10
    plate_pose = [gymapi.Transform() for _ in range(num_plates)]
    plate_handle = [None] * num_plates
    if t == 1:
        plate_pose[0].p = gymapi.Vec3(
            0, 0, platform_height + platform_thickness/2 + object_bbox[2]
        )
    elif t == 2:
        plate_pose[0].p = gymapi.Vec3(0, 0, platform_height + 1.0)
    plate_pose[0].r = gymapi.Quat.from_euler_zyx(0.0, 3.1415926, 0.0)
    plate_handle[0] = gym.create_actor(env, plate_asset, plate_pose[0], "plate1", 0, 0)

    # --- second plate placement (stacked, slightly misaligned) ---
    for i in range(1,10):
        # within 0.1 normal distribution random misplacement
        misalign_offset = np.random.normal(0.0, 0.01)
        plate_pose[i].p = gymapi.Vec3(
            plate_pose[0].p.x + misalign_offset,
            plate_pose[0].p.y + misalign_offset,
            plate_pose[i-1].p.z + object_bbox[2]  # stack on top of first plate
        )
        plate_pose[i].r = gymapi.Quat.from_euler_zyx(0.0, 3.1415926, 0.0)
        plate_handle[i] = gym.create_actor(env, plate_asset, plate_pose[i], f"plate{i+1}", 0, 0)

    # --- simulation loop ---
    while not gym.query_viewer_has_closed(viewer):
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)

    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)

if __name__ == "__main__":
    main()
