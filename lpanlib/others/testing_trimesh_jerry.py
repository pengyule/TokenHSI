import trimesh
import numpy as np

# ---- Parameters ----
bottom_radius = 0.12
top_radius    = 0.04
height        = 0.02

# Compute full cone height so slicing at `height` yields your frustum
full_cone_height = height / (1.0 - (top_radius / bottom_radius))

# Create a full cone (base at z=0, tip at z=full_cone_height)
full_cone = trimesh.creation.cone(radius=bottom_radius, height=full_cone_height)

# Slice plane: keep part below `height`
slice_origin = np.array([0.0, 0.0, height])
slice_normal = np.array([0.0, 0.0, -1.0])  # plane pointing upward

truncated = full_cone.slice_plane(
    plane_origin=slice_origin,
    plane_normal=slice_normal
)

# --- Cap the open top by triangulating the section loop ---
section = full_cone.section(plane_origin=slice_origin, plane_normal=slice_normal)
if section is not None:
    # Move the 3D section to 2D, triangulate, then map back to 3D
    slice_2D, to_3D = section.to_planar()

    tri_2D_list = []
    for poly in slice_2D.polygons_full:
        vertices_2D, faces = trimesh.creation.triangulate_polygon(poly)

        # Pad 2D → 3D (z=0), then transform into section plane
        vertices_3D = np.column_stack([vertices_2D, np.zeros(len(vertices_2D))])
        vertices_3D = trimesh.transform_points(vertices_3D, to_3D)

        tri_3D = trimesh.Trimesh(vertices=vertices_3D, faces=faces, process=False)
        tri_2D_list.append(tri_3D)

    # Combine triangulated cap pieces
    cap = trimesh.util.concatenate(tri_2D_list)
    truncated_capped = trimesh.util.concatenate([truncated, cap])

else:
    truncated_capped = truncated  # fallback if section failed

# Fix normals/winding
trimesh.repair.fix_normals(truncated_capped)

# Export OBJ
truncated_capped.export('truncated_cone.obj')

truncated_capped.show()

print("Exported truncated_cone.obj")
print("Watertight:", truncated_capped.is_watertight)
