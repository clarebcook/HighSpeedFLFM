from .ct_scan_points import key_features_ant, stable_vertices_ant, M_mesh_ant

# a couple important file paths 
mesh_filename = "stl_files/ant_mesh_no_mandibles.stl"
mesh_with_mandibles_filename = "stl_files/ant_mesh_downsampled.stl"
mesh_scale = 100
display_mesh_scale = 10

__all__ = ["key_features_ant", "stable_vertices_ant", "M_mesh_ant", "mesh_filename", "mesh_with_mandibles_filename"]