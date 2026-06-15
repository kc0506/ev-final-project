# Stub package: makes `import simple_knn._C` resolve WITHOUT the built CUDA ext.
# Rationale: physdreamer.gaussian_3d.scene.gaussian_model imports `distCUDA2` at
# module load, but that symbol is only CALLED inside create_from_pcd / random-init
# (GaussianModel(3)+load_ply never touches it). The MPM->flow path never renders
# gaussians, so it never calls distCUDA2 -- it only needs the import line to resolve.
# The real ext must still be built for RGB render / scene-from-PLY (see Task: env debt).
