"""reuse_mpm: a clean layer on top of PhysDreamer for the reuse-MPM-to-video task.

One shared simulate_and_render code path drives forward generation, dataset
generation, and (later) inverse training, so render conventions match by
construction.
"""
