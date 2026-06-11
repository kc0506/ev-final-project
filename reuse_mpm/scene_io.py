"""Single entry to load a scene from a `SceneSpec`, dispatching pd/pg.

Replaces the pd/pg branch (load_scene vs load_physgaussian_scene + cache-path
derivation) that was copy-pasted across every entrypoint. `grid_size` is taken
from the SimConfig (not the SceneSpec) so it can never disagree with the rollout.

Side effect: if `spec.cache_path` was None, it is filled in with the resolved
default path, so the serialized SceneSpec records exactly which discretisation
cache was used.
"""
from __future__ import annotations

from .config import SceneSpec, SimConfig
from .scene import SceneBundle, default_cache_path, load_scene


def load_from_spec(spec: SceneSpec, sim: SimConfig) -> SceneBundle:
    """Load a SceneBundle described by `spec`, using `sim.grid_size` for the MPM grid."""
    if spec.kind == "pg":
        from .scene_physgaussian import (
            default_pg_cache_path,
            load_physgaussian_scene,
        )

        if spec.cache_path is None:
            spec.cache_path = default_pg_cache_path(
                spec.path, spec.downsample_scale, sim.grid_size, spec.top_k
            )
        return load_physgaussian_scene(
            spec.path,
            name=spec.name,
            device=spec.device,
            downsample_scale=spec.downsample_scale,
            grid_size=sim.grid_size,
            top_k=spec.top_k,
            max_particles=spec.max_particles,
            freeze_frac=spec.freeze_frac,
            freeze_axis=spec.freeze_axis,
            cache_path=spec.cache_path,
        )
    elif spec.kind == "pd":
        if spec.cache_path is None:
            spec.cache_path = default_cache_path(
                spec.path, spec.downsample_scale, sim.grid_size, spec.top_k
            )
        return load_scene(
            spec.path,
            name=spec.name,
            device=spec.device,
            downsample_scale=spec.downsample_scale,
            grid_size=sim.grid_size,
            top_k=spec.top_k,
            max_particles=spec.max_particles,
            cache_path=spec.cache_path,
            freeze_mode=spec.freeze_mode,
            freeze_frac=spec.freeze_frac,
            freeze_axis=spec.freeze_axis,
        )
    raise ValueError(f"unknown scene kind {spec.kind!r} (expected 'pd' or 'pg')")
