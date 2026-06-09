"""Differentiable MPM rollout as a torch.autograd.Function.

This is a vendored copy of PhysDreamer's `MPMDifferentiableSimulation`
(projects/uncleaned_train/.../interface.py), re-bound to our single source of
truth `physdreamer.warp_mpm.*` instead of the duplicate `thirdparty_code.warp_mpm`.
We copy it (rather than import) precisely to avoid pulling in the second warp_mpm
copy and the motionrep namespace.

Behaviour (unchanged):
  - forward re-initialises state from (init_pos, init_velocity), runs
    `extra_no_grad_steps` substeps detached, then `num_substeps` substeps under a
    warp tape, and returns the final particle positions.
  - When E is a 0-dim tensor (scalar global E), backward returns a *scalar*
    aggregated dL/dE -- exactly what we need for global-E recovery.
  - The windowed BPTT memory trick = (extra_no_grad_steps detached) +
    (num_substeps with grad).
"""
from typing import Optional, Union

import torch
import torch.autograd as autograd
from torch import Tensor

import warp as wp

from . import _env  # ensures physdreamer + warp on path
from physdreamer.warp_mpm.warp_utils import from_torch_safe, MyTape, CondTape
from physdreamer.warp_mpm.mpm_utils import (
    compute_posloss_with_grad,
    aggregate_grad,
)
from physdreamer.warp_mpm.mpm_data_structure import get_float_array_product


class MPMDifferentiableSimulation(autograd.Function):
    @staticmethod
    def forward(
        ctx,
        mpm_solver,
        mpm_state,
        mpm_model,
        substep: int,
        substep_size: float, # [note] substep delta_t
        num_substeps: int,
        init_pos: Tensor,
        init_velocity: Tensor,
        E: Tensor,
        nu: Tensor,
        particle_density: Optional[Tensor] = None,
        density_change_mask: Optional[Tensor] = None,
        static_pos: Optional[Tensor] = None,
        device: str = "cuda:0",
        requires_grad: bool = True,
        extra_no_grad_steps: int = 0,
    ) -> Tensor:
        num_particles = init_pos.shape[0]
        if static_pos is None:
            mpm_state.reset_state(
                init_pos.clone(), None, init_velocity,
                device=device, requires_grad=requires_grad,
            )
        else:
            mpm_state.reset_state(
                static_pos.clone(), None, init_velocity,
                device=device, requires_grad=requires_grad,
            )
            init_xyzs_wp = from_torch_safe(
                init_pos.clone().detach().contiguous(), dtype=wp.vec3,
                requires_grad=requires_grad,
            )
            mpm_solver.restart_and_compute_F_C(
                mpm_model, mpm_state, init_xyzs_wp, device=device
            )

        if E.ndim == 0:
            E_inp = E.item()
            ctx.aggregating_E = True
        else:
            E_inp = from_torch_safe(E, dtype=wp.float32, requires_grad=requires_grad)
            ctx.aggregating_E = False
        if nu.ndim == 0:
            nu_inp = nu.item()
            ctx.aggregating_nu = True
        else:
            nu_inp = from_torch_safe(nu, dtype=wp.float32, requires_grad=requires_grad)
            ctx.aggregating_nu = False

        mpm_solver.set_E_nu(mpm_model, E_inp, nu_inp, device=device)
        mpm_state.reset_density(
            tensor_density=particle_density,
            selection_mask=density_change_mask,
            device=device, requires_grad=requires_grad,
        )

        prev_state = mpm_state
        if extra_no_grad_steps > 0:
            with torch.no_grad():
                wp.launch(
                    kernel=get_float_array_product, dim=num_particles,
                    inputs=[mpm_state.particle_density, mpm_state.particle_vol,
                            mpm_state.particle_mass],
                    device=device,
                )
                mpm_solver.prepare_mu_lam(mpm_model, mpm_state, device=device)
                for _ in range(extra_no_grad_steps):
                    next_state = prev_state.partial_clone(requires_grad=requires_grad)
                    mpm_solver.p2g2p_differentiable(
                        mpm_model, prev_state, next_state, substep_size, device=device
                    )
                    prev_state = next_state

        wp_tape = MyTape()
        cond_tape = CondTape(wp_tape, requires_grad)
        next_state_list = []
        with cond_tape:
            wp.launch(
                kernel=get_float_array_product, dim=num_particles,
                inputs=[prev_state.particle_density, prev_state.particle_vol,
                        prev_state.particle_mass],
                device=device,
            )
            mpm_solver.prepare_mu_lam(mpm_model, prev_state, device=device)
            for _ in range(num_substeps):
                next_state = prev_state.partial_clone(requires_grad=requires_grad)
                mpm_solver.p2g2p_differentiable(
                    mpm_model, prev_state, next_state, substep_size, device=device
                )
                next_state_list.append(next_state)
                prev_state = next_state

        ctx.mpm_solver = mpm_solver
        ctx.mpm_state = mpm_state
        ctx.mpm_model = mpm_model
        ctx.tape = cond_tape.tape
        ctx.device = device
        ctx.num_particles = num_particles
        ctx.next_state_list = next_state_list
        ctx.save_for_backward(density_change_mask)

        last_state = next_state_list[-1]
        return wp.to_torch(last_state.particle_x).detach().clone()

    @staticmethod
    def backward(ctx, out_pos_grad: Tensor):
        # out_pos_grad = dL/dx.

        num_particles = ctx.num_particles
        tape, device = ctx.tape, ctx.device
        mpm_solver, mpm_state, mpm_model = ctx.mpm_solver, ctx.mpm_state, ctx.mpm_model
        last_state = ctx.next_state_list[-1]
        density_change_mask = ctx.saved_tensors[0]

        # [note] why this? Because we have to emulate a dL/dx where warp knows it.
        grad_pos_wp = from_torch_safe(out_pos_grad, dtype=wp.vec3, requires_grad=False)
        target_pos_detach = wp.clone(
            last_state.particle_x, device=device, requires_grad=False
        )
        with tape:
            loss_wp = torch.zeros(1, device=device)
            loss_wp = wp.from_torch(loss_wp, requires_grad=True)
            wp.launch(
                compute_posloss_with_grad, dim=num_particles,
                inputs=[last_state, target_pos_detach, grad_pos_wp, 0.5, loss_wp],
                device=device,
            )
        tape.backward(loss_wp)

        pos_grad = None
        velo_grad = (
            None if mpm_state.particle_v.grad is None
            else wp.to_torch(mpm_state.particle_v.grad).detach().clone()
        )

        if ctx.aggregating_E:
            E_grad = wp.from_torch(torch.zeros(1, device=device), requires_grad=False)
            wp.launch(aggregate_grad, dim=num_particles,
                      inputs=[E_grad, mpm_model.E.grad], device=device)
            E_grad = wp.to_torch(E_grad)[0] / num_particles
        else:
            E_grad = wp.to_torch(mpm_model.E.grad).detach().clone()

        if ctx.aggregating_nu:
            nu_grad = wp.from_torch(torch.zeros(1, device=device), requires_grad=False)
            wp.launch(aggregate_grad, dim=num_particles,
                      inputs=[nu_grad, mpm_model.nu.grad], device=device)
            nu_grad = wp.to_torch(nu_grad)[0] / num_particles
        else:
            nu_grad = wp.to_torch(mpm_model.nu.grad).detach().clone()

        if mpm_state.particle_density.grad is None:
            density_grad = None
        else:
            density_grad = wp.to_torch(mpm_state.particle_density.grad).detach()
            density_grad = density_grad[density_change_mask.type(torch.bool)]

        tape.zero()
        return (None, None, None, None, None, None,
                pos_grad, velo_grad, E_grad, nu_grad,
                density_grad, None, None, None, None, None)
