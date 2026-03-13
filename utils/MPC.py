import torch
from tqdm import tqdm
import math
import warnings


def empty_cache():
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


class MPC:
    """Model Predictive Control for single-agent (control-only) systems.

    Uses the same refactored pattern as RobustMPC:
      - optimize(): unified sample-and-select loop
      - _select_best(): select best sample and update tensors
    """

    def __init__(
        self,
        dT,
        horizon,
        receding_horizon,
        num_samples,
        dynamics_,
        device,
        mode="MPC",
        sample_mode="gaussian",
        lambda_=0.01,
        style="direct",
        num_warm_start_iters=0,
        num_full_horizon_iters=0,
        # Legacy compat: if provided, overrides the two above
        num_iterative_refinement=None,
        warm_start_only=False,
    ):
        self.horizon = horizon
        self.num_samples = num_samples
        self.device = device
        self.receding_horizon = receding_horizon
        self.dynamics_ = dynamics_

        self.dT = dT
        self.lambda_ = lambda_

        self.mode = mode
        self.sample_mode = sample_mode
        self.style = style  # "direct" or "receding"
        self.policy_dynamics_ = dynamics_

        # Resolve iteration counts
        if num_warm_start_iters == 0 and num_full_horizon_iters == 0:
            # Raise warning about deprecated num_iterative_refinement
            warnings.warn(
                "num_iterative_refinement is deprecated. Use num_warm_start_iters and num_full_horizon_iters instead."
            )
            # Legacy path: num_iterative_refinement=-1 means terminal MPC (pure policy rollout)
            if num_iterative_refinement == -1:
                num_warm_start_iters = -1
                num_full_horizon_iters = 0
            elif warm_start_only:
                # warm_start_only: all iters are warm-start, skip full-horizon
                num_warm_start_iters = num_iterative_refinement
                num_full_horizon_iters = 0
            else:
                # Original: warm takes int(n*0.4) iters; full takes the remainder.
                # Total = n+1 steps (matches MPC.py's get_control: range(n+1-warm)).
                num_warm_start_iters = int(num_iterative_refinement * 0.4)
                num_full_horizon_iters = (num_iterative_refinement + 1) - num_warm_start_iters
        self.num_warm_start_iters = num_warm_start_iters
        self.num_full_horizon_iters = num_full_horizon_iters

    def update_attributes(self, T, t, policy):
        self.T = T
        self.horizon = math.ceil(T / self.dT)  # Full horizon of problem
        self.incremental_horizon = math.ceil((T - t) / self.dT)  # Incremental horizon of MPC
        self.t = t
        self.policy = policy

    def get_batch_data(self, initial_condition_tensor, T, policy=None, t=0.0):
        """Generate a batch of the MPC dataset.
        Steps:
            1. Updates attributes based no T (rollout start) and t (incremental horizon rollout end)
            2. Runs self.get_opt_trajs to receive arrays of size [B, T, D] of optimal trajectories & cost labels
            3. Filters (coords, labels) to include valid data (i.e. until worst value along a traj. is reached)
                - After the worst value is reached, the cost labels are not accurate.

        Args:
            initial_condition_tensor: [B, D] initial states (B=batch size, D=dim of state)
            T: MPC total time horizon
            t: Time remaining after incremental horizon (current curriculum time reached)
                - This is the time up to which the deepreach model should be accurate
            policy: Current DeepReach model

        Returns:
            (costs, state_trajs, coords, value_labels)
        """
        self.update_attributes(T, t, policy)
        self.batch_size = initial_condition_tensor.shape[0]
        state_trajs, bdry_vals = self.get_opt_trajs(initial_condition_tensor)
        if self.dynamics_.set_mode in ["avoid", "reach"]:
            lxs = bdry_vals[0]
            costs, _ = torch.min(lxs, dim=-1)
        elif self.dynamics_.set_mode == "reach_avoid":
            avoid_values, reach_values = bdry_vals
            costs = torch.min(torch.maximum(reach_values, torch.cummax(-avoid_values, dim=-1).values), dim=-1).values
        else:
            raise NotImplementedError

        # Generate bootstrapped (coord, value) training data
        coords = torch.empty((0, self.dynamics_.state_dim + 1)).to(self.device)
        value_labels = torch.empty(0).to(self.device)

        if self.dynamics_.set_mode in ["avoid", "reach"]:
            _, min_idx = torch.min(lxs, dim=-1)
        elif self.dynamics_.set_mode == "reach_avoid":
            _, min_idx = torch.min(
                torch.clamp(reach_values, min=torch.max(-avoid_values, dim=-1).values.unsqueeze(-1)), dim=-1
            )

        for i in range(self.horizon):
            coord_i = torch.zeros(self.batch_size, self.dynamics_.state_dim + 1).to(self.device)
            coord_i[:, 0] = self.T - i * self.dT
            coord_i[:, 1:] = state_trajs[:, i, :] * 1.0

            # Indices are only valid until they've reached the "worst" value along a trajectory
            if self.dynamics_.set_mode in ["avoid", "reach"]:
                valid_idx = (min_idx > i).nonzero(as_tuple=True)
                value_labels_i = torch.min(lxs[valid_idx[0], i:], dim=-1).values
                coord_i = coord_i[valid_idx]
            elif self.dynamics_.set_mode == "reach_avoid":
                valid_idx = (min_idx > i).nonzero(as_tuple=True)
                value_labels_i = torch.min(
                    torch.clamp(
                        reach_values[valid_idx[0], i:],
                        min=torch.max(-avoid_values[valid_idx[0], i:], dim=-1).values.unsqueeze(-1),
                    ),
                    dim=-1,
                ).values
                coord_i = coord_i[valid_idx]
            else:
                raise NotImplementedError
            # add to data
            coords = torch.cat((coords, coord_i), dim=0)
            value_labels = torch.cat((value_labels, value_labels_i), dim=0)

        ##################### only use in range labels ###################################################
        output1 = torch.all(coords[..., 1:] >= self.dynamics_.state_range_[:, 0] - 0.01, -1, keepdim=False)
        output2 = torch.all(coords[..., 1:] <= self.dynamics_.state_range_[:, 1] + 0.01, -1, keepdim=False)
        in_range_index = torch.logical_and(torch.logical_and(output1, output2), ~torch.isnan(value_labels))

        coords = coords[in_range_index]
        value_labels = value_labels[in_range_index]
        ###################################################################################################
        coords = self.dynamics_.coord_to_input(coords)

        empty_cache()

        return (
            costs,
            state_trajs,
            coords.detach().cpu().clone(),
            value_labels.detach().cpu().clone(),
        )

    def _get_opt_trajs_terminal(self, init_state):
        """
        Generate optimal costs & trajectories by rolling out policy only (no MPC)

        Args:
            init_state: [B, D] initial states

        Returns:
            (state_trajs, boundary_tuple)
        """
        best_trajs, _ = self._rollout_deterministic_policy(
            init_state,
            t_start=self.horizon * self.dT,
            horizon=self.horizon,
            start_iter=0,
        )
        boundary_tuple = self.dynamics_.get_boundary_values_tuple(best_trajs)
        return best_trajs, boundary_tuple

    def _get_opt_trajs_direct(self, init_state):
        """
        Generate optimal costs & trajectories associated with initial states.
            1. If num_warm_start_iters > 0, optimize MPC with bootstrapped cost_to_go over self.incremental_horizon
               for self.num_warm_start_iters iterations
            2. If num_full_horizon_iters > 0, optimize MPC over full horizon for num_full_horizon_iters iterations
                - If num_warm_start_iters > 0, it will initialize with the best_controls guess from the warmstarting

        Args:
            init_state: [B, D] initial states

        Returns:
            (state_trajs, boundary_tuple)
        """
        # Phase 1: Assumes cost-to-go at self.t is accurate
        if self.num_warm_start_iters > 0:
            if self.T > self.t:
                # Run MPC optimization from self.T to self.t and bootstrap with policy self.t to 0
                best_trajs = self.optimize(
                    init_state,
                    t_start=self.horizon * self.dT,
                    num_iters=self.num_warm_start_iters,
                    t_remaining=self.t,
                    horizon=self.incremental_horizon,
                )
            elif self.policy is not None:
                # Only do deterministic rollout from self.T to 0 (policy)
                best_trajs, _ = self._rollout_deterministic_policy(
                    init_state,
                    t_start=self.horizon * self.dT,
                    horizon=self.horizon,
                    start_iter=0,
                )

        # Phase 2: Full-horizon refinement
        if self.num_full_horizon_iters > 0:
            best_trajs = self.optimize(
                init_state,
                t_start=self.horizon * self.dT,
                num_iters=self.num_full_horizon_iters,
                t_remaining=0,
                horizon=self.horizon,
            )

        boundary_tuple = self.dynamics_.get_boundary_values_tuple(best_trajs)
        return best_trajs, boundary_tuple

    def _get_opt_trajs_receding(self, init_state):
        if self.dynamics_.set_mode == "reach_avoid":
            raise NotImplementedError

        state_trajs = torch.zeros((self.batch_size, self.horizon + 1, self.dynamics_.state_dim)).to(self.device)
        state_trajs[:, 0, :] = init_state

        self.init_input_tensors()
        if self.policy is not None:
            num_limited_horizon_rollouts = int(self.num_iterative_refinement * 0.4)
            self.optimize(
                init_state,
                t_start=self.t + self.incremental_horizon * self.dT,
                num_iters=num_limited_horizon_rollouts,
                t_remaining=self.t,
                horizon=self.incremental_horizon,
            )

        lxs = torch.zeros(self.batch_size, self.horizon).to(self.device)
        for i in tqdm(range(int(self.horizon / self.receding_horizon))):
            best_controls, _ = self._optimize_receding(state_trajs[:, i, :])
            for k in range(self.receding_horizon):
                lxs[:, i * self.receding_horizon + k] = self.dynamics_.boundary_fn(
                    state_trajs[:, i * self.receding_horizon + k, :]
                )
                state_trajs[:, i * self.receding_horizon + 1 + k, :] = self.get_next_step_state(
                    state_trajs[:, i * self.receding_horizon + k, :], best_controls[:, k, :]
                )
                self.receding_start += 1
        lxs[:, -1] = self.dynamics_.boundary_fn(state_trajs[:, -1, :])
        return state_trajs, (lxs)

    def get_opt_trajs(self, init_state):
        """Orchestrate warm-start + full-horizon refinement."""
        self.init_input_tensors()
        if self.style == "terminal" or self.num_warm_start_iters == -1:
            return self._get_opt_trajs_terminal(init_state)
        elif self.style == "direct":
            return self._get_opt_trajs_direct(init_state)
        elif self.style == "receding":
            return self._get_opt_trajs_receding(init_state)
        else:
            raise ValueError(f"Unknown style {self.style}")

    def set_control_tensors(self, control_tensors, start_iter, end_iter):
        """
        Current "optimal" control tensor [B, H, D] updated.
        Only gets queried in other methods for rollout.
        """
        self.control_tensors[:, start_iter:end_iter] = control_tensors

    def optimize(self, init_state, t_start, num_iters, t_remaining=None, horizon=None):
        """Unified optimization loop for MPC.
        Runs sampling-based MPC with optionally bootstrapped final cost to go estimate

        Args:
            init_state: [B, D] initial states
            num_iters: Number of sample-and-select iterations (how many optimization cycles)
            t_remaining: Time remaining for terminal VF evaluation
            horizon: Rollout horizon

        Returns:
            (best_traj): Best trajectory found for the cost objective
        """
        best_traj = None
        for _ in range(num_iters):
            # 1. Sample and rollout
            state_trajs, inputs_tuple = self._rollout_with_sampling(
                init_state,
                t_start,
                horizon=horizon,
                start_iter=0,
            )
            # 2. Compute costs over trajectory
            costs = self.dynamics_.cost_fn(state_trajs)

            # 3. Optionally add terminal VF cost
            if t_remaining > 0:
                terminal_states = state_trajs[:, :, -1]
                terminal_values = self._eval_vf(terminal_states, t_remaining)
                if horizon > 0:
                    costs = torch.minimum(costs, terminal_values)
                    if self.dynamics_.set_mode == "reach_avoid":
                        avoid_value_max = torch.max(-self.dynamics_.avoid_fn(state_trajs), dim=-1).values
                        costs = torch.maximum(costs, avoid_value_max)
                else:
                    costs = terminal_values * 1.0

            # 4. Select best and update tensors
            best_traj = self._select_best(costs, inputs_tuple, state_trajs)

            empty_cache()

        # 5. If limited horizon, fill final best_traj's remaining horizon with policy
        if t_remaining > 0:
            remaining_horizon = self.horizon - horizon
            best_final_state = best_traj[:, -1, :]
            remaining_traj, _ = self._rollout_deterministic_policy(
                best_final_state,
                t_start=remaining_horizon * self.dT,
                horizon=remaining_horizon,
                start_iter=horizon,
            )
            best_traj = torch.cat([best_traj[:, :-1, :], remaining_traj], dim=1)
        return best_traj

    def _optimize_receding(self, init_state, t_start=None):
        """Single-iteration sample-and-select for receding horizon style."""
        state_trajs, inputs_tuple = self._rollout_with_sampling(
            init_state,
            t_start,
            horizon=self.horizon - self.receding_start,
            start_iter=self.receding_start,
        )
        best_traj = self._select_best(
            costs=self.dynamics_.cost_fn(state_trajs),
            inputs_tuple=inputs_tuple,
            state_trajs=state_trajs,
            style_override="receding",
        )
        current_controls = self.control_tensors[:, self.receding_start : self.receding_start + self.receding_horizon, :]
        return current_controls, best_traj

    def _rollout_with_sampling(self, init_state, t_start, horizon, start_iter, eps_var_factor=1):
        """Rollout with sampled perturbations around nominal controls."""
        if self.sample_mode == "gaussian":
            control_randn = torch.randn(self.batch_size, self.num_samples, horizon, self.dynamics_.control_dim).to(
                self.device
            )
            eps = (control_randn * torch.sqrt(self.dynamics_.eps_var_control).to(self.device) * eps_var_factor).to(
                self.device
            )
            # Keep nominal control
            eps[:, 0, ...] = 0.0
            end_iter = start_iter + horizon
            offset = (
                self.control_tensors[:, start_iter:end_iter, :].unsqueeze(1).repeat(1, self.num_samples, 1, 1)
            ).to(self.device)
            permuted_controls = offset + eps
            permuted_controls = self.dynamics_.bound_control(permuted_controls)
        elif self.sample_mode == "binary":
            permuted_controls = torch.sign(
                torch.empty(self.batch_size, self.num_samples, horizon, self.dynamics_.control_dim).uniform_(-1, 1)
            ).to(self.device)
            # Keep nominal control
            permuted_controls[:, 0, ...] = self.control_tensors[:, start_iter : start_iter + horizon, :] * 1.0
        else:
            raise NotImplementedError(f"Unknown sample_mode: {self.sample_mode}")

        state_trajs = torch.zeros(self.batch_size, self.num_samples, horizon + 1, self.dynamics_.state_dim).to(
            self.device
        )
        state_trajs[:, :, 0, :] = init_state.unsqueeze(1).repeat(1, self.num_samples, 1)
        state = state_trajs[:, :, 0, :]
        for k in range(horizon):
            control = permuted_controls[:, :, k, :]
            control = self.dynamics_.clamp_control(state, control)
            permuted_controls[:, :, k, :] = control
            state = self.get_next_step_state(state, control)
            state_trajs[:, :, k + 1, :] = state
        return state_trajs, (permuted_controls,)

    def _rollout_deterministic_policy(self, init_state, t_start, horizon, start_iter):
        """Deterministic rollout using the policy for control at each step."""
        state_trajs = torch.zeros(self.batch_size, horizon + 1, self.dynamics_.state_dim).to(self.device)
        controls = torch.zeros(self.batch_size, horizon, self.dynamics_.control_dim).to(self.device)
        state = init_state * 1.0
        state_trajs[:, 0, :] = state
        traj_times = torch.ones(self.batch_size, 1).to(self.device) * horizon * self.dT

        for k in range(horizon):
            traj_coords = torch.cat((traj_times, self.dynamics_.clip_state(state)), dim=-1)
            traj_policy_results = self.policy(
                {"coords": self.policy_dynamics_.coord_to_input(traj_coords.to(self.device))}
            )
            traj_dvs = self.policy_dynamics_.io_to_dv(
                traj_policy_results["model_in"], traj_policy_results["model_out"].squeeze(dim=-1)
            ).detach()

            control = self.dynamics_.optimal_control(
                traj_coords[:, 1:].to(self.device), traj_dvs[..., 1:].to(self.device)
            )
            control = self.dynamics_.clamp_control(state, control)
            state = self.get_next_step_state(state, control)

            controls[:, k, :] = control
            state_trajs[:, k + 1, :] = state
            traj_times = traj_times - self.dT

        self.set_control_tensors(controls, start_iter, start_iter + horizon)
        return state_trajs, (controls,)

    def rollout_nominal_trajs(self, init_state, t_start, horizon=None):
        """Rollout using current self.control_tensors (no sampling or policy)."""
        if horizon is None:
            horizon = self.horizon
        state_trajs = torch.zeros(self.batch_size, horizon + 1, self.dynamics_.state_dim).to(self.device)
        state = init_state * 1.0
        state_trajs[:, 0, :] = state
        for k in range(horizon):
            state = self.get_next_step_state(state, self.control_tensors[:, k, :])
            state_trajs[:, k + 1, :] = state
        return state_trajs, (self.control_tensors,)

    def _select_best(self, costs, inputs_tuple, state_trajs, style_override=None):
        """Select the best sample by cost and update self.control_tensors.

        Args:
            costs: [B, N] cost for each sample
            inputs_tuple: (controls, disturbances)
            state_trajs: [B, N, H+1, D_x]
            style_override: Override self.style for tensor update ("receding")

        Returns:
            best_traj: [B, H+1, D_x]
        """
        assert self.mode == "MPC", "Only MPC mode supported"
        if self.dynamics_.set_mode in ["avoid"]:
            best_costs, best_idx = costs.max(1)
        elif self.dynamics_.set_mode in ["reach", "reach_avoid"]:
            best_costs, best_idx = costs.min(1)
        else:
            raise NotImplementedError

        H = state_trajs.size(2) - 1
        idx_controls = best_idx[..., None, None, None].expand(-1, -1, H, self.dynamics_.control_dim)
        idx_trajs = best_idx[..., None, None, None].expand(-1, -1, H + 1, self.dynamics_.state_dim)
        best_controls = torch.gather(inputs_tuple[0], dim=1, index=idx_controls).squeeze(1)
        best_traj = torch.gather(state_trajs, dim=1, index=idx_trajs).squeeze(1)

        horizon = best_controls.shape[1]

        effective_style = style_override or self.style
        if effective_style == "direct":
            self.set_control_tensors(best_controls, 0, horizon)
        elif effective_style == "receding":
            self.set_control_tensors(best_controls, self.receding_start, self.receding_start + horizon)

        return best_traj

    def _eval_vf(self, terminal_states, t_eval):
        """Evaluate learned value function at terminal states.

        Args:
            terminal_states: [B, N, D] terminal states
            t_eval: Time at which to evaluate the VF

        Returns:
            terminal_values: [B, N] value estimates at the final state
        """
        traj_times = torch.ones(self.batch_size, self.num_samples, 1).to(self.device) * t_eval
        terminal_states_clamped = self.dynamics_.clip_state(terminal_states)
        traj_coords = torch.cat((traj_times, terminal_states_clamped), dim=-1)
        traj_policy_results = self.policy({"coords": self.policy_dynamics_.coord_to_input(traj_coords.to(self.device))})
        terminal_values = self.policy_dynamics_.io_to_value(
            traj_policy_results["model_in"].detach(),
            traj_policy_results["model_out"].squeeze(dim=-1).detach(),
        )
        return terminal_values

    def init_input_tensors(self):
        """Initialize control tensors to the dynamics' default control init."""
        self.receding_start = 0
        self.control_tensors = (
            self.dynamics_.control_init.unsqueeze(0)
            .repeat(self.batch_size, 1)
            .unsqueeze(1)
            .repeat(1, self.horizon, 1)
            .to(self.device)
        )

    def get_next_step_state(self, state, controls):
        """Single-step Euler integration (control-only, no disturbance)."""
        current_dsdt = self.dynamics_.dsdt(state, controls, None)
        next_states = self.dynamics_.equivalent_wrapped_state(state + current_dsdt * self.dT)
        return next_states


class RobustMPC(MPC):
    """Robust MPC with adversarial disturbance optimization.

    Extends MPC to handle both control and disturbance sampling/optimization.
    One of control_sample_mode/disturbance_sample_mode must be "policy".
    """

    def __init__(
        self,
        dT,
        horizon,
        receding_horizon,
        num_samples,
        dynamics_,
        device,
        mode="MPC",
        control_sample_mode="gaussian",
        disturbance_sample_mode="policy",
        lambda_=0.01,
        style="direct",
        num_warm_start_iters=0,
        num_full_horizon_iters=0,
        # Legacy compat: if provided, overrides the two above
        num_iterative_refinement=None,
        warm_start_only=False,
    ):
        # Basic timing & dynamics
        self.dT = dT
        self.horizon = horizon
        self.receding_horizon = receding_horizon
        self.num_samples = num_samples
        self.device = device
        self.dynamics_ = dynamics_
        self.policy_dynamics_ = dynamics_

        # Modes
        self.mode = mode
        self.control_sample_mode = control_sample_mode
        self.disturbance_sample_mode = disturbance_sample_mode
        assert control_sample_mode == "policy" or disturbance_sample_mode == "policy", (
            "At least one of control/disturbance must use policy mode"
        )

        self.lambda_ = lambda_

        # Planning style
        self.style = style

        # Resolve iteration counts (legacy compat)
        if num_warm_start_iters == 0 and num_full_horizon_iters == 0:
            warnings.warn(
                "num_iterative_refinement is deprecated. Use num_warm_start_iters and num_full_horizon_iters instead."
            )
            if num_iterative_refinement == -1:
                num_warm_start_iters = -1
                num_full_horizon_iters = 0
            elif warm_start_only:
                # warm_start_only: all iters are warm-start, skip full-horizon
                num_warm_start_iters = num_iterative_refinement
                num_full_horizon_iters = 0
            else:
                # Original: warm takes int(n*0.4) iters; full takes the remainder.
                # Total = n+1 steps (matches MPC.py's get_control_and_disturbance: range(n+1-warm)).
                num_warm_start_iters = int(num_iterative_refinement * 0.4)
                num_full_horizon_iters = (num_iterative_refinement + 1) - num_warm_start_iters
        self.num_warm_start_iters = num_warm_start_iters
        self.num_full_horizon_iters = num_full_horizon_iters

    def set_disturbance_tensors(self, disturbances, start_iter, end_iter):
        self.disturbance_tensors[:, start_iter:end_iter, :] = disturbances

    def _get_opt_trajs_receding(self, init_state):
        raise NotImplementedError("Receding horizon not implemented for RobustMPC")

    def _rollout_with_sampling(self, init_state, t_start, horizon, start_iter, eps_var_factor=1):
        """Rollout with sampling on the non-policy input."""
        assert self.control_sample_mode == "policy" or self.disturbance_sample_mode == "policy"

        controls = torch.zeros(self.batch_size, self.num_samples, horizon, self.dynamics_.control_dim).to(self.device)
        disturbances = torch.zeros(self.batch_size, self.num_samples, horizon, self.dynamics_.disturbance_dim).to(
            self.device
        )

        # Sample controls if in sample mode
        if self.control_sample_mode in ("gaussian", "sample"):
            control_randn = torch.randn(self.batch_size, self.num_samples, horizon, self.dynamics_.control_dim).to(
                self.device
            )
            eps = control_randn * torch.sqrt(self.dynamics_.eps_var_control.to(self.device)) * eps_var_factor
            # Keep nominal control
            eps[:, 0, ...] = 0.0
            end_iter = start_iter + horizon
            offset = (
                self.control_tensors[:, start_iter:end_iter, :]
                .to(self.device)
                .unsqueeze(1)
                .repeat(1, self.num_samples, 1, 1)
            )
            controls = self.dynamics_.bound_control(offset + eps)

        # Sample disturbances if in sample mode
        if self.disturbance_sample_mode in ("gaussian", "sample"):
            dist_randn = torch.randn(self.batch_size, self.num_samples, horizon, self.dynamics_.disturbance_dim).to(
                self.device
            )
            eps = dist_randn * torch.sqrt(self.dynamics_.eps_var_disturbance.to(self.device)) * eps_var_factor
            # Keep nominal disturbance
            eps[:, 0, ...] = 0.0
            end_iter = start_iter + horizon
            offset = (
                self.disturbance_tensors[:, start_iter:end_iter, :]
                .to(self.device)
                .unsqueeze(1)
                .repeat(1, self.num_samples, 1, 1)
            )
            disturbances = self.dynamics_.bound_disturbance(offset + eps)

        # Forward simulate
        state_trajs = torch.zeros(self.batch_size, self.num_samples, horizon + 1, self.dynamics_.state_dim).to(
            self.device
        )
        state_trajs[:, :, 0, :] = init_state.unsqueeze(1).repeat(1, self.num_samples, 1)

        policy_traj_times = torch.ones(self.batch_size, self.num_samples, 1).to(self.device) * t_start

        state = state_trajs[:, :, 0, :]
        for k in range(horizon):
            # Query policy for the "policy" side
            t_eval = torch.clamp(policy_traj_times, max=self.t)  # Policy is only valid up to time self.t (curr time)
            traj_coords = torch.cat((t_eval, self.dynamics_.clip_state(state)), dim=-1)
            traj_policy_results = self.policy(
                {"coords": self.policy_dynamics_.coord_to_input(traj_coords.to(self.device))}
            )
            traj_dvs = self.policy_dynamics_.io_to_dv(
                traj_policy_results["model_in"],
                traj_policy_results["model_out"].squeeze(dim=-1),
            ).detach()

            if self.disturbance_sample_mode == "policy":
                dist = self.policy_dynamics_.optimal_disturbance(
                    traj_coords[..., 1:].to(self.device), traj_dvs[..., 1:].to(self.device)
                )
            elif self.disturbance_sample_mode in ("gaussian", "sample"):
                dist = disturbances[:, :, k, :]
            if self.control_sample_mode == "policy":
                control = self.policy_dynamics_.optimal_control(
                    traj_coords[..., 1:].to(self.device), traj_dvs[..., 1:].to(self.device)
                )
            elif self.control_sample_mode in ("gaussian", "sample"):
                control = controls[:, :, k, :]

            control = self.dynamics_.clamp_control(state, control)
            dist = self.dynamics_.clamp_disturbance(state, dist)

            state = self.get_next_step_state(state, control, dist)
            state_trajs[:, :, k + 1, :] = state
            controls[:, :, k, :] = control
            disturbances[:, :, k, :] = dist
            policy_traj_times = policy_traj_times - self.dT

        return state_trajs, (controls, disturbances)

    def _rollout_deterministic_policy(self, init_state, t_start, horizon, start_iter):
        """Deterministic rollout using policy for both control and disturbance."""
        state_trajs = torch.zeros(self.batch_size, horizon + 1, self.dynamics_.state_dim).to(self.device)
        controls = torch.zeros(self.batch_size, horizon, self.dynamics_.control_dim).to(self.device)
        disturbances = torch.zeros(self.batch_size, horizon, self.dynamics_.disturbance_dim).to(self.device)
        state = init_state * 1.0
        state_trajs[:, 0, :] = state
        traj_times = torch.ones(self.batch_size, 1).to(self.device) * horizon * self.dT

        for k in range(horizon):
            traj_coords = torch.cat((traj_times, self.dynamics_.clip_state(state)), dim=-1)
            traj_policy_results = self.policy(
                {"coords": self.policy_dynamics_.coord_to_input(traj_coords.to(self.device))}
            )
            traj_dvs = self.policy_dynamics_.io_to_dv(
                traj_policy_results["model_in"],
                traj_policy_results["model_out"].squeeze(dim=-1),
            ).detach()

            control = self.policy_dynamics_.optimal_control(
                traj_coords[:, 1:].to(self.device), traj_dvs[..., 1:].to(self.device)
            )
            control = self.policy_dynamics_.clamp_control(state, control)

            dist = self.policy_dynamics_.optimal_disturbance(
                traj_coords[:, 1:].to(self.device), traj_dvs[..., 1:].to(self.device)
            )
            dist = self.policy_dynamics_.clamp_disturbance(state, dist)

            state = self.get_next_step_state(state, control, dist)

            controls[:, k, :] = control
            disturbances[:, k, :] = dist
            state_trajs[:, k + 1, :] = state
            traj_times = traj_times - self.dT

        self.set_control_tensors(controls, start_iter, start_iter + horizon)
        self.set_disturbance_tensors(disturbances, start_iter, start_iter + horizon)

        return state_trajs, (controls, disturbances)

    def rollout_nominal_trajs(self, init_state, t_start, horizon=None):
        """Rollout using current control/disturbance tensors (no sampling, no policy)."""
        if horizon is None:
            horizon = self.horizon
        state_trajs = torch.zeros(self.batch_size, horizon + 1, self.dynamics_.state_dim).to(self.device)
        controls = self.control_tensors[:, :horizon, :]
        disturbances = self.disturbance_tensors[:, :horizon, :]
        state = init_state * 1.0
        state_trajs[:, 0, :] = state
        for k in range(horizon):
            state = self.get_next_step_state(state, controls[:, k, :], disturbances[:, k, :])
            state_trajs[:, k + 1, :] = state

        return state_trajs, (controls, disturbances)

    def _select_best(self, costs, inputs_tuple, state_trajs):
        """Select the best sample by cost and update internal tensors.

        Args:
            costs: [B, N] cost for each sample
            controls: [B, N, H, D_u]
            disturbances: [B, N, H, D_d]
            state_trajs: [B, N, H+1, D_x]

        Returns:
            best_traj: [B, H+1, D_x] trajectory of the best sample
        """
        assert self.mode == "MPC", "Only MPC mode supported"

        # Determine best index based on set_mode and which side is policy
        if self.dynamics_.set_mode in ["avoid"]:
            if self.disturbance_sample_mode == "policy":
                best_costs, best_idx = costs.max(1)
            else:
                best_costs, best_idx = costs.min(1)
        else:  # reach or reach_avoid
            if self.disturbance_sample_mode == "policy":
                best_costs, best_idx = costs.min(1)
            else:
                best_costs, best_idx = costs.max(1)

        H = state_trajs.size(2) - 1
        idx_controls = best_idx[..., None, None, None].expand(-1, -1, H, self.dynamics_.control_dim)
        idx_disturbances = best_idx[..., None, None, None].expand(-1, -1, H, self.dynamics_.disturbance_dim)
        idx_trajs = best_idx[..., None, None, None].expand(-1, -1, H + 1, self.dynamics_.state_dim)

        best_controls = torch.gather(inputs_tuple[0], dim=1, index=idx_controls).squeeze(1)
        best_disturbances = torch.gather(inputs_tuple[1], dim=1, index=idx_disturbances).squeeze(1)
        best_traj = torch.gather(state_trajs, dim=1, index=idx_trajs).squeeze(1)

        # Update tensors
        horizon = best_controls.shape[1]
        self.set_control_tensors(best_controls, 0, horizon)
        self.set_disturbance_tensors(best_disturbances, 0, horizon)

        return best_traj

    def init_input_tensors(self):
        """Initialize both control and disturbance tensors."""
        super().init_input_tensors()
        self.disturbance_tensors = (
            self.dynamics_.disturbance_init.unsqueeze(0)
            .repeat(self.batch_size, 1)
            .unsqueeze(1)
            .repeat(1, self.horizon, 1)
            .to(self.device)
        )

    def get_next_step_state(self, state, control, disturbance):
        """State integration with choice of method (Euler or RK4)."""
        current_dsdt = self.dynamics_.dsdt(state, control, disturbance)
        next_state = self.dynamics_.equivalent_wrapped_state(state + current_dsdt * self.dT)
        return next_state
