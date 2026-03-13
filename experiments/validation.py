"""Training validation utilities.

Extracted from experiments.py to keep validation/scenario optimization
logic separate from the training loop.
"""

import torch
import wandb
from utils.error_evaluators import (
    scenario_optimization,
    ValueThresholdValidator,
    SliceSampleGenerator,
    FixedStateSampleGenerator,
)


def _generate_validated_states(dynamics, num_states=10000, num_candidates=50000, v_min=0.0, v_max=2.0, seed=100):
    """Generate states that pass boundary value validation.

    Samples uniformly from the state space and filters to states whose
    boundary function value lies in [v_min, v_max].

    Args:
        dynamics: Dynamics object with boundary_fn and state_dim.
        num_states: Target number of validated states.
        num_candidates: Number of candidates to sample per iteration.
        v_min: Minimum boundary value for validation.
        v_max: Maximum boundary value for validation.

    Returns:
        Tensor of shape (num_states, state_dim) with validated states.
    """
    validator = ValueThresholdValidator(v_min=float(v_min), v_max=float(v_max))
    generator = SliceSampleGenerator(
        dynamics=dynamics,
        slices=[None] * dynamics.state_dim,
        generator=seed,
    )
    states = []
    while len(states) < num_states:
        candidates = generator.sample(num_candidates)
        boundary_values = dynamics.boundary_fn(candidates)
        valid_mask = validator.validate(candidates, boundary_values)
        states.extend(candidates[valid_mask])
        states = states[:num_states]
    return torch.stack(states)


def run_training_validation(model, dynamics, tMin, tMax, current_time, use_wandb, epoch, cached_states=None, seed=100):
    """Run scenario optimization at multiple time horizons during training.

    Evaluates the learned value function at 25%, 50%, 75%, 100% of tMax
    and at 5 seconds, logging violation statistics.

    Args:
        model: The DeepReach model.
        dynamics: Dynamics object.
        tMin: Minimum time.
        tMax: Maximum time.
        current_time: Current curriculum time (how far in time we've trained).
        use_wandb: Whether to log to wandb.
        epoch: Current training epoch (for wandb step).
        cached_states: Dict with pre-generated states from a previous call,
            or None to generate fresh states. Keys: 'states_25p', 'states_50p',
            'states_75p', 'states_100p', 'states_5s'.

    Returns:
        Dict of cached states (pass back on next call to avoid regenerating).
    """
    # Generate or reuse validated states
    if cached_states is None:
        cached_states = {
            "states_25p": _generate_validated_states(dynamics, seed=seed),
            "states_50p": _generate_validated_states(dynamics, seed=seed),
            "states_75p": _generate_validated_states(dynamics, seed=seed),
            "states_100p": _generate_validated_states(dynamics, seed=seed),
            "states_5s": _generate_validated_states(dynamics, seed=seed),
        }

    # Create fixed-state generators from cached states
    generators = {key: FixedStateSampleGenerator(dynamics=dynamics, states=cached_states[key]) for key in cached_states}

    # Compute time horizons (aligned to dt)
    dt = 0.02
    time_configs = {
        "stats25%": (int(0.25 * tMax / dt) * dt, generators["states_25p"]),
        "stats50%": (int(0.5 * tMax / dt) * dt, generators["states_50p"]),
        "stats75%": (int(0.75 * tMax / dt) * dt, generators["states_75p"]),
        "stats100%": (int(tMax / dt) * dt, generators["states_100p"]),
        "stats5s": (int(5.0 / dt) * dt, generators["states_5s"]),
    }

    stats = {}
    for label, (t_horizon, sample_generator) in time_configs.items():
        stats[label] = scenario_optimization(
            model=model,
            dynamics=dynamics,
            tMin=tMin,
            tMax=t_horizon,
            dt=dt,
            set_type="BRT",
            control_type="value",
            scenario_batch_size=10000,
            sample_batch_size=10000,
            sample_generator=sample_generator,
            sample_validator=ValueThresholdValidator(v_min=float("-inf"), v_max=float("inf")),
            max_scenarios=10000,
            vf_eval_time=min(t_horizon, current_time),
            statistics_only=True,
        )

    # Flatten nested stats dict and log
    flattened_stats = _flatten_dict(stats)
    if use_wandb:
        wandb.log(flattened_stats, step=epoch)

    return cached_states


def _flatten_dict(d, parent_key="", sep="/"):
    """Flatten a nested dictionary with '/' separators."""
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(_flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items
