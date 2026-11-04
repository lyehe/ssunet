"""Analytic and support regressions for classical photon probability estimates."""

import pytest
import torch

from src.ssunet.neutrino_classical import (
    fit_population_prior,
    gaussian_transfer,
    kernel_counts,
    prior_shrinkage,
)
from src.ssunet.poisson_tv import poisson_prox, poisson_tv, tv_adjoint, tv_gradient


def test_gaussian_columns_conserve_boundary_mass_and_exclude_dead_detectors():
    """A source photon stays one photon even at a truncated kernel boundary."""
    times = torch.arange(5, dtype=torch.float64)
    squared = (times[:, None] - times[None]).square()
    kernel = gaussian_transfer(squared, 0.8)
    torch.testing.assert_close(kernel.sum(0), torch.ones(5, dtype=torch.float64))
    live = torch.tensor([True, False, True, True, False])
    physical = gaussian_transfer(squared, 0.8, live)
    torch.testing.assert_close(physical.sum(0), live.double())
    assert torch.count_nonzero(physical[~live]) == 0


@pytest.mark.parametrize("adaptive", [False, True])
def test_kde_keeps_photon_mass_and_physical_sensor_permutation(adaptive):
    """Neither boundary smoothing nor input-dependent bandwidth creates or drops counts."""
    counts = torch.arange(30).reshape(2, 3, 5) % 4
    positions = torch.tensor([[0, 0, 0], [0, 0, 20], [10, 0, 0]])
    times = torch.arange(5) * 25
    live = torch.tensor([True, False, True])
    options = {"time_bandwidth_ns": 25, "space_bandwidth_m": 20, "adaptive": adaptive}
    actual = kernel_counts(counts, positions, times, live, **options)
    torch.testing.assert_close(actual.sum((1, 2)), counts[:, live].sum((1, 2)).double())
    assert torch.count_nonzero(actual[:, ~live]) == 0
    order = torch.tensor([2, 0, 1])
    permuted = kernel_counts(counts[:, order], positions[order], times, live[order], **options)
    torch.testing.assert_close(permuted, actual[:, order])


def test_dirichlet_prediction_is_analytic_and_empty_input_returns_the_prior():
    """The raw-count estimator exactly matches the conjugate posterior predictive."""
    counts = torch.tensor([[[4, 0], [0, 0]], [[0, 0], [0, 0]]])
    live = torch.tensor([True, False])
    prior = torch.tensor([[0.25, 0.75], [0.0, 0.0]], dtype=torch.float64)
    q = prior_shrinkage(counts, counts, prior, live, 2)
    torch.testing.assert_close(q[0], (counts[0] + 2 * prior) / 6)
    torch.testing.assert_close(q[1], prior)
    torch.testing.assert_close(q.sum((1, 2)), torch.ones(2, dtype=torch.float64))
    assert torch.count_nonzero(q[:, ~live]) == 0


def test_tv_shape_rescaling_preserves_the_observed_count_concentration():
    """Shrunk TV intensity is normalized separately from the posterior's A count total."""
    counts = torch.tensor([[[4, 0]]])
    density = torch.tensor([[[0.25, 0.25]]])
    prior = torch.tensor([[0.8, 0.2]])
    q = prior_shrinkage(density, counts, prior, torch.tensor([True]), 2)
    torch.testing.assert_close(q[0, 0], (torch.tensor([2.0, 2.0]) + 2 * prior[0]).double() / 6)
    with pytest.raises(ValueError, match="nonempty estimated"):
        prior_shrinkage(density * 0, counts, prior, torch.tensor([True]), 2)


def test_population_prior_is_positive_only_on_actual_support():
    """Training-only prior smoothing supplies finite support for unseen arrival locations."""
    counts = torch.zeros(2, 3, 5, dtype=torch.int64)
    counts[0, 0, 0] = 10
    live = torch.tensor([True, False, True])
    prior = fit_population_prior(
        counts,
        torch.zeros(3, 3),
        torch.arange(5) * 25,
        live,
        time_bandwidth_ns=50,
    )
    assert torch.all(prior[live] > 0)
    assert torch.count_nonzero(prior[~live]) == 0
    torch.testing.assert_close(prior.sum(), torch.tensor(1.0, dtype=torch.float64))


def test_poisson_prox_satisfies_positive_count_stationarity_and_zero_count_threshold():
    """Verify the closed-form Poisson proximal operator against its scalar KKT equation."""
    values = torch.tensor([-1000.0, -2.0, 20.0, 2.0], dtype=torch.float64)
    observed = torch.tensor([1.0, 10.0, 0.0, 0.0], dtype=torch.float64)
    step = 0.3
    result = poisson_prox(values, observed, step)
    residual = result[:2] - values[:2] + step * (1 - observed[:2] / result[:2])
    torch.testing.assert_close(residual, torch.zeros(2, dtype=torch.float64), atol=1e-10, rtol=0)
    torch.testing.assert_close(result[2:], (values[2:] - step).clamp_min(0))


def test_tv_operator_and_adjoint_have_the_same_inner_product():
    """The primal-dual solver uses a true adjoint, including disabled physical edges."""
    torch.manual_seed(31)
    values = torch.randn(2, 4, 5, dtype=torch.float64)
    weights = torch.tensor([1.0, 0.0, 0.3], dtype=torch.float64)
    dt, dz = tv_gradient(values, weights)
    p, q = torch.randn_like(dt), torch.randn_like(dz)
    expected = (dt * p).sum() + (dz * q).sum()
    actual = (values * tv_adjoint(p, q, weights, values.shape)).sum()
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(("regularization", "expected"), [(0.25, [3.2, 0.0]), (2.0, [2.0, 2.0])])
def test_poisson_tv_matches_the_analytic_two_bin_solution(regularization, expected):
    """Two-bin Poisson-TV has known solutions on both sides of its fusion threshold."""
    values, diagnostics = poisson_tv(
        torch.tensor([[[4, 0]]]),
        torch.zeros(1, 3),
        torch.tensor([True]),
        regularization=regularization,
        max_iterations=2000,
        tolerance=1e-12,
        check_interval=10,
    )
    torch.testing.assert_close(
        values.flatten(), torch.tensor(expected, dtype=torch.float64), atol=1e-6, rtol=1e-6
    )
    assert diagnostics["converged"]
    assert diagnostics["maximum_gap_per_photon"] <= 1e-12


def test_poisson_tv_constant_and_empty_observations_are_fixed_points():
    """Zero and constant count fields have zero TV and retain valid nonnegative estimates."""
    counts = torch.full((2, 3, 5), 3, dtype=torch.int64)
    counts[0] = 0
    positions = torch.tensor([[0, 0, 0], [0, 0, 10], [0, 0, 20]])
    result, diagnostics = poisson_tv(
        counts, positions, torch.ones(3, dtype=torch.bool), regularization=1, vertical_strength=0.5
    )
    torch.testing.assert_close(result, counts.double())
    assert diagnostics["converged"]


def test_event_weighted_prior_does_not_let_one_bright_event_dominate():
    """A 100-photon event and a one-photon event each supply one shape to an event prior."""
    counts = torch.tensor([[[100, 0]], [[0, 1]]])
    args = torch.zeros(1, 3), torch.tensor([0, 25]), torch.tensor([True])
    prior = fit_population_prior(counts, *args, time_bandwidth_ns=0, weighting="event")
    torch.testing.assert_close(prior, torch.full((1, 2), 0.5, dtype=torch.float64))
    photon = fit_population_prior(counts, *args, time_bandwidth_ns=0, weighting="photon")
    assert photon[0, 0] > 0.99


def test_inner_training_split_never_shares_configuration_groups():
    """Repeated observations of one source cannot leak from prior fitting into parameter tuning."""
    import numpy as np

    from src.tools.neutrino_classical import inner_group_split

    groups = np.array([1, 1, 2, 3, 3, 4, 5, 6])
    fit, tune = inner_group_split(groups, 0.25, 42)
    assert set(groups[fit]).isdisjoint(set(groups[tune]))
    assert sorted(np.concatenate([fit, tune]).tolist()) == list(range(len(groups)))
    again_fit, again_tune = inner_group_split(groups, 0.25, 42)
    np.testing.assert_array_equal(fit, again_fit)
    np.testing.assert_array_equal(tune, again_tune)
