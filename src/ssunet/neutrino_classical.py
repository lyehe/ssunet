"""Count-aware classical photon-location estimates on actual detector support."""

import torch


def gaussian_transfer(squared_distance, bandwidth, live=None):
    """Normalize each source column so boundary photons retain unit probability mass."""
    bandwidth = torch.as_tensor(
        bandwidth, dtype=squared_distance.dtype, device=squared_distance.device
    )
    if torch.any(bandwidth <= 0):
        message = "Gaussian transfer bandwidth must be positive"
        raise ValueError(message)
    energy = -0.5 * squared_distance / bandwidth[..., None, None].square()
    if live is not None:
        if live.dtype != torch.bool or not live.any():
            message = "Gaussian transfer requires nonempty boolean detector support"
            raise ValueError(message)
        energy = energy.masked_fill(~live[:, None], -torch.inf)
    transfer = torch.softmax(energy, dim=-2)
    return transfer.masked_fill(~live[None, :], 0) if live is not None else transfer


def kernel_counts(
    counts, positions_m, times_ns, live, *, time_bandwidth_ns, space_bandwidth_m=0, adaptive=False
):
    """Smooth raw A counts with a mass-preserving temporal or 4D Gaussian kernel.

    Optional sample-size scaling uses N^(-1/(d+4)), with d=1 for time-only
    smoothing or d=4 for joint space/time smoothing. Base bandwidths are
    selected on inner training splits; they are not covariance estimates.
    """
    values = counts.double().masked_fill(~live[None, :, None], 0)
    if time_bandwidth_ns < 0 or space_bandwidth_m < 0:
        message = "Kernel bandwidths must be nonnegative"
        raise ValueError(message)
    dimensions = 4 if space_bandwidth_m > 0 else 1
    factor = (
        values.sum((1, 2)).clamp_min(1).pow(-1 / (dimensions + 4))
        if adaptive
        else values.new_ones(1)
    )
    result = values
    if space_bandwidth_m > 0:
        coordinates = positions_m.double()
        squared = (coordinates[:, None] - coordinates[None]).square().sum(-1)
        spatial = gaussian_transfer(squared, factor * space_bandwidth_m, live)
        result = spatial @ result
    if time_bandwidth_ns > 0:
        times = times_ns.double()
        squared = (times[:, None] - times[None]).square()
        temporal = gaussian_transfer(squared, factor * time_bandwidth_ns)
        result = result @ temporal.transpose(-1, -2)
    return result


def fit_population_prior(
    counts, positions_m, times_ns, live, *, time_bandwidth_ns, pseudocount=1e-4, weighting="photon"
):
    """Fit a smoothed population PMF from fit-partition observations only."""
    if pseudocount <= 0:
        message = "A positive prior pseudocount is needed on every live location"
        raise ValueError(message)
    values = counts.double().masked_fill(~live[None, :, None], 0)
    if weighting == "event":
        values = values / values.sum((1, 2), keepdim=True).clamp_min(1)
    elif weighting != "photon":
        message = "Prior weighting must be photon or event"
        raise ValueError(message)
    pooled = values.sum(0, keepdim=True)
    smooth = kernel_counts(
        pooled,
        positions_m,
        times_ns,
        live,
        time_bandwidth_ns=time_bandwidth_ns,
    )[0]
    support = live[:, None].expand_as(smooth)
    result = smooth + pseudocount * support
    result = result.masked_fill(~support, 0)
    return result / result.sum()


def prior_shrinkage(density, counts, prior, live, concentration):
    """Turn a shape estimate into a positive, count-adaptive predictive PMF.

    For density=counts this is the Dirichlet-multinomial posterior predictive
    (A + alpha*pi)/(N_A + alpha). Smoothed or TV estimates use the same
    shrinkage rule after restoring their shape mass to N_A. Empty A uses pi.
    """
    if concentration <= 0:
        message = "Prior concentration must be positive"
        raise ValueError(message)
    support = live[None, :, None]
    values = density.double().masked_fill(~support, 0)
    if not torch.isfinite(values).all() or (values < 0).any():
        message = "Classical density estimates must be finite and nonnegative"
        raise ValueError(message)
    totals = counts.double().masked_fill(~support, 0).sum((1, 2), keepdim=True)
    mass = values.sum((1, 2), keepdim=True)
    if ((mass == 0) & (totals > 0)).any():
        message = "A nonempty observation needs a nonempty estimated density shape"
        raise ValueError(message)
    shape = values / mass.clamp_min(torch.finfo(values.dtype).tiny)
    population = prior.double().masked_fill(~live[:, None], 0)
    if not torch.isfinite(population).all() or (population < 0).any() or population.sum() <= 0:
        message = "Population prior must have finite nonnegative probability mass"
        raise ValueError(message)
    population = population / population.sum()
    result = (totals * shape + concentration * population) / (totals + concentration)
    return result.masked_fill(~support, 0)
