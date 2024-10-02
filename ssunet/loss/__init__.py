"""Loss module."""

from .loss import l1_loss, mse_loss, photon_loss, photon_loss_2d

__all__ = ["l1_loss", "mse_loss", "photon_loss", "photon_loss_2d", "loss_functions"]

loss_functions = {
    "l1": l1_loss,
    "mse": mse_loss,
    "photon": photon_loss,
    "photon_2d": photon_loss_2d,
}
