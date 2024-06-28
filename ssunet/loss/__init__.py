from .loss import l1_loss, mse_loss, photon_loss, photon_loss_2D

__all__ = ["l1_loss", "mse_loss", "photon_loss", "photon_loss_2D", "loss_functions"]

loss_functions = {
    "l1": l1_loss,
    "mse": mse_loss,
    "photon": photon_loss,
    "photon_2D": photon_loss_2D,
}
