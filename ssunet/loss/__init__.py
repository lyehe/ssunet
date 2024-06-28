from .loss import l1_loss, mse_loss, photon_loss, photon_loss_2D

__all__ = ["l1_loss", "mse_loss", "photon_loss", "photon_loss_2D", "loss_functions"]

loss_functions = {
    "l1_loss": l1_loss,
    "mse_loss": mse_loss,
    "photon_loss": photon_loss,
    "photon_loss_2D": photon_loss_2D,
}
