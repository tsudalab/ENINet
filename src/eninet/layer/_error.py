import torch
from torch.nn import Module


class TensorPerAtomRMSE(Module):
    """Define RMSE Loss following the work:
    Wilkins, David M., et al. "Accurate molecular polarizabilities with coupled cluster theory and machine learning."
    Proceedings of the National Academy of Sciences 116.9 (2019): 3401-3406.
    """

    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, preds, labels, n_atoms):
        batch_size = preds.size(0)
        preds = preds.view(batch_size, -1)
        labels = labels.view(batch_size, -1)

        mse = torch.mean(torch.sum((preds - labels) ** 2, dim=1) / (n_atoms**2))

        return mse**0.5
