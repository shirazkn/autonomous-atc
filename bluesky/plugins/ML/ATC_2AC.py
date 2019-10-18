import torch.nn as nn


class ATC_Net(nn.Module):
    def __init__(self):
        super(ATC_Net, self).__init__()

        # Network input & output layers
        self.fc1 = nn.Linear(3, 5)
        self.fc2 = nn.Linear(5, 5)
        self.fc3 = nn.Linear(5, 1)

    def forward(self, x):
        x = self.fc3(self.fc2(self.fc1(x)))
        return x


# Calculates loss incurred after a conf. resolution step
def accumulate_loss(next_asas):
    """
    :param next_asas: <asas> state of the following time-step (to check whether conflict was resolved)
    :return: float
    """
    if next_asas.confpairs:
        return 1.0

    return 0.0


def cost_of_resolution(atc_output, running_loss):
    """
        # Assigns penalty to undesirably large conf. resolution steps
        :param atc_output: <torch.tensor>
        :return: <tensor(positive <float>)>
        """
    return atc_output**2 + running_loss
