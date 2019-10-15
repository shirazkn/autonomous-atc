import torch.nn as nn


class ATC_Net(nn.Module):
    def __init__(self):
        super(ATC_Net, self).__init__()

        # Network input & output layers
        self.fc1 = nn.Linear(3, 5)
        self.fc2 = nn.Linear(5, 5)
        self.fc3 = nn.Linear(5, 2)

    def forward(self, x):
        x = self.fc3(self.fc2(self.fc1(x)))
        return x


# Calculates loss incurred after a conf. resolution step
def get_loss(atc_output, next_asas):
    """
    :param atc_output: <torch.tensor> Action suggested by net
    :param next_asas: <asas> state of the following time-step (to check whether conflict was resolved)
    :return: <torch.tensor(<float>)>
    """
    # TODO
    loss = cost_of_resolution(atc_output)
    if next_asas.confpairs:
        loss += 100.0
    return


def cost_of_resolution(atc_output):
    """
        # Assigns penalty to undesirably large conf. resolution steps
        :param atc_output: <torch.tensor>
        :return: <tensor(positive <float>)>
        """
    # TODO
    return 0.0
