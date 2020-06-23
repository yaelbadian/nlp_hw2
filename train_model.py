from torch.nn import LogSoftmax
import torch.optim as optim

acumulate_grad_steps = 50
optimizer = optim.Adam(model.parameters(), lr=0.01)


def loss_func(scores, target, nllloss):
    # scores.shape: [batch_size, seq_length, seq_length]
    # target.shape: [batch_size, seq_length]
    m = LogSoftmax(dim=1)
    output = nllloss(m(scores), target)
    output = output / acumulate_grad_steps
    return output