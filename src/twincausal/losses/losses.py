import torch


# Complete log-loss function
def logistic_loss(m11, m10, t, y):
    m11 = torch.clamp(m11, 0.01, 0.99)
    m10 = torch.clamp(m10, 0.01, 0.99)
    m1t = t * m11 + (1 - t) * m10
    loss = y * torch.log(m1t) + (1 - y) * torch.log(1 - m1t)

    return (-1) * torch.mean(loss)


# Complete uplif-loss function (pseudo-likelihood)
def uplift_loss(m11, m10, t, y):
    m11 = torch.clamp(m11, 0.01, 0.99)
    m10 = torch.clamp(m10, 0.01, 0.99)
    m1t = t * m11 + (1 - t) * m10
    py1 = y * m11 / (m11 + m10) + (1.0 - y) * (1.0 - m11) / ((1.0 - m11) + (1.0 - m10))
    loss = t * torch.log(py1) + (1 - t) * torch.log(1 - py1) + y * torch.log(m1t) + (1 - y) * torch.log(1 - m1t)

    return (-1) * torch.mean(loss)


# Complete twin-network loss function when interpreted as the likelihood P(T,Y|X)
def likelihood_version_loss(m11, m10, t, y):
    m11 = torch.clamp(m11, 0.01, 0.99)
    m10 = torch.clamp(m10, 0.01, 0.99)
    py1 = y * m11 / (m11 + m10) + (1.0 - y) * (1.0 - m11) / ((1.0 - m11) + (1.0 - m10))
    loss = t * torch.log(py1) + (1 - t) * torch.log(1 - py1) + y * torch.log(m11 + m10) + (1 - y) * torch.log(2 - m11 - m10)

    return (-1) * torch.mean(loss)


# From ICML Paper / Not used here
# Our twin-network second part of the loss function
def indirect_loss(p1, p0, treatment, y):
    p1 = torch.clamp(p1, 0.01, 0.99)
    p0 = torch.clamp(p0, 0.01, 0.99)
    prob_treatment_one = y * p1 / (p1 + p0) + (1.0 - y) * (1.0 - p1) / ((1.0 - p1) + (1.0 - p0))
    loss = torch.nn.BCELoss()

    return loss(prob_treatment_one, treatment)


# Exponential loss for TO
def smooth_sign_loss(output, target):
    x = output * target
    smooth_sign = torch.exp(-x)

    return torch.mean(smooth_sign)
