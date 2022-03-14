from torch import nn
from models.quantization import quan_Conv2d, quan_Linear


def piecewise_clustering(var, lambda_coeff, l_norm):
    var1 = (var[var.ge(0)] - var[var.ge(0)].mean()).pow(l_norm).sum()
    var2 = (var[var.le(0)] - var[var.le(0)].mean()).pow(l_norm).sum()
    return lambda_coeff * (var1 + var2)


def clustering_loss(model, lambda_coeff, l_norm=2):
    pc_loss = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or \
                isinstance(m, quan_Conv2d) or isinstance(m, quan_Linear):
            pc_loss += piecewise_clustering(m.weight, lambda_coeff, l_norm)
    return pc_loss
