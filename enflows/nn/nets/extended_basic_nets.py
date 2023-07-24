import torch
import torch.nn as nn

class ExtendedSequential(nn.Sequential):
    def build_clone(self):
        modules = []
        for m in self:
            modules.append(m.build_clone())
        return ExtendedSequential(*modules)

    def build_jvp_net(self, *args):
        with torch.no_grad():
            modules = []
            y = args
            for m in self:
                jvp_net_and_y = m.build_jvp_net(*y)
                jvp_net = jvp_net_and_y[0]
                y = jvp_net_and_y[1:]
                modules.append(jvp_net)
            return ExtendedSequential(*modules), *y


class ExtendedLinear(nn.Linear):
    def build_clone(self):
        with torch.no_grad():
            weight = self.weight.detach().requires_grad_(False)
            # weight = self.compute_weight(update=False).detach().requires_grad_(False)
            if self.bias is not None:
                bias = self.bias.detach().requires_grad_(False)
            m = nn.Linear(self.in_features, self.out_features, bias=self.bias is not None, device=self.weight.device)
            m.weight.data.copy_(weight)
            if self.bias is not None:
                m.bias.data.copy_(bias)
            return m

    def build_jvp_net(self, x):
        '''
        Bias is omitted in contrast to self.build_clone().
        '''
        with torch.no_grad():
            # weight = self.compute_weight(update=False).detach().requires_grad_(False)
            weight = self.weight.detach().requires_grad_(False)
            m = nn.Linear(self.in_features, self.out_features, bias=None, device=self.weight.device)
            m.weight.data.copy_(weight)
            return m, self.forward(x).detach().clone()