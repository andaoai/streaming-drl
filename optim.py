import torch

class ObGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1.0, gamma=0.99, lamda=0.8, kappa=2.0):
        # 初始化优化器，设置学习率、衰减因子等超参数
        defaults = dict(lr=lr, gamma=gamma, lamda=lamda, kappa=kappa)
        super(ObGD, self).__init__(params, defaults)  # 调用父类构造函数

    def step(self, delta, reset=False):
        # 执行一个优化步骤
        z_sum = 0.0  # 初始化资格迹总和

        for group in self.param_groups:  # 遍历所有参数组
            for p in group["params"]:  # 遍历当前参数组中的每个参数
                state = self.state[p]  # 获取当前参数的状态字典

                if len(state) == 0:  # 如果状态字典为空
                    state["eligibility_trace"] = torch.zeros_like(p.data)  # 初始化资格迹为零张量

                e = state["eligibility_trace"]  # 获取当前参数的资格迹
                # 更新资格迹：乘以衰减因子并加上当前梯度
                e.mul_(group["gamma"] * group["lamda"]).add_(p.grad, alpha=1.0)
                z_sum += e.abs().sum().item()  # 计算资格迹的绝对值总和

        # 计算 delta 的绝对值和 1 的最大值
        delta_bar = max(abs(delta), 1.0)
        # 计算点积，用于后续的步长调整
        dot_product = delta_bar * z_sum * group["lr"] * group["kappa"]

        # 根据点积决定步长
        if dot_product > 1:
            step_size = group["lr"] / dot_product  # 如果点积大于 1，调整学习率
        else:
            step_size = group["lr"]  # 否则使用原学习率

        for group in self.param_groups:  # 遍历所有参数组
            for p in group["params"]:  # 遍历当前参数组中的每个参数
                state = self.state[p]  # 获取当前参数的状态字典
                e = state["eligibility_trace"]  # 获取资格迹
                # 更新参数：根据资格迹和计算的步长调整参数值
                p.data.add_(delta * e, alpha=-step_size)

                if reset:  # 如果需要重置资格迹
                    e.zero_()  # 将资格迹重置为零

class AdaptiveObGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1.0, gamma=0.99, lamda=0.8, kappa=2.0, beta2=0.999, eps=1e-8):
        defaults = dict(lr=lr, gamma=gamma, lamda=lamda, kappa=kappa, beta2=beta2, eps=eps)
        self.counter = 0
        super(AdaptiveObGD, self).__init__(params, defaults)
    def step(self, delta, reset=False):
        z_sum = 0.0
        self.counter += 1
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if len(state) == 0:
                    state["eligibility_trace"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)
                e, v = state["eligibility_trace"], state["v"]
                e.mul_(group["gamma"] * group["lamda"]).add_(p.grad, alpha=1.0)

                v.mul_(group["beta2"]).addcmul_(delta*e, delta*e, value=1.0 - group["beta2"])
                v_hat = v / (1.0 - group["beta2"] ** self.counter)
                z_sum += (e / (v_hat + group["eps"]).sqrt()).abs().sum().item()

        delta_bar = max(abs(delta), 1.0)
        dot_product = delta_bar * z_sum * group["lr"] * group["kappa"]
        if dot_product > 1:
            step_size = group["lr"] / dot_product
        else:
            step_size = group["lr"]

        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                v, e = state["v"], state["eligibility_trace"]
                v_hat = v / (1.0 - group["beta2"] ** self.counter)
                p.data.addcdiv_(delta * e, (v_hat + group["eps"]).sqrt(), value=-step_size)
                if reset:
                    e.zero_()
