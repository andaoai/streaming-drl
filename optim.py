import torch

class ObGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1.0, gamma=0.99, lamda=0.8, kappa=2.0):
        # 初始化优化器,设置学习率、衰减因子等超参数
        defaults = dict(lr=lr, gamma=gamma, lamda=lamda, kappa=kappa)
        super(ObGD, self).__init__(params, defaults)  # 调用父类构造函数

    def step(self, delta, reset=False):
        '''
        ### step1

        先把每个权重的资格迹进行初始化,要是有了,就没必要初始化。

        同时把z_sum 记录下来。z_sum 主要是用于控制更新权重的超参数

        ### step2

        delta 是 时序差分（Temporal Difference, TD）学习 的δ 通常指的是 时间差分误差（TD Error）。

        delta 通过价值网路获取,真实的价值与预测价值的差。

        dot_product 点积就比较骚,是结合了 时间差分误差 ,资格迹总和,学习率,kappa进行一起控制更新权重的比重。

        ### step3

        根据dot_product 来选择更新权重的步长,为什么大于1,这个有待观察
        
        经过观察,dot_product基本大于1,同时都是上万,这里有个问题,要是千的倍率,是不是收敛会更加快？

        ### step4

        更新算法,主要通过dot_product 来控制权重步长,这样会收敛更加缓慢,不至于传统的流学习带来的不稳定性。

        obdg重点在于资格迹,既是z_sum 对所有权重的资格迹进行求和,要是越大,
        
        那么说明这个状态的没必要更新？

        要是越小,说明需要更新？

        这里存在需要持续观察的。

        如何观察？怎么知道状态是有重复的？

        同时,为什么不是用资格迹,对每个权重参数,进行记录更新？反而用计算资格迹的绝对值总和？

        https://arxiv.org/pdf/2410.14606
        '''
        # step1
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
        # step2
        # 计算 delta 的绝对值和 1 的最大值。
        # 猜想：有可能前期在计算delta<1的时候,本质算法收敛很慢,还不如直接取1进行收敛。
        delta_bar = max(abs(delta), 1.0)
        # 计算点积,用于后续的步长调整
        dot_product = delta_bar * z_sum * group["lr"] * group["kappa"]

        # step3
        # 根据点积决定步长
        if dot_product > 1:
            step_size = group["lr"] / dot_product  # 如果点积大于 1,调整学习率
        else:
            step_size = group["lr"]  # 否则使用原学习率


        # step4
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
