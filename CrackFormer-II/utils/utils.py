class Averagvalue(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def updateLR(max_lr, epoch, total_epoch):
    """
    Implements gradual warmup, if train_steps < warmup_steps, the
    learning rate will be `train_steps/warmup_steps * init_lr`.
    Args:
        warmup_steps:warmup步长阈值,即train_steps<warmup_steps,使用预热学习率,否则使用预设值学习率
        train_steps:训练了的步长数
        init_lr:预设置学习率
    """
    warmup_steps = 20
    lr_0 = 1e-6
    end_lr = 1e-7
    warmup_learning_rate = max_lr
    sigma = 0.98

    if epoch < warmup_steps:
        lr_step = (max_lr - lr_0) / warmup_steps
        warmup_learning_rate = lr_step * epoch  # gradual warmup_lr
        lr = warmup_learning_rate
    else:
        lr_step = (max_lr - end_lr) / (total_epoch - warmup_steps)
        lr = warmup_learning_rate - ((epoch - warmup_steps) * lr_step) ** sigma # 预热学习率结束后,学习率呈指数衰减(近似模拟指数衰减)  0.85
        if lr <= end_lr:
            lr = end_lr
    return lr
