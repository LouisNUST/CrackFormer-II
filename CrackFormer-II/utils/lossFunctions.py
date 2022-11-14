from torch.nn.functional import sigmoid
import torch.nn as nn

def bce2d(input, target):
    n, c, h, w = input.size()
    # assert(max(target) == 1)

    # print(target.size())
    log_p = input.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)
    # target_t = target.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)
    target_t = target.view(1, -1) # 摊平了

    target_trans = target_t.clone()
    pos_index = (target_t > 0)
    neg_index = (target_t == 0)
    target_trans[pos_index] = 1
    target_trans[neg_index] = 0
    pos_index = pos_index.data.cpu().numpy().astype(bool)
    neg_index = neg_index.data.cpu().numpy().astype(bool)
    weight = torch.Tensor(log_p.size()).fill_(0)
    weight = weight.numpy()
    pos_num = pos_index.sum()
    neg_num = neg_index.sum()
    sum_num = pos_num + neg_num
    weight[pos_index] = neg_num * 1.0 / sum_num
    weight[neg_index] = pos_num * 1.0 / sum_num

    weight = torch.from_numpy(weight)
    weight = weight.cuda()

    loss = F.binary_cross_entropy(log_p, target_trans.float(), weight, size_average=True)
    return loss

def cross_entropy_loss2d(inputs, targets, balance=1.1):
    """
    :param inputs: inputs is a 4 dimensional data nx1xhxw
    :param targets: targets is a 3 dimensional data nx1xhxw
    :return:
    """
    n, c, h, w = inputs.size()
    weights = np.zeros((n, c, h, w))
    for i in range(n):
        t = targets[i, :, :, :].cpu().data.numpy()
        pos = (t == 1).sum()
        neg = (t == 0).sum()
        valid = neg + pos
        weights[i, t == 1] = neg * 1. / valid
        weights[i, t == 0] = pos * balance / valid
    weights = torch.Tensor(weights)

    weights = weights.cuda()
    inputs = F.sigmoid(inputs)

    loss = nn.BCELoss(weights, size_average=False)(inputs, targets)
    return loss

def cross_entropy_loss_RCF(prediction, label):
    label = label.long()
    # label2 = label.float()
    mask = label.float()
    num_positive = torch.sum((mask==1).float()).float()
    num_negative = torch.sum((mask==0).float()).float()

    mask[mask == 1] = 1.0 * num_negative / (num_positive + num_negative)
    mask[mask == 0] = 1.1 * num_positive / (num_positive + num_negative)
    mask[mask == 2] = 0
    prediction = sigmoid(prediction)
    # print(label.shape)
    cost = torch.nn.functional.binary_cross_entropy(
            prediction.float(),label.float(),weight = mask, reduce=False)
    # weight = mask
    # return torch.sum(cost)
    # loss = F.binary_cross_entropy(prediction, label2, mask, size_average=True)
    return torch.sum(cost) /(num_positive+num_negative)


def BinaryFocalLoss(inputs, targets):
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    BCE_loss = criterion(inputs, targets)
    pt = torch.exp(-BCE_loss)
    F_loss = (1-pt)**2 * BCE_loss
    return F_loss.mean()
