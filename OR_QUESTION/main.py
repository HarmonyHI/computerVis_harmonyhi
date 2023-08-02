import torch
from torch import nn
from torch.functional import F


class OrQuestionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.core1 = nn.Linear(2, 10)
        self.core2 = nn.Linear(10, 1)
        self.loss_func = torch.nn.MSELoss()

    def forward(self, data):
        ans = self.core1(data)
        ans = F.relu(ans)
        ans = self.core2(ans)
        correct = torch.tensor([1 if (data[0] != data[1]) else 0]).to(torch.float32)
        loss = self.loss_func(ans, correct)
        return ans, loss


if __name__ == '__main__':
    net = OrQuestionNet()
    net.train()
    data_out = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]]).to(torch.float32)
    for i in range(2000):
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
        print(f"=========epoch {i + 1}===============")
        for j in range(data_out.shape[0]):
            ans_out, loss_out = net(data_out[j])
            loss_out.backward()
            optimizer.step()
            optimizer.zero_grad()
            print(f"input {data_out[j][0]} {data_out[j][1]} out {ans_out[0]} loss {loss_out}")
