import torch
from torch import nn
from torch.optim import Adam
from utils import create_pos_data, create_neg_data, overlay_y_on_x

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

torch.manual_seed(42)


class FFLayer(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.relu = torch.nn.ReLU()
        self.optimizer = Adam(self.parameters(), lr=0.03)
        self.threshold = 2.0
        self.num_epochs = 1000

    def forward(self, X):
        # FF uses the simplest version of layer normalisation which does not
        # subtract the mean before dividing by the lenght of the activity vector
        X_norm = X.norm(2, 1, keepdim=True)
        X_dir = X / (X_norm + 1e-4)  # X_norm could be zero
        res = torch.mm(X_dir, self.weight.T) + self.bias
        return self.relu(res)

    def train(self, X_pos, X_neg):
        # The goodness function for a layer is simply the sum of the squares of
        # the activities of the rectified linear neurons in that layer
        for i in range(self.num_epochs):
            g_pos = self.forward(X_pos).pow(2).mean(1)
            g_neg = self.forward(X_neg).pow(2).mean(1)

            # We want the loss for positive samples to be larger than the
            # threshold and negative samples to be smaller than the threshold
            loss = torch.sigmoid(
                torch.cat([-g_pos + self.threshold, g_neg - self.threshold])
            ).mean()

            # Many implementations online use softplus loss function
            # https://keras.io/examples/vision/forwardforward/
            # loss = torch.log(
            #     1
            #     + torch.exp(
            #         torch.cat([-g_pos + self.threshold, g_neg - self.threshold])
            #     )
            # ).mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if i % 20 == 0:
                print("Loss:", loss.item())

        return self.forward(X_pos).detach(), self.forward(X_neg).detach()


class FFNetwork(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.layers = []
        for d in range(len(dims) - 1):
            self.layers += [FFLayer(dims[d], dims[d + 1]).to(device)]

    def train(self, train_dataloader):
        # for _ in range(self.num_epochs):
        for X, y in train_dataloader:
            X_pos = create_pos_data(X, y).to(device)
            X_neg = create_neg_data(X, y).to(device)
            for i, layer in enumerate(self.layers):
                print("Training layer", i, "...")
                X_pos, X_neg = layer.train(X_pos, X_neg)

    @torch.no_grad()
    def calc_goodness(self, X):
        # Sum up the goodness of every layer
        goodness = []
        for layer in self.layers:
            X = layer(X)
            goodness += [X.pow(2).mean(1)]
        # return torch.stack(goodness).sum(0)
        return sum(goodness)  # torch.Size([10000])

    def predict(self, test_dataloader):
        # No batching for test set, this loop runs only once
        for X, y in test_dataloader:
            X, y = X.to(device), y.to(device)
            goodness_per_label = []
            for label in range(10):
                X_ = overlay_y_on_x(X, label)
                goodness_per_label += [self.calc_goodness(X_)]
            goodness_per_label = torch.stack(
                goodness_per_label, 1
            )  # (torch.Size([10000, 10]))
            y_ = goodness_per_label.argmax(1)  # torch.Size([10000])
            return y_.eq(y).float().mean().item()
