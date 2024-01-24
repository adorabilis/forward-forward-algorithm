from model import FFNetwork
from utils import load_mnist, demo_samples

train_dataloader, test_dataloader = load_mnist()

for X, y in test_dataloader:
    print(f"Shape of X: {X.shape} {X.dtype}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    demo_samples(X, y)
    break

model = FFNetwork([784, 500, 500])
model.train(train_dataloader)
print("Test error:", 1.0 - model.predict(test_dataloader))
