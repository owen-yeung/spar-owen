from torch.optim.lr_scheduler import LambdaLR

import torch.optim as optim

# Custom scheduler that increases learning rate exponentially from 10^-50 to 10^-10

# Define the exponential function
def exponential_lr(epoch):
    initial_lr = 1e-50
    final_lr = 1e-10
    num_epochs = 100  # Adjust this value as needed

    # Calculate the exponential learning rate
    exponent = (epoch / num_epochs) * (final_lr / initial_lr)
    lr = initial_lr * (final_lr / initial_lr) ** exponent

    return lr

# Create the optimizer and scheduler
optimizer = optim.SGD(model.parameters(), lr=1e-50)
scheduler = LambdaLR(optimizer, lr_lambda=exponential_lr)

