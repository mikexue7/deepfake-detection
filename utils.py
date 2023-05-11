import torch
import torch.nn as nn
import torch.nn.functional as F

def train(model, optimizer, loader_train, device, epochs=1):
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    for e in range(epochs):
        print(f"Begin training for epoch {e + 1}")
        for t, (x, y) in enumerate(loader_train):
            model.train()  # put model to training mode
            x = x.to(device=device)  # move to device, e.g. GPU
            y = y.to(device=device)

            scores = model(x)
            loss = F.binary_cross_entropy(torch.sigmoid(scores), y)

            # Zero out all of the gradients for the variables which the optimizer
            # will update.
            optimizer.zero_grad()

            # This is the backwards pass: compute the gradient of the loss with
            # respect to each  parameter of the model.
            loss.backward()

            # Actually update the parameters of the model using the gradients
            # computed by the backwards pass.
            optimizer.step()

            if t % 10 == 0:
                print('Iteration %d, loss = %.4f' % (t, loss.item()))