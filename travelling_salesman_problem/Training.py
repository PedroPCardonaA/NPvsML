import torch

def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Adam,
               device: str):
    model.train()
    train_loss = 0.0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    train_loss /= len(dataloader)

    return train_loss

def val_step(model: torch.nn.Module,
             dataloader: torch.utils.data.DataLoader,
             loss_fn: torch.nn.Module,
             device: str):
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            val_loss += loss.item()
    
    val_loss /= len(dataloader)

    return val_loss
    
def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: str):
    model.eval()
    test_loss = 0.0

    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            test_loss += loss.item()
    
    test_loss /= len(dataloader)

    return test_loss


def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          val_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Adam,
          loss_fn: torch.nn.Module,
          epochs: int = 5,
          device: str = 'cpu'):
    results = {
        "train_loss": [],
        "val_loss": [],
        "test_loss": []
    }

    for epoch in range(epochs):
        train_loss = train_step(model, train_dataloader, loss_fn, optimizer, device)
        val_loss = val_step(model, val_dataloader, loss_fn, device)
        test_loss = test_step(model, test_dataloader, loss_fn, device)

        print(f"Epoch: {epoch+1} | Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f} | Test loss: {test_loss:.4f}")
        results["train_loss"].append(train_loss)
        results["val_loss"].append(val_loss)
        results["test_loss"].append(test_loss)

    return results
