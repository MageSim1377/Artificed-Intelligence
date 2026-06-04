import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torchvision import transforms, datasets
    from torch.utils.data import DataLoader
    import numpy as np
    from PIL import Image

    return DataLoader, Image, datasets, nn, np, optim, torch, transforms


@app.cell
def _(transforms):
    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandAugment(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return test_transform, train_transform


@app.cell
def _(DataLoader, datasets, test_transform, train_transform):
    train_dataset = datasets.ImageFolder(root="leapGestRecog/train", transform=train_transform)
    test_dataset = datasets.ImageFolder(root="leapGestRecog/test", transform=test_transform)
    batch_size = 32
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_dataloader, train_dataloader, train_dataset


@app.cell
def _(nn):
    class CNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.convLayer = nn.Sequential(
                nn.Conv2d(3, 7, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.MaxPool2d(4, 4)
            )
            self.linlayer = nn.Linear(32*32 * 7, 3)
        def forward(self, x):
            y = self.convLayer(x)
            y = y.view(x.size(0), -1)
            y = self.linlayer(y)
            return y

    return (CNN,)


@app.cell
def _(torch):
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available else "cpu"
    return (device,)


@app.cell
def _(CNN, device):
    model = CNN().to(device)
    return (model,)


@app.cell
def _(model, nn, optim):
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=1e-3)
    return loss_fn, opt


@app.cell
def _(torch):
    def accuracy(output, labels):
        pred = torch.argmax(output, dim=1)
        correct = (pred == labels).sum().cpu().numpy()
        return correct/len(labels)

    return (accuracy,)


@app.cell
def _(
    accuracy,
    device,
    loss_fn,
    model,
    np,
    opt,
    test_dataloader,
    torch,
    train_dataloader,
):
    epochs = 10

    max_acc = 0

    for epoch in range(epochs):
        model.train()

        running_loss = 0
        for batch, (imgs, labels) in enumerate(train_dataloader):
            imgs, labels = imgs.to(device), labels.to(device)

            pred = model(imgs)

            loss = loss_fn(pred, labels)

            running_loss += loss

            loss.backward()
            opt.step()
            opt.zero_grad()


            if batch % 10 == 0:
                print(f'[{epoch + 1}, {batch + 1:5d}] loss: {running_loss / 10:.3f}')
                running_loss = 0

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        model.eval()

        accuracies = []
        with torch.no_grad():
            for imgs, labels in test_dataloader:
                imgs, labels = imgs.to(device), labels.to(device)
                pred = model(imgs)
                accuracies.append(accuracy(pred, labels))
        acc = np.mean(np.array(accuracies))
        print(f"Validation accuracy: {acc:.3f}")

        if acc > max_acc:
            max_acc = acc
            print("Save model because it is better")
            torch.save(model, f'models/{acc:.3f}.pth')
    return


@app.cell
def _(Image, device, test_transform, torch, train_dataset):
    loaded_model = torch.load('models/0.840.pth', weights_only=False).to(device)

    loaded_model.eval()

    test_img = Image.open("D:/Уник работы/2 курс/Artificed Intelligence/test_img/img3.png").convert('RGB')
    transformed = test_transform(test_img).unsqueeze(0).to(device)

    predicted = loaded_model(transformed)
    print(train_dataset.classes)
    print(predicted)
    print(train_dataset.classes[torch.argmax(predicted)])
    return


if __name__ == "__main__":
    app.run()
