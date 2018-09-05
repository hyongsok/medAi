import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision import models


if __name__ == '__main__':

    # Device configuration
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    # Hyper parameters
    num_epochs = 10
    num_classes = 4
    batch_size = 10
    learning_rate = 0.01
    image_size = 224

    print("Paremeters:\nEpochs:\t\t{num_epochs}\nBatch size:\t{batch_size}\nLearning rate:\t{learning_rate}".format(num_epochs=num_epochs, batch_size=batch_size, learning_rate=learning_rate))

    data_transform = transforms.Compose([
            transforms.Resize((image_size,image_size)),
            transforms.ToTensor(),
        ])

    train_dataset = torchvision.datasets.ImageFolder(root='./train',
                                                    transform=data_transform)

    test_dataset = torchvision.datasets.ImageFolder(root='./test',
                                                    transform=data_transform)

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=batch_size,
                                            shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=batch_size,
                                            shuffle=False)


    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    model = model_ft.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    total_step = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 1 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                    .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

    # Test the model
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        correct = 0
        total = 0
        confusion = torch.zeros((4,4), dtype=torch.int)

        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            for pred, lab in zip(predicted, labels):
                confusion[pred, lab] += 1            

        print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))
        print('Confusion matrix:\n', confusion)

    # Save the model checkpoint
    torch.save(model.state_dict(), 'model.ckpt')
