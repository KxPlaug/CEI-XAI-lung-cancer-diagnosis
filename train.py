import torch
from torchvision import datasets, transforms, models
import torch.nn as nn
import torch.optim as optim
import os
model_name = 'wide_resnet50_2'
width = 299 if model_name == 'inception_v3' else 224
# 数据转换
transform = transforms.Compose([
    transforms.Resize((width, width)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

print("Processing and splitting dataset...")
data_dir = 'data/The IQ-OTHNCCD lung cancer dataset'
dataset = datasets.ImageFolder(root=data_dir, transform=transform)

dataset_size = len(dataset)
train_split = 0.8
test_split = 0.2
train_size = int(train_split * dataset_size)
test_size = dataset_size - train_size

train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))

batch_size = 32
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 模型加载函数
def load_model(model_name):
    if model_name == 'resnet50':
        model = models.resnet50(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 3)
    elif model_name == 'vgg16':
        model = models.vgg16(pretrained=True)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, 3)
    elif model_name == 'inception_v3':
        model = models.inception_v3(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 3)
    elif model_name == 'vit_b_16':
        model = models.vit_b_16(pretrained=False, num_classes=3)
    elif model_name == 'googlenet':
        model = models.googlenet(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 3)
    elif model_name == 'mobilenet_v2':
        model = models.mobilenet_v2(pretrained=True)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, 3)
    elif model_name == 'resnext50_32x4d':
        model = models.resnext50_32x4d(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 3)
    elif model_name == 'wide_resnet50_2':
        model = models.wide_resnet50_2(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 3)
    else:
        raise ValueError(f"Model {model_name} not recognized.")
    return model


model = load_model(model_name)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

num_epochs = 50
best_acc = 0.0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        if model_name == 'inception_v3':
            outputs = outputs.logits
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            # _, predicted = torch.max(outputs.data, 1)
            predicted = torch.argmax(outputs, -1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_acc = correct / total
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {epoch_acc:.4f}')

    if epoch_acc > best_acc:
        best_acc = epoch_acc
        torch.save(model.state_dict(), model_name + '.pth')
        print(f'Saving model with accuracy {best_acc:.4f}')

print('Finished Training')
