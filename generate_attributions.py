import torch.nn as nn

from torchvision import datasets, models
from torchvision import transforms
import torch
import numpy as np
import argparse
import torch
import random
import os
from torchvision import transforms
from saliency.saliency_zoo import guided_ig, mfaba_smooth, agi, saliencymap
from tqdm import tqdm
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

setup_seed(3407)


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





parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='inception_v3')
parser.add_argument('--attr_method', type=str, default='mfaba')
parser.add_argument("--dataset",type=str,default="lung_cancer",choices=["lung_cancer"])
parser.add_argument("--smooth_sigma",type=float,default=0.25)
parser.add_argument("--use_smooth",action="store_true")
args = parser.parse_args()

perfix = f"attributions_{args.dataset}" + ("_new" if args.use_smooth else "")
os.makedirs(perfix,exist_ok=True)

if __name__ == "__main__":
    import os
    width = 299 if args.model == 'inception_v3' else 224
    transform = transforms.Compose([
        transforms.Resize((width, width)),
        transforms.ToTensor(),
    ])
    os.makedirs("logs",exist_ok=True)
    f = open(f"logs/{args.model}_{args.attr_method}.txt","w")
    

    if args.attr_method == 'guided_ig':
        attr_method = guided_ig
    elif args.attr_method == 'mfaba':
        attr_method = mfaba_smooth
    elif args.attr_method == "agi":
        attr_method = agi
    elif args.attr_method == "saliencymap":
        attr_method = saliencymap
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    

    print("Processing and splitting dataset...")
    data_dir = 'data/The IQ-OTHNCCD lung cancer dataset'
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    dataset_size = len(dataset)
    train_split = 0.8
    test_split = 0.2
    train_size = int(train_split * dataset_size)
    test_size = dataset_size - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))

    batch_size = 1
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    

    sm = nn.Softmax(dim=-1)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    norm_layer = transforms.Normalize(mean, std)
    sm = nn.Softmax(dim=-1)
    model = load_model(args.model)
    model.load_state_dict(torch.load(f"{args.model}.pth"))
    model = nn.Sequential(norm_layer, model, sm).eval().to(device)
    from saliency.core.smooth import Smooth
    model = Smooth(model,sigma=args.smooth_sigma) if args.use_smooth else model
    attributions = []
    if args.attr_method == 'guided_ig':
        batch_size = 1
    elif args.attr_method == 'agi':
        batch_size = 64
    elif args.attr_method == 'mfaba':
        batch_size = 128
    elif args.attr_method == 'saliencymap':
        batch_size = 128
    batch_size = 1
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    for i, (img, target) in enumerate(tqdm(test_loader)):
        
        img = img.to(device)
        target = target.to(device)
        attribution = attr_method(model, img, target)
        attributions.append(attribution)

    f.close()
