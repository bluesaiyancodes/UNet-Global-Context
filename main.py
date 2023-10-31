import matplotlib.pyplot as plt
import os
import argparse
import cv2
import torch
import torch.nn as nn
from tqdm import tqdm

from torchvision import transforms
from torch.utils.data import DataLoader

from models.Unet_skip import UNetWithSkip
from models.Unet_noskip import UNetWithoutSkip
from dataset import CarSegmentationDataset

# Create directory
def createDir(dirs):
    '''
    Create a directory if it does not exist
    dirs: a list of directories to create
    '''
    for dir in dirs:
        if not os.path.exists(dir):
            os.makedirs(dir)
        else:
            print('Directory %s already exists' %dir)

def arg_init():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, default='none', help='image path')
    return parser.parse_args()

def visualize_global_context_1(model, input_tensor, output_path):
    '''
    save plot of global context
    '''
    output, global_context = model(input_tensor)
    global_context = global_context *255
    plt.imshow(global_context[0, 0].detach().cpu().numpy(), cmap='gray')
    plt.title('Global Context')
    
    # create dir if not exists
    createDir(['outputs'])

    # save the figure
    plt.savefig(output_path)

def visualize_global_context(model, data_loader, output_path):
    data_iter = iter(data_loader)
    images, _ = next(data_iter)
    images = images.to(device)
    output, global_context = model(images)
    plt.imshow(global_context[0, 0].detach().cpu().numpy(), cmap='gray')
    plt.title('Global Context')
    # create dir if not exists
    createDir(['outputs'])
    # save the figure
    plt.savefig(output_path)

def train_model(model, optimizer, epochs=10, device='cuda'):
    for epoch in range(epochs):
        for batch_idx, (data, _) in tqdm(enumerate(train_loader)): 
            data = data.to(device)
            output, _ = model(data)
            loss = criterion(output, data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")


if __name__ == '__main__':
    '''
    Using Mnist datset to visualize the global context
    
    '''

    # Load args
    args = arg_init()

    if args.img_path == 'none':
        print('Proceeding with default image')
        args.img_path = 'data/car.jpg'

    # Set seed
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    # Set epochs
    epochs = 5

    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load image
    img = cv2.imread(args.img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img / 255.0
    img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float().to(device)




    # Data pre-processing and loading

    transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    ])

    image_dir = 'data/car-segmentation/images'
    mask_dir = 'data/car-segmentation/masks'

    train_dataset = CarSegmentationDataset(image_dir, mask_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    '''# Test dataset for visualization
    test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)'''


    # Load models
    model_with_skip = UNetWithSkip().to(device)
    model_without_skip = UNetWithoutSkip().to(device)

    criterion = nn.MSELoss()
    optimizer_with_skip = torch.optim.Adam(model_with_skip.parameters(), lr=0.001)
    optimizer_without_skip = torch.optim.Adam(model_without_skip.parameters(), lr=0.001)


    # Train models
    print("Training U-Net with Skip Connections:")
    train_model(model_with_skip, optimizer_with_skip, epochs=epochs, device=device)

    print("Training U-Net without Skip Connections:")
    train_model(model_without_skip, optimizer_without_skip, epochs=epochs, device=device)



    # Visualize Global Context
    print("Global Context with Skip Connections")
    visualize_global_context_1(model_with_skip, img, "outputs/global_context_with_skip_car.png")

    print("Global Context without Skip Connections")
    visualize_global_context_1(model_without_skip, img, "outputs/global_context_without_skip_car.png")
