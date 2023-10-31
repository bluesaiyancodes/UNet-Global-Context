import matplotlib.pyplot as plt
import os
import argparse
import cv2
import torch


from models.Unet_skip import UNetWithSkip
from models.Unet_noskip import UNetWithoutSkip

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
            print(f'Directory {dir} already exists')

def arg_init():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, default='none', help='image path')
    return parser.parse_args()

def visualize_global_context(model, input_tensor, output_path):
    '''
    save plot of global context
    '''
    output, global_context = model(input_tensor)
    plt.imshow(global_context[0, 0].detach().cpu().numpy(), cmap='gray')
    plt.title('Global Context')
    
    # create dir if not exists
    createDir(['outputs'])

    # save the figure
    plt.savefig(output_path)


if __name__ == '__main__':
    '''
    run 
    python main.py --img_path data/1.png
    
    '''
    # Set seed
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    # Load args
    args = arg_init()

    if args.img_path == 'none':
        print('Proceeding with default image')
        args.img_path = 'data/car.jpg'

    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load image
    img = cv2.imread(args.img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img / 255.0
    img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float().to(device)


    # Load models
    model_with_skip = UNetWithSkip().to(device)
    model_without_skip = UNetWithoutSkip().to(device)

    # Visualize Global Context
    print("Global Context with Skip Connections")
    visualize_global_context(model_with_skip, img, "outputs/global_context_with_skip.png")

    print("Global Context without Skip Connections")
    visualize_global_context(model_without_skip, img, "outputs/global_context_without_skip.png")



        


