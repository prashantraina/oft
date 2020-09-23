import time
import torch
from torchvision.transforms.functional import to_tensor 
from argparse import ArgumentParser
import matplotlib.pyplot as plt

import oft
from oft import KittiObjectDataset, OftNet, ObjectEncoder, visualize_objects

def parse_args():
    parser = ArgumentParser()

    parser.add_argument('model_path', type=str,
                        help='path to checkpoint file containing trained model')
    parser.add_argument('-g', '--gpu', type=int, default=0,
                        help='gpu to use for inference (-1 for cpu)')
    
    # Data options
    parser.add_argument('--root', type=str, default='data/kitti',
                        help='root directory of the KITTI dataset')
    parser.add_argument('--grid-size', type=float, nargs=2, default=(80., 80.),
                        help='width and depth of validation grid, in meters')
    parser.add_argument('--yoffset', type=float, default=1.74,
                        help='vertical offset of the grid from the camera axis')
    parser.add_argument('--nms-thresh', type=float, default=0.2,
                        help='minimum score for a positive detection')

    # Model options
    parser.add_argument('--grid-height', type=float, default=4.,
                        help='size of grid cells, in meters')
    parser.add_argument('-r', '--grid-res', type=float, default=0.5,
                        help='size of grid cells, in meters')
    parser.add_argument('--frontend', type=str, default='resnet18',
                        choices=['resnet18', 'resnet34'],
                        help='name of frontend ResNet architecture')
    parser.add_argument('--topdown', type=int, default=8,
                        help='number of residual blocks in topdown network')
    
    return parser.parse_args()


def main():

    # Parse command line arguments
    args = parse_args()

    # Load validation dataset to visualise
    dataset = KittiObjectDataset(
        args.root, 'val', args.grid_size, args.grid_res, args.yoffset)
    
    # Build model
    model = OftNet(num_classes=1, frontend=args.frontend, 
                   topdown_layers=args.topdown, grid_res=args.grid_res, 
                   grid_height=args.grid_height)

    model = torch.nn.DataParallel(model, [args.gpu])
    if args.gpu >= 0:
        torch.cuda.set_device(args.gpu)
        model.cuda()
    
    # Load checkpoint
    ckpt = torch.load(args.model_path)
    model.load_state_dict(ckpt['model'])

    # Create encoder
    encoder = ObjectEncoder(nms_thresh=args.nms_thresh)

    # Set up plots
    _, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4)
    plt.ion()

    # Iterate over validation images
    for _, image, calib, objects, grid in dataset:

        # Move tensors to gpu
        #ax1.imshow(image)
        image = to_tensor(image)
        if args.gpu >= 0:
            image, calib, grid = image.cuda(), calib.cuda(), grid.cuda()

        # Run model forwards
        pred_encoded = model(image[None], calib[None], grid[None])
        
        # Decode predictions
        pred_encoded = [t[0].cpu() for t in pred_encoded]
        
        scores, pos_offsets, dim_offsets, ang_offsets = pred_encoded

        print(scores.shape, scores.dtype)
        print(pos_offsets.shape, dim_offsets.dtype)

        image = image.cpu()
        calib = calib.cpu()
        grid = grid.cpu() 
        gt_encoded = encoder.encode(objects, grid)


        oft.vis_score(scores[0], grid, ax=ax4)
        oft.vis_score(gt_encoded[0][0], grid, ax=ax3)

        detections = encoder.decode(*pred_encoded, grid.cpu())

        # Visualize predictions
        visualize_objects(image, calib, detections, ax=ax1)
        ax1.set_title('Detections')
        visualize_objects(image, calib, objects, ax=ax2)
        ax2.set_title('Ground truth')

        plt.draw()
        #plt.pause(0.01)
        plt.waitforbuttonpress()
        #time.sleep(0.5)


if __name__ == '__main__':
    main()