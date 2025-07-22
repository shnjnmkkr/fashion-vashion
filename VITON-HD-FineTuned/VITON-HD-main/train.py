# This is the new, optimized training script.
# It uses Multi-GPU (DataParallel), Automatic Mixed Precision (AMP),
# and generates a side-by-side visualization after each epoch.

import argparse
import os
import torch
from torch import nn
from torch.nn import functional as F
import torch.optim as optim
import math
import kornia as tgm
import random
from torch.cuda.amp import GradScaler, autocast # For Mixed Precision
from PIL import Image # For creating the composite visualization image
import numpy as np

from datasets import VITONDataset, VITONDataLoader
from networks import SegGenerator, GMM, ALIASGenerator
from utils import load_checkpoint, save_images, gen_noise

def get_opt():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--name', type=str, required=True, help="Name for this training run")
    parser.add_argument('--lr', type=float, default=2e-5, help='learning rate for fine-tuning')
    parser.add_argument('--epochs', type=int, default=30, help='total training epochs')
    parser.add_argument('--max_steps', type=int, default=5000, help='Limit steps per epoch. 0 means no limit.')
    
    # Batch size can be increased for multi-GPU + AMP
    parser.add_argument('-b', '--batch_size', type=int, default=4, help='Batch size. For T4x2 with AMP, you can try 4.')
    parser.add_argument('-j', '--workers', type=int, default=2)
    parser.add_argument('--load_height', type=int, default=1024)
    parser.add_argument('--load_width', type=int, default=768)
    
    parser.add_argument('--dataset_dir', type=str, default='/kaggle/input/clothes-tryon-dataset/', help='root directory of your dataset')
    parser.add_argument('--dataset_mode', type=str, default='train', help='train or test')
    parser.add_argument('--dataset_list', type=str, default='train_pairs.txt', help='list of training pairs')
    
    parser.add_argument('--save_checkpoint_dir', type=str, default='/kaggle/working/', help='directory to save new checkpoints')
    parser.add_argument('--save_dir', type=str, default='/kaggle/working/visuals/', help='directory to save visualization results')
    parser.add_argument('--display_freq', type=int, default=100, help='frequency of showing training results')
    
    parser.add_argument('--seg_checkpoint', type=str, default='./checkpoints/seg_final.pth', help='Full path to the segmentation model checkpoint')
    parser.add_argument('--gmm_checkpoint', type=str, default='./checkpoints/gmm_final.pth', help='Full path to the GMM model checkpoint')
    parser.add_argument('--alias_checkpoint', type=str, default='./checkpoints/alias_final.pth', help='Full path to the ALIAS model checkpoint')
    
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2. use -1 for CPU')

    # Common options
    parser.add_argument('--semantic_nc', type=int, default=13)
    parser.add_argument('--init_type', choices=['normal', 'xavier', 'xavier_uniform', 'kaiming', 'orthogonal', 'none'], default='xavier')
    parser.add_argument('--init_variance', type=float, default=0.02)
    parser.add_argument('--grid_size', type=int, default=5)
    parser.add_argument('--norm_G', type=str, default='spectralaliasinstance')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--num_upsampling_layers', choices=['normal', 'more', 'most'], default='most')
    
    opt = parser.parse_args()
    return opt

def set_requires_grad(model, requires_grad):
    for param in model.parameters():
        param.requires_grad = requires_grad

# ### NEW ### Enhanced visualization function
def tensor_to_pil(tensor):
    """Converts a torch tensor to a PIL image."""
    # Denormalize from [-1, 1] to [0, 255]
    tensor = (tensor.clone() + 1) * 0.5 * 255
    tensor = tensor.cpu().clamp(0, 255)
    
    # Handle batch dimension if present
    if tensor.dim() == 4:
        tensor = tensor[0] # Take the first image in the batch

    array = tensor.numpy().astype('uint8')
    if array.shape[0] == 1: # Grayscale
        array = array.squeeze(0)
    elif array.shape[0] == 3: # RGB
        array = array.transpose(1, 2, 0)

    return Image.fromarray(array)

def visualize_results(opt, seg, gmm, alias, epoch):
    print(f"Generating visualization for epoch {epoch+1}...")
    seg.eval()
    gmm.eval()
    alias.eval()

    test_opt = argparse.Namespace(**vars(opt))
    test_opt.dataset_mode = 'test'
    test_opt.dataset_list = 'test_pairs.txt'
    test_opt.shuffle = True
    
    test_dataset = VITONDataset(test_opt)
    test_loader = VITONDataLoader(test_opt, test_dataset)

    gauss = tgm.image.GaussianBlur((15, 15), (3, 3)).cuda()

    with torch.no_grad():
        inputs = next(iter(test_loader.data_loader))
        
        img_orig = inputs['img'].cuda() # Get original person image
        img_agnostic = inputs['img_agnostic'].cuda()
        parse_agnostic = inputs['parse_agnostic'].cuda()
        pose = inputs['pose'].cuda()
        c = inputs['cloth']['unpaired'].cuda()
        cm = inputs['cloth_mask']['unpaired'].cuda()

        with autocast():
            # ... (Forward pass logic remains the same)
            parse_agnostic_down = F.interpolate(parse_agnostic, size=(256, 192), mode='bilinear')
            pose_down = F.interpolate(pose, size=(256, 192), mode='bilinear')
            c_masked_down = F.interpolate(c * cm, size=(256, 192), mode='bilinear')
            cm_down = F.interpolate(cm, size=(256, 192), mode='bilinear')
            seg_input = torch.cat((cm_down, c_masked_down, parse_agnostic_down, pose_down, gen_noise(cm_down.size()).cuda()), dim=1)
            parse_pred_down = seg(seg_input)
            
            up = nn.Upsample(size=(opt.load_height, opt.load_width), mode='bilinear')
            parse_pred = gauss(up(parse_pred_down))
            parse_pred = parse_pred.argmax(dim=1)[:, None]

            parse_old = torch.zeros(parse_pred.size(0), 13, opt.load_height, opt.load_width, dtype=torch.float).cuda()
            parse_old.scatter_(1, parse_pred, 1.0)

            labels = {
                0: ['background', [0]], 1: ['paste', [2, 4, 7, 8, 9, 10, 11]], 2: ['upper', [3]],
                3: ['hair', [1]], 4: ['left_arm', [5]], 5: ['right_arm', [6]], 6: ['noise', [12]]
            }
            parse = torch.zeros(parse_pred.size(0), 7, opt.load_height, opt.load_width, dtype=torch.float).cuda()
            for j in range(len(labels)):
                for label in labels[j][1]:
                    parse[:, j] += parse_old[:, label]

            agnostic_gmm = F.interpolate(img_agnostic, size=(256, 192), mode='nearest')
            parse_cloth_gmm = F.interpolate(parse[:, 2:3], size=(256, 192), mode='nearest')
            pose_gmm = F.interpolate(pose, size=(256, 192), mode='nearest')
            c_gmm = F.interpolate(c, size=(256, 192), mode='nearest')
            gmm_input = torch.cat((parse_cloth_gmm, pose_gmm, agnostic_gmm), dim=1)
            _, warped_grid = gmm(gmm_input, c_gmm)
            warped_c = F.grid_sample(c, warped_grid, padding_mode='border')
            warped_cm = F.grid_sample(cm, warped_grid, padding_mode='border')

            misalign_mask = parse[:, 2:3] - warped_cm
            misalign_mask[misalign_mask < 0.0] = 0.0
            parse_div = torch.cat((parse, misalign_mask), dim=1)
            parse_div[:, 2:3] -= misalign_mask
            output = alias(torch.cat((img_agnostic, pose, warped_c), dim=1), parse, parse_div, misalign_mask)

        # ### NEW ### Create the side-by-side composite image
        person_pil = tensor_to_pil(img_orig)
        cloth_pil = tensor_to_pil(c)
        output_pil = tensor_to_pil(output)

        # Resize cloth to match the height of the person image
        h, w = person_pil.size
        cloth_h = h
        cloth_w = int(cloth_pil.width * (h / cloth_pil.height))
        cloth_pil_resized = cloth_pil.resize((cloth_w, cloth_h), Image.LANCZOS)

        # Create a new blank image to paste the three images into
        total_width = w + cloth_w + w
        composite_image = Image.new('RGB', (total_width, h))
        composite_image.paste(person_pil, (0, 0))
        composite_image.paste(cloth_pil_resized, (w, 0))
        composite_image.paste(output_pil, (w + cloth_w, 0))

        # Save the composite visualization
        save_dir = os.path.join(opt.save_dir, opt.name)
        os.makedirs(save_dir, exist_ok=True)
        composite_save_path = os.path.join(save_dir, f"composite_epoch_{epoch+1}.jpg")
        composite_image.save(composite_save_path)
        print(f"Saved composite visualization to {composite_save_path}")


def fine_tune_train(opt, seg, gmm, alias):
    seg.train()
    gmm.eval()
    alias.train()

    print("Setting up model parameter gradients...")
    
    set_requires_grad(gmm, False)
    print("GMM model is FROZEN.")

    set_requires_grad(seg, False)
    for name, param in seg.named_parameters():
        if 'up9' in name or 'conv9' in name:
            param.requires_grad = True
    print("SegGenerator is partially unfrozen (last 2 blocks).")

    set_requires_grad(alias, False)
    for name, param in alias.named_parameters():
        if 'head_0' in name or 'G_middle' in name or 'up_' in name:
            param.requires_grad = True
    print("ALIASGenerator is partially unfrozen (ResBlocks).")

    trainable_params = list(filter(lambda p: p.requires_grad, seg.parameters())) + \
                     list(filter(lambda p: p.requires_grad, alias.parameters()))
    
    optimizer = optim.Adam(trainable_params, lr=opt.lr, betas=(0.5, 0.999))
    scaler = GradScaler()

    criterionL1 = nn.L1Loss()
    criterionCE = nn.CrossEntropyLoss()

    opt.shuffle = True 
    train_dataset = VITONDataset(opt)
    train_loader = VITONDataLoader(opt, train_dataset)

    total_steps_per_epoch = len(train_loader.data_loader)
    if opt.max_steps > 0:
        total_steps_per_epoch = min(opt.max_steps, total_steps_per_epoch)

    print(f"Starting fine-tuning for {opt.epochs} epochs...")

    for epoch in range(opt.epochs):
        for i, inputs in enumerate(train_loader.data_loader):
            if opt.max_steps > 0 and i >= opt.max_steps:
                break

            img_agnostic = inputs['img_agnostic'].cuda()
            parse_agnostic = inputs['parse_agnostic'].cuda()
            pose = inputs['pose'].cuda()
            c = inputs['cloth']['unpaired'].cuda()
            cm = inputs['cloth_mask']['unpaired'].cuda()
            
            gt_parse_full = inputs['parse_agnostic'].cuda() 
            gt_image = inputs['img'].cuda()

            with autocast():
                # ... (Forward pass logic remains the same)
                parse_agnostic_down = F.interpolate(parse_agnostic, size=(256, 192), mode='bilinear')
                pose_down = F.interpolate(pose, size=(256, 192), mode='bilinear')
                c_masked_down = F.interpolate(c * cm, size=(256, 192), mode='bilinear')
                cm_down = F.interpolate(cm, size=(256, 192), mode='bilinear')
                
                seg_input = torch.cat((cm_down, c_masked_down, parse_agnostic_down, pose_down, gen_noise(cm_down.size()).cuda()), dim=1)
                parse_pred_down = seg(seg_input)
                
                loss_seg = criterionCE(parse_pred_down, F.interpolate(gt_parse_full, size=(256, 192), mode='nearest').argmax(dim=1))
                
                labels = {
                    0: ['background', [0]], 1: ['paste', [2, 4, 7, 8, 9, 10, 11]], 2: ['upper', [3]],
                    3: ['hair', [1]], 4: ['left_arm', [5]], 5: ['right_arm', [6]], 6: ['noise', [12]]
                }
                parse = torch.zeros(gt_parse_full.size(0), 7, opt.load_height, opt.load_width, dtype=torch.float).cuda()
                for j in range(len(labels)):
                    for label in labels[j][1]:
                        parse[:, j] += gt_parse_full[:, label]

                with torch.no_grad():
                    agnostic_gmm = F.interpolate(img_agnostic, size=(256, 192), mode='nearest')
                    parse_cloth_gmm = F.interpolate(parse[:, 2:3], size=(256, 192), mode='nearest')
                    pose_gmm = F.interpolate(pose, size=(256, 192), mode='nearest')
                    c_gmm = F.interpolate(c, size=(256, 192), mode='nearest')
                    gmm_input = torch.cat((parse_cloth_gmm, pose_gmm, agnostic_gmm), dim=1)
                    _, warped_grid = gmm(gmm_input, c_gmm)
                    warped_c = F.grid_sample(c, warped_grid, padding_mode='border')
                    warped_cm = F.grid_sample(cm, warped_grid, padding_mode='border')

                misalign_mask = parse[:, 2:3] - warped_cm
                misalign_mask[misalign_mask < 0.0] = 0.0
                parse_div = torch.cat((parse, misalign_mask), dim=1)
                parse_div[:, 2:3] -= misalign_mask
                output = alias(torch.cat((img_agnostic, pose, warped_c), dim=1), parse, parse_div, misalign_mask)

                loss_alias_l1 = criterionL1(output, gt_image)
                loss_total = loss_seg + loss_alias_l1
            
            optimizer.zero_grad()
            scaler.scale(loss_total).backward()
            scaler.step(optimizer)
            scaler.update()

            if (i + 1) % opt.display_freq == 0:
                print(f"Epoch [{epoch+1}/{opt.epochs}], Step [{i+1}/{total_steps_per_epoch}], Total Loss: {loss_total.item():.4f}, Seg Loss: {loss_seg.item():.4f}, Alias L1 Loss: {loss_alias_l1.item():.4f}")

        save_dir = opt.save_checkpoint_dir
        os.makedirs(save_dir, exist_ok=True)
        
        save_path_seg = os.path.join(save_dir, f'{opt.name}_seg_epoch_{epoch+1}.pth')
        save_path_alias = os.path.join(save_dir, f'{opt.name}_alias_epoch_{epoch+1}.pth')
        
        if isinstance(seg, nn.DataParallel):
            torch.save(seg.module.state_dict(), save_path_seg)
            torch.save(alias.module.state_dict(), save_path_alias)
        else:
            torch.save(seg.state_dict(), save_path_seg)
            torch.save(alias.state_dict(), save_path_alias)
        print(f"Saved fine-tuned checkpoints for epoch {epoch+1} to {save_dir}")

        visualize_results(opt, seg, gmm, alias, epoch)
        seg.train()
        alias.train()


def main():
    opt = get_opt()
    print("----------- Options -----------")
    for k, v in sorted(vars(opt).items()):
        print(f"{str(k)}: {str(v)}")
    print("-------------------------------")

    str_ids = opt.gpu_ids.split(',')
    opt.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            opt.gpu_ids.append(id)
    if len(opt.gpu_ids) > 0:
        torch.cuda.set_device(opt.gpu_ids[0])

    seg = SegGenerator(opt, input_nc=opt.semantic_nc + 8, output_nc=opt.semantic_nc)
    gmm = GMM(opt, inputA_nc=7, inputB_nc=3)
    opt.semantic_nc = 7
    alias = ALIASGenerator(opt, input_nc=9)
    opt.semantic_nc = 13

    # Load original pre-trained weights to start from scratch
    load_checkpoint(seg, opt.seg_checkpoint)
    load_checkpoint(gmm, opt.gmm_checkpoint)
    load_checkpoint(alias, opt.alias_checkpoint)
    print("Original pre-trained weights loaded for a fresh start.")

    if len(opt.gpu_ids) > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        seg = nn.DataParallel(seg, device_ids=opt.gpu_ids)
        gmm = nn.DataParallel(gmm, device_ids=opt.gpu_ids)
        alias = nn.DataParallel(alias, device_ids=opt.gpu_ids)

    seg.cuda()
    gmm.cuda()
    alias.cuda()

    fine_tune_train(opt, seg, gmm, alias)

if __name__ == '__main__':
    main()
