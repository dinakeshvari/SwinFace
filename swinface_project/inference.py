import argparse
import io
import cv2
import numpy as np
import torch
from tqdm import tqdm
import pickle
import os

from model import build_model

@torch.no_grad()
def inference(cfg, weight, img):
    if img is None:
        img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.uint8)
    else:
        img = cv2.imread(img)
        img = cv2.resize(img, (112, 112))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()
    img.div_(255).sub_(0.5).div_(0.5)

    model = build_model(cfg)
    dict_checkpoint = torch.load(weight, map_location=torch.device('cpu'))
    model.backbone.load_state_dict(dict_checkpoint["state_dict_backbone"])
    model.fam.load_state_dict(dict_checkpoint["state_dict_fam"])
    model.tss.load_state_dict(dict_checkpoint["state_dict_tss"])
    model.om.load_state_dict(dict_checkpoint["state_dict_om"])

    model.eval()
    output = model(img)
    embeddings = {k: v[0].numpy() for k, v in output.items()}

    return embeddings

class SwinFaceCfg:
    network = "swin_t"
    fam_kernel_size=3
    fam_in_chans=2112
    fam_conv_shared=False
    fam_conv_mode="split"
    fam_channel_attention="CBAM"
    fam_spatial_attention=None
    fam_pooling="max"
    fam_la_num_list=[2 for j in range(11)]
    fam_feature="all"
    fam = "3x3_2112_F_s_C_N_max"
    embedding_size = 512

import os


if __name__ == "__main__":
    # cfg = SwinFaceCfg()
    # parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')
    # parser.add_argument('--weight', type=str, default='/content/checkpoint_step_79999_gpu_0.pt')
    # parser.add_argument("--img_dir", type=str, default="/content/CFD_03/AF-200/CFD-AF-200-228-N.jpg", help="/content/CFD_03/AF-200/CFD-AF-200-228-N.jpg")  
    # args, _ = parser.parse_known_args()  # Changed to parse_known_args()

    # # Get list of all image paths in the directory
    # img_paths = [os.path.join(args.img_dir, img) for img in os.listdir(args.img_dir) if img.endswith(('.png', '.jpg', '.jpeg'))]

    # # Sort the list of image paths
    # img_paths.sort()

    # # Loop over all images and perform inference
    # embeddings_dict = {}
    # for img_path in tqdm(img_paths, desc="Processing images"):
    #     embeddings = inference(cfg, args.weight, img_path)
    #     embeddings_dict[img_path] = embeddings
    # np.save('embeddings_dict.npy', embeddings_dict)
    # # Save embeddings_dict to a pickle file
    # with open('embeddings_dict.pickle', 'wb') as handle:
    #     pickle.dump(embeddings_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # inference(cfg, args.weight, args.img)

    ## defining identities

    dataset_path = '/content/CFD_03'
    identities = [os.path.join(dataset_path, folder) for folder in os.listdir(dataset_path)]


    cfg = SwinFaceCfg()
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')
    parser.add_argument('--weight', type=str, default='/content/checkpoint_step_79999_gpu_0.pt')
    parser.add_argument('--img', type=str, default="/content/CFD_03/AF-200/CFD-AF-200-228-N.jpg")
    args = parser.parse_args()
    # print(inference(cfg, args.weight, args.img))
    # emb1=(inference(cfg, args.weight, args.img))

    embeddings = {}
    for identity in tqdm(identities, desc="Precomputing embeddings"):
        images = os.listdir(identity)
        for img in images:
            img_path = os.path.join(identity, img)
            if img_path not in embeddings:  # Avoid redundant computations
                try:
                    embeddings[img_path] = inference(cfg, args.weight, img_path)
                except Exception as e:
                    print(f"Error processing image {img_path}: {e}")

    with open('embeddings_dict.pickle', 'wb') as handle:
      pickle.dump(embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)
