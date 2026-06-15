
from torchvision import transforms
from utils.dataset import CocoGroundTruthDataset
from PIL import Image
import torch.nn.functional as F
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from tqdm import tqdm
import time
import argparse
from tokenizer.tokenizer_image.vq_model import VQ_models
import numpy as np


def post_process(img):
    img = (img + 1.0) / 2.0 
    img = torch.clamp(img, 0.0, 1.0)
    img = (img * 255.0).clamp_(0,255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()[0]
    img = Image.fromarray(img)
    return img

def pre_process(img):
    img = (img * 2.0) - 1.0 
    image = img.permute(0, 3, 1, 2)
    return image

def tokenize_and_reconstruct_batch_latent_optim(vq_model,original_image_batch, lr=1e-2, iters=100):
    image_batch = original_image_batch.clone()
    encoded_tokens, hidden_states, quantized_states, codebook_loss = vq_model.encode_with_internals(image_batch.clone())
    fhat_optim = torch.nn.Parameter(quantized_states.detach()).cuda()
    optimizer = torch.optim.Adam([fhat_optim], lr=lr)
    for i in range(iters):
        optimizer.zero_grad()     
        rec_gen_img = vq_model.decode(fhat_optim)
        rec_gen_img = torch.clamp(rec_gen_img, -1.0, 1.0)
        loss = F.mse_loss(rec_gen_img, image_batch.clone())
        loss.backward()
        optimizer.step()
        # print(loss)
        if i%50==0:
            for g in optimizer.param_groups:
                g['lr'] = g['lr']*0.5
    return rec_gen_img, fhat_optim, loss

def calc_latent_tracer(vq_model, args, dataset_name_image_path, device):

    for dataset,path in dataset_name_image_path.items():
        print(f"Calculating latent tracer for {dataset} from {path}", flush=True)
        gen_dataset = CocoGroundTruthDataset(path, num_samples=args.num_samples)
        transform = transforms.Compose([transforms.ToTensor()])
        def pil_to_tensor_collate(batch):
            images, labels = zip(*batch)
            images = [transform(img) if isinstance(img, Image.Image) else img for img in images]
            images = torch.stack(images, dim=0)
            return images, labels
        dataloaders = torch.utils.data.DataLoader(gen_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=pil_to_tensor_collate)
        img_rec_loss_mse_list = []
        timings = []
        for images, _ in tqdm(dataloaders):
            start_time = time.time()
            images = (images.to(device) * 2.0) - 1.0 
            reconstructed_image, _, loss = tokenize_and_reconstruct_batch_latent_optim(vq_model,images)
            img_rec_loss_mse = torch.mean((reconstructed_image - images) ** 2, dim=[1, 2, 3])
            end_time = time.time()
            timings.append(end_time - start_time)
            img_rec_loss_mse_list.append(img_rec_loss_mse.cpu().detach())
        print(f"Average time per image for {dataset}: {np.mean(timings)} seconds", flush=True)
        img_rec_loss_mse_all = torch.cat(img_rec_loss_mse_list, dim=0)
        torch.save(img_rec_loss_mse_all, f"/storage2/freewatermark/llamagen/latenttracer/{dataset}_mse_list.pt")


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=10000, help="Number of samples to process per dataset")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for processing images")
    parser.add_argument("--dataset_config", type=str, default="dataset_config.json", help="Path to JSON file with dataset names and image paths")

    # Model configuration arguments
    parser.add_argument("--gpt_model", type=str, default="GPT-L", help="GPT model type")
    parser.add_argument("--gpt_ckpt", type=str, default="", help="Path to GPT checkpoint")
    parser.add_argument("--gpt_type", type=str, default="c2i", help="GPT type")
    parser.add_argument("--from_fsdp", action="store_true", help="Whether loading from FSDP checkpoint")
    parser.add_argument("--cls_token_num", type=int, default=1, help="Number of class tokens")
    parser.add_argument("--precision", type=str, default="bf16", help="Precision type")
    parser.add_argument("--compile", action="store_true", help="Whether to compile the model")
    parser.add_argument("--vq_model", type=str, default="VQ-16", help="VQ model type")
    parser.add_argument("--vq_ckpt", type=str, default="", help="Path to VQ checkpoint")
    parser.add_argument("--codebook_size", type=int, default=16384, help="Codebook size")
    parser.add_argument("--codebook_embed_dim", type=int, default=8, help="Codebook embedding dimension")
    parser.add_argument("--image_size", type=int, default=256, help="Image size")
    parser.add_argument("--downsample_size", type=int, default=16, help="Downsample size")
    parser.add_argument("--num_classes", type=int, default=1000, help="Number of classes")
    parser.add_argument("--cfg_scale", type=float, default=4.0, help="CFG scale")
    parser.add_argument("--cfg_interval", type=int, default=10, help="CFG interval")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--top_k", type=int, default=2000, help="Top-k sampling")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for sampling")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p sampling")

    args = parser.parse_args()
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
    vq_model = VQ_models[args.vq_model](
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim)
    vq_model.to(device)
    vq_model.eval()

    checkpoint = torch.load(args.vq_ckpt, map_location=device, weights_only=False)
    vq_model.load_state_dict(checkpoint["model"], strict=False)
    
    import json
    try:
        with open(args.dataset_config, 'r') as f:
            dataset_name_image_path = json.load(f)
        print(f"Loaded dataset configuration from: {args.dataset_config}")
    except FileNotFoundError:
        print(f"Dataset config file '{args.dataset_config}' not found. Please create it or specify a valid path.")
        return
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON config file '{args.dataset_config}': {e}")
        return

    calc_latent_tracer(vq_model, args, dataset_name_image_path, device)

if __name__ == "__main__":
    main()