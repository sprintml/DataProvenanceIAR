import random
import sys
import os
sys.path.append(os.path.dirname(os.getcwd()))  # Add parent directory to path
import torch
import numpy as np
import torch.nn.functional as F
# parse arguments
import argparse
from tokenizer.tokenizer_image.vq_model import VQ_models
from utils.dataset import CocoGroundTruthDataset
from tqdm import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def tokenize_and_reconstruct(vq_model, original_image,display_img=False, img_rec_quant_1=None, img_rec_no_quant_1=None):
    with torch.no_grad():
        if img_rec_quant_1 is not None:
            _, hidden_states, _, _ = vq_model.encode_with_internals(img_rec_quant_1.to(device))
            encoded_tokens, hidden_states, quantized_states, codebook_loss = vq_model.encode_with_internals(img_rec_quant_1.to(device))
            img_rec_quant = vq_model.decode(quantized_states)
            hidden_states = torch.einsum('b c h w -> b h w c', hidden_states).contiguous()
            hidden_states = F.normalize(hidden_states, p=2, dim=-1)
            hidden_states = torch.einsum('b h w c -> b c h w', hidden_states)
            sorcery = vq_model.decode(hidden_states)
            encoded_tokens, hidden_states, quantized_states, _ = vq_model.encode_with_internals(img_rec_no_quant_1.to(device))
            hidden_states = torch.einsum('b c h w -> b h w c', hidden_states).contiguous()
            hidden_states = F.normalize(hidden_states, p=2, dim=-1)
            hidden_states = torch.einsum('b h w c -> b c h w', hidden_states)
            img_rec_no_quant = vq_model.decode(hidden_states)
            img_rec_no_quant_loss_mse = torch.mean((img_rec_no_quant.cpu() - img_rec_no_quant_1.cpu()) ** 2, dim=[1, 2, 3])
            img_rec_quant_loss_mse = torch.mean((img_rec_quant.cpu() - img_rec_quant_1.cpu()) ** 2, dim=[1, 2, 3])
            sorcery = torch.mean((sorcery.cpu() - img_rec_no_quant_1.cpu()) ** 2, dim=[1, 2, 3])
        else:
            #image = load_image_to_tensor(original_image,device)
            encoded_tokens, hidden_states, quantized_states, codebook_loss = vq_model.encode_with_internals(original_image.unsqueeze(0).to(device))
            img_rec_quant = vq_model.decode(quantized_states)
            hidden_states = torch.einsum('b c h w -> b h w c', hidden_states).contiguous()
            hidden_states = F.normalize(hidden_states, p=2, dim=-1)
            hidden_states = torch.einsum('b h w c -> b c h w', hidden_states)
            img_rec_no_quant = vq_model.decode(hidden_states)
            sorcery = None
            img_rec_no_quant_loss_mse = torch.mean((img_rec_no_quant.cpu() - original_image.cpu()) ** 2, dim=[1, 2, 3])
            img_rec_quant_loss_mse = torch.mean((img_rec_quant.cpu() - original_image.cpu()) ** 2, dim=[1, 2, 3])

    return img_rec_quant, img_rec_no_quant, img_rec_quant_loss_mse, img_rec_no_quant_loss_mse, codebook_loss[0], sorcery





# calculate losses for any image dataset
def calculate_losses_dataset(vq_model, dataset, dataset_name,get_overlap=False):
    codebook_loss_mses, rec_quant_loss_mses, rec_no_quant_loss_mses = [], [], []
    codebook_loss_mses_double, rec_quant_loss_mses_double, rec_no_quant_loss_mses_double = [], [], []
    codebook_loss_mses_double_ratio, rec_quant_loss_mses_double_ratio, rec_no_quant_loss_mses_double_ratio = [], [], []
    sorcery_loss, sorcery_loss_ratio_list = [], []
    overlapping_ratios = []
    main_combined, sorcery_combined = [], []

    for image, _ in tqdm(dataset):
        img_rec_quant1, img_rec_no_quant1, img_rec_quant_mse, img_rec_no_quant_mse, codebook_loss_mse,_ = tokenize_and_reconstruct(vq_model, image, display_img=False)

        codebook_loss_mses.append(codebook_loss_mse.item())
        rec_quant_loss_mses.append(img_rec_quant_mse.item())
        rec_no_quant_loss_mses.append(img_rec_no_quant_mse.item())
        
        img_rec_quant2, img_rec_no_quant2,  img_rec_quant_mse_double, img_rec_no_quant_mse_double, codebook_loss_mse_double, sorcery = tokenize_and_reconstruct(vq_model, None, display_img=False, img_rec_quant_1=img_rec_quant1, img_rec_no_quant_1=img_rec_no_quant1)
        codebook_loss_mses_double.append(codebook_loss_mse_double.item())
        rec_no_quant_loss_mses_double.append(img_rec_no_quant_mse_double.item())
        rec_quant_loss_mses_double.append(img_rec_quant_mse_double.item())
        sorcery_loss.append(sorcery.item()) 
        sorcery_loss_ratio = img_rec_no_quant_mse/sorcery
        sorcery_loss_ratio_list.append(sorcery_loss_ratio.item())
        eps = 1e-10

        codebook_loss_mse_double_ratio = (codebook_loss_mse + eps) / (codebook_loss_mse_double + eps) 
        img_rec_quant_loss_mse_double_ratio = (img_rec_quant_mse + eps) / (img_rec_quant_mse_double + eps)
        img_rec_no_quant_loss_mse_double_ratio = (img_rec_no_quant_mse + eps) / (img_rec_no_quant_mse_double + eps)
        codebook_loss_mses_double_ratio.append(codebook_loss_mse_double_ratio.item())
        rec_quant_loss_mses_double_ratio.append(img_rec_quant_loss_mse_double_ratio.item())
        rec_no_quant_loss_mses_double_ratio.append(img_rec_no_quant_loss_mse_double_ratio.item())
        main_combined.append(img_rec_no_quant_loss_mse_double_ratio.item() * codebook_loss_mse.item())
        sorcery_combined.append(sorcery_loss_ratio.item() * codebook_loss_mse.item())
    if get_overlap:
        print(f"Average overlapping ratio for {dataset_name} images: {np.mean(overlapping_ratios)}")
    return codebook_loss_mses, codebook_loss_mses_double, codebook_loss_mses_double_ratio, rec_quant_loss_mses, rec_quant_loss_mses_double, rec_quant_loss_mses_double_ratio, rec_no_quant_loss_mses, rec_no_quant_loss_mses_double, rec_no_quant_loss_mses_double_ratio, sorcery_loss, sorcery_loss_ratio_list, sorcery_combined, main_combined


# List dir but with absolute path

import numpy as np
from sklearn import metrics

def calc_metrics(data_dict):

    # Prepare data
    datasets = list(data_dict.keys())
    data_arrays = [np.array(data_dict[key]) for key in datasets]
    # Calculate TPR@1%FPR for each dataset vs the last dataset
    
    
    if len(datasets) > 1:
        reference_data = data_arrays[-1]  # Last dataset as reference
        reference_name = datasets[-1]
        
        for i, (name, data) in enumerate(zip(datasets[:-1], data_arrays[:-1])):

            all_labels = np.concatenate([np.zeros(len(data)), np.ones(len(reference_data))])
            all_scores = np.concatenate([data, reference_data])
            all_scores_inverted = -all_scores

            # fpr, tpr, threshold = metrics.roc_curve(all_labels, all_scores)
            fpr, tpr, threshold_inverted = metrics.roc_curve(all_labels, all_scores_inverted)
            idx = np.where(fpr < 0.01)[0][-1]
            threshold_at_1fpr = -threshold_inverted[idx]
            tpr_at_1fpr = tpr[idx]


    stats_dict = {}

    
    for name, data in zip(datasets, data_arrays):
        stats = {
            'tpr_at_1fpr': tpr_at_1fpr,
            'mean': np.mean(data),
            'std': np.std(data),
            'min': np.min(data),
            'max': np.max(data),
            'median': np.median(data),
            'q25': np.percentile(data, 25),
            'q75': np.percentile(data, 75)
        }
        stats_dict[name] = stats
        
        
    
    return stats_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ft_path", type=str, default="", help="Path to the inversed decoder")
    parser.add_argument("--num_samples", type=int, default=1000, help="number of samples to evaluate")
    parser.add_argument("--dataset_config", type=str, default="dataset_config.json", help="Path to JSON file containing dataset name to path mapping")
    
    # Model configuration arguments

    parser.add_argument("--vq-model", type=str, choices=VQ_models.keys() if VQ_models else [], default="VQ-16", help="VQ Model architecture type")
    parser.add_argument("--vq-ckpt", type=str, default="", help="Path to the pre-trained VQ model checkpoint")
    parser.add_argument("--codebook-size", type=int, default=16384, help="Size of the codebook")
    parser.add_argument("--codebook-embed-dim", type=int, default=8, help="Dimension of codebook embeddings")
    parser.add_argument("--image_size", type=int, default=256, help="Image size")
    parser.add_argument("--downsample_size", type=int, default=16, help="Downsample size")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    
    args = parser.parse_args()
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    

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
    vq_model = VQ_models[args.vq_model](
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim)
    vq_model.to(device)
    vq_model.eval()

    checkpoint = torch.load(args.vq_ckpt, map_location=device, weights_only=False)
    vq_model.load_state_dict(checkpoint['model'], strict=False)
    if args.ft_path != "":
        checkpoint = torch.load(args.ft_path, map_location=device, weights_only=False)
        vq_model.encoder.load_state_dict(checkpoint['encoder_state_dict'], strict=False)
        print(f"Loaded inversed decoder from {args.ft_path}")
    vq_model.to(device)
    vq_model.eval()

    del checkpoint

    extended_map = {
        0: "Codebook Loss MSE",
        1: "Codebook Loss MSE Double",
        2: "Codebook Loss MSE Double Ratio",
        3: "Reconstruction Quant Loss MSE",
        4: "Reconstruction Quant Loss MSE Double",
        5: "Reconstruction Quant Loss MSE Double Ratio",
        6: "Reconstruction No Quant Loss MSE",
        7: "Reconstruction No Quant Loss MSE Double",
        8: "Reconstruction No Quant Loss MSE Double Ratio",
        9: "Sorcery Loss",
        10: 'Sorcery Loss Ratio',
        11: "Sorcery Combined",
        12: "Main Combined"
    }

    gen_path = dataset_name_image_path.get("LlamaGen Generated")
    gen_dataset = CocoGroundTruthDataset(gen_path, num_samples=args.num_samples)
    gen_results = calculate_losses_dataset(vq_model, gen_dataset, "gen")
    torch.save(gen_results, f'gen_results_finetuned.pt')
    for mode in range(len(gen_results)):
        mean = np.mean(gen_results[mode])
        std = np.std(gen_results[mode])
        print(f"Gen - {extended_map[mode]}: Mean = {mean:.8f}, Std = {std:.8f}")

    all_results = {}
    from PIL import Image
    for i,(dataset, path) in enumerate(dataset_name_image_path.items()):
        print(f"Dataset: {dataset}, Path: {path}")
        test_dataset = CocoGroundTruthDataset(path, num_samples=args.num_samples)
        test_results = calculate_losses_dataset(vq_model, test_dataset, dataset)
        torch.save(test_results, f'test_results_{dataset}.pt')
        print(f"Test results for {dataset}: {test_results}")

        all_results[dataset] = {}
        modes = range(len(test_results))
        for mode in modes:
            all_results[dataset][extended_map[mode]] = {}
            data_dict = {
                dataset: test_results[mode],
                'Generated': gen_results[mode],
            }

            stats = calc_metrics(
                data_dict=data_dict,
            )

            all_results[dataset][extended_map[mode]] = stats[dataset]["tpr_at_1fpr"].item()

            mean = np.mean(test_results[mode])
            std = np.std(test_results[mode])
            print(f"{dataset} - {extended_map[mode]}: Mean = {mean:.8f}, Std = {std:.8f}")
        print(all_results[dataset])

    import pandas as pd

    df = pd.DataFrame(all_results)

    latex_df = df.applymap(lambda x: f"{x*100:.1f}" if pd.notnull(x) else "--")
    print(latex_df.to_latex(escape=False, na_rep="--"))

if __name__ == "__main__":
    main()