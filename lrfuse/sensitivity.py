import os
import torch
import torch.nn as nn
from modules.svd_linear import SVDLinear
from evaluate import evaluate_model, evaluate_perplexity
from tqdm import tqdm
import numpy as np
import click

@torch.no_grad()
def calib_sensitivity_ppl(model, calib_loader, args, use_cache=True, step=0.1, act_aware=True):
    model_id = model.config._name_or_path
    cache_file = f"cache/lists/{model_id.replace('/','_')}_sensitivity_{args.scaling_method}_{args.alpha}_{args.n_calib_samples}_{args.calib_dataset}_step_{args.step}.pt"
    print(f"Search cache_file={cache_file}")
    if os.path.exists(cache_file) and use_cache:
        print(f"Load cache_file={cache_file}")
        sensitivity_dict = torch.load(cache_file, map_location="cpu")
        return sensitivity_dict
    model.eval()
    
    print(f"No cache_file={cache_file}")

    full_name_dict = {module: name for name, module in model.named_modules()}
    linear_info = {}
    modules = [model]
    while len(modules) > 0:
        submodule = modules.pop()
        for name, raw_linear in submodule.named_children():
            if isinstance(raw_linear, nn.Linear):
                full_name = full_name_dict[raw_linear]
                linear_info[raw_linear] = {
                    "father": submodule,
                    "name": name,
                    "full_name": full_name,
                }
            else:
                modules.append(raw_linear)

    sensitivity_dict = {}
    # param_ratio_candidates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # param_ratio_candidates = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
    # param_ratio_candidates = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, \
    #                           0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    # generate a list in range 0 to 1 with step 0.01
    param_ratio_candidates = np.arange(step, 1.0, step=step).tolist()
    # Round to 2 decimal places
    param_ratio_candidates = [round(_, 2) for _ in param_ratio_candidates]
    
    input_ids = torch.cat([_["input_ids"] for _ in calib_loader], 0)
    print(f"input_ids.shape={input_ids.shape}")
    pbar = tqdm(total=len(linear_info) * len(param_ratio_candidates))
    for raw_linear, info in linear_info.items():
        if info["full_name"] == "lm_head":
            continue
        sensitivity_dict[info["full_name"]] = {}
        for param_ratio in param_ratio_candidates:
            svd_linear = SVDLinear.from_linear(
                raw_linear,
                param_ratio=param_ratio,
                alpha=args.alpha,
                act_aware=act_aware,
            )
            setattr(info["father"], info["name"], svd_linear)

            ppl = evaluate_perplexity(model, input_ids, args.n_calib_samples)
            sensitivity_dict[info["full_name"]][param_ratio] = ppl
            print(f"{info['full_name']} {param_ratio} {ppl}")
            pbar.update(1)
        setattr(info["father"], info["name"], raw_linear)
    torch.save(sensitivity_dict, cache_file)
    return sensitivity_dict

@torch.no_grad()
def calib_sensitivity_ppl_rank_ratio(model, calib_loader, args, use_cache=True, step=0.1):
    model_id = model.config._name_or_path
    cache_file = f"cache/lists/{model_id.replace('/','_')}_sensitivity_rank_ratio_{args.scaling_method}_{args.alpha}_{args.n_calib_samples}_{args.calib_dataset}_step_{args.step}.pt"
    if os.path.exists(cache_file) and use_cache:
        sensitivity_dict = torch.load(cache_file, map_location="cpu")
        return sensitivity_dict
    model.eval()

    full_name_dict = {module: name for name, module in model.named_modules()}
    linear_info = {}
    modules = [model]
    while len(modules) > 0:
        submodule = modules.pop()
        for name, raw_linear in submodule.named_children():
            if isinstance(raw_linear, nn.Linear):
                full_name = full_name_dict[raw_linear]
                linear_info[raw_linear] = {
                    "father": submodule,
                    "name": name,
                    "full_name": full_name,
                }
            else:
                modules.append(raw_linear)

    sensitivity_dict = {}

    # generate a list in range 0 to 1 with step 0.01
    rank_ratio_candidates = np.arange(step, 1.0, step=step).tolist()
    # Round to 2 decimal places
    rank_ratio_candidates = [round(_, 2) for _ in rank_ratio_candidates]

    # param_ratio_candidates = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
    input_ids = torch.cat([_["input_ids"] for _ in calib_loader], 0)
    print(f"input_ids.shape={input_ids.shape}")
    pbar = tqdm(total=len(linear_info) * len(rank_ratio_candidates))
    for raw_linear, info in linear_info.items():
        if info["full_name"] == "lm_head":
            continue
        sensitivity_dict[info["full_name"]] = {}
        for ratio_ratio in rank_ratio_candidates:
            svd_linear = SVDLinear.from_linear_rank_ratio(
                raw_linear,
                rank_ratio=ratio_ratio,
                alpha=args.alpha,
                act_aware=True,
            )
            setattr(info["father"], info["name"], svd_linear)

            ppl = evaluate_perplexity(model, input_ids, args.n_calib_samples)
            sensitivity_dict[info["full_name"]][ratio_ratio] = ppl
            print(f"{info['full_name']} {ratio_ratio} {ppl}")
            pbar.update(1)
        setattr(info["father"], info["name"], raw_linear)
    torch.save(sensitivity_dict, cache_file)
    return sensitivity_dict

@torch.no_grad()
def calib_sensitivity_ppl_rank(model, calib_loader, args, use_cache=True, step=1):
    model_id = model.config._name_or_path
    cache_file = f"cache/lists/{model_id.replace('/','_')}_sensitivity_rank_{args.scaling_method}_{args.alpha}_{args.n_calib_samples}_{args.calib_dataset}_step_{args.step}.pt"
    if os.path.exists(cache_file) and use_cache:
        sensitivity_dict = torch.load(cache_file, map_location="cpu")
        return sensitivity_dict
    model.eval()

    full_name_dict = {module: name for name, module in model.named_modules()}
    linear_info = {}
    modules = [model]
    while len(modules) > 0:
        submodule = modules.pop()
        for name, raw_linear in submodule.named_children():
            if isinstance(raw_linear, nn.Linear):
                full_name = full_name_dict[raw_linear]
                linear_info[raw_linear] = {
                    "father": submodule,
                    "name": name,
                    "full_name": full_name,
                }
            else:
                modules.append(raw_linear)

    sensitivity_dict = {}
    
    rank_ratio_candidates = [ i for i in range(4086, 4096, 1) ] 

    # param_ratio_candidates = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
    input_ids = torch.cat([_["input_ids"] for _ in calib_loader], 0)
    print(f"input_ids.shape={input_ids.shape}")
    pbar = tqdm(total=len(linear_info) * len(rank_ratio_candidates))
    for raw_linear, info in linear_info.items():
        if info["full_name"] == "lm_head":
            continue
        sensitivity_dict[info["full_name"]] = {}
        for rank in rank_ratio_candidates:
            svd_linear = SVDLinear.from_linear_rank(
                raw_linear,
                rank=rank,
                alpha=args.alpha,
                act_aware=True,
            )
            setattr(info["father"], info["name"], svd_linear)

            ppl = evaluate_perplexity(model, input_ids, args.n_calib_samples)
            sensitivity_dict[info["full_name"]][rank] = ppl
            print(f"{info['full_name']} {rank} {ppl}")
            pbar.update(1)
        setattr(info["father"], info["name"], raw_linear)
    torch.save(sensitivity_dict, cache_file)
    return sensitivity_dict

@torch.no_grad()
def calib_sensitivity_ppl_rank_ratio_original_point(model, calib_loader, args, use_cache=True):
    model_id = model.config._name_or_path
    cache_file = f"cache/lists/{model_id.replace('/','_')}_sensitivity_original_point.pt"
    if os.path.exists(cache_file) and use_cache:
        sensitivity_dict = torch.load(cache_file, map_location="cpu")
        return sensitivity_dict
    model.eval()

    full_name_dict = {module: name for name, module in model.named_modules()}
    linear_info = {}
    modules = [model]
    while len(modules) > 0:
        submodule = modules.pop()
        for name, raw_linear in submodule.named_children():
            if isinstance(raw_linear, nn.Linear):
                full_name = full_name_dict[raw_linear]
                linear_info[raw_linear] = {
                    "father": submodule,
                    "name": name,
                    "full_name": full_name,
                }
            else:
                modules.append(raw_linear)

    sensitivity_dict = {}

    rank_ratio_candidates = [1.0]

    # param_ratio_candidates = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
    input_ids = torch.cat([_["input_ids"] for _ in calib_loader], 0)
    print(f"input_ids.shape={input_ids.shape}")
    pbar = tqdm(total=len(linear_info) * len(rank_ratio_candidates))
    for raw_linear, info in linear_info.items():
        if info["full_name"] == "lm_head":
            continue
        sensitivity_dict[info["full_name"]] = {}
        for ratio_ratio in rank_ratio_candidates:
            svd_linear = SVDLinear.from_linear_rank_ratio(
                raw_linear,
                rank_ratio=ratio_ratio,
                act_aware=True,
            )
            setattr(info["father"], info["name"], svd_linear)

            ppl = evaluate_perplexity(model, input_ids, args.n_calib_samples)
            sensitivity_dict[info["full_name"]][ratio_ratio] = ppl
            print(f"{info['full_name']} {ratio_ratio} {ppl}")
            pbar.update(1)
        setattr(info["father"], info["name"], raw_linear)
    torch.save(sensitivity_dict, cache_file)
    return sensitivity_dict


@torch.no_grad()
def calib_sensitivity_stable_rank(model, calib_loader, args, use_cache=True):
    model_id = model.config._name_or_path
    cache_file = f"cache/{model_id.replace('/','_')}_sensitivity_stable_rank_{args.scaling_method}_{args.alpha}_{args.n_calib_samples}_{args.calib_dataset}.pt"
    if os.path.exists(cache_file) and use_cache:
        sensitivity_dict = torch.load(cache_file, map_location="cpu")
        return sensitivity_dict
    model.eval()

    full_name_dict = {module: name for name, module in model.named_modules()}
    linear_info = {}
    modules = [model]
    while len(modules) > 0:
        submodule = modules.pop()
        for name, raw_linear in submodule.named_children():
            if isinstance(raw_linear, nn.Linear):
                full_name = full_name_dict[raw_linear]
                linear_info[raw_linear] = {
                    "father": submodule,
                    "name": name,
                    "full_name": full_name,
                }
            else:
                modules.append(raw_linear)

    sensitivity_dict = {}
    param_ratio_candidates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    input_ids = torch.cat([_["input_ids"] for _ in calib_loader], 0)
    print(f"input_ids.shape={input_ids.shape}")
    pbar = tqdm(total=len(linear_info) * len(param_ratio_candidates))
    for raw_linear, info in linear_info.items():
        sensitivity_dict[info["full_name"]] = {}

        # stable rank is defined to be the ratio between squared Frobenius norm and the squared spectral norm of a matrix
        w=raw_linear.weight
        w=w#*raw_linear.scaling_diag_matrix.view(1,-1)**args.alpha
        w_fro=torch.norm(w, p='fro')**2
        _,singular_values,_=torch.svd(w.float(),compute_uv=False)
        spectral_norm=torch.max(singular_values)
        w_spec=spectral_norm**2
        sr=(w_fro/w_spec)**0.5

        for param_ratio in param_ratio_candidates:
            sensitivity_dict[info["full_name"]][param_ratio] = -sr*param_ratio**0.1
            pbar.update(1)
    torch.save(sensitivity_dict, cache_file)
    return sensitivity_dict

@torch.no_grad()
def calib_sensitivity_vanilla_svd(model, calib_loader, args, use_cache=True, step=0.1, act_aware=False):
    model_id = model.config._name_or_path
    cache_file = f"cache/lists/{model_id.replace('/','_')}_sensitivity_{args.n_calib_samples}_{args.calib_dataset}_vanilla_svd_step_{args.step}.pt"
    if os.path.exists(cache_file) and use_cache:
        sensitivity_dict = torch.load(cache_file, map_location="cpu")
        return sensitivity_dict
    model.eval()

    full_name_dict = {module: name for name, module in model.named_modules()}
    linear_info = {}
    modules = [model]
    while len(modules) > 0:
        submodule = modules.pop()
        for name, raw_linear in submodule.named_children():
            if isinstance(raw_linear, nn.Linear):
                full_name = full_name_dict[raw_linear]
                linear_info[raw_linear] = {
                    "father": submodule,
                    "name": name,
                    "full_name": full_name,
                }
            else:
                modules.append(raw_linear)

    sensitivity_dict = {}
    # param_ratio_candidates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # param_ratio_candidates = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
    # param_ratio_candidates = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, \
    #                           0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    # generate a list in range 0 to 1 with step 0.01
    param_ratio_candidates = np.arange(step, 1.0, step=step).tolist()
    # Round to 2 decimal places
    param_ratio_candidates = [round(_, 2) for _ in param_ratio_candidates]
    
    input_ids = torch.cat([_["input_ids"] for _ in calib_loader], 0)
    print(f"input_ids.shape={input_ids.shape}")
    pbar = tqdm(total=len(linear_info) * len(param_ratio_candidates))
    for raw_linear, info in linear_info.items():
        if info["full_name"] == "lm_head":
            continue
        sensitivity_dict[info["full_name"]] = {}
        for param_ratio in param_ratio_candidates:
            svd_linear = SVDLinear.from_linear(
                raw_linear,
                param_ratio=param_ratio,
                alpha=args.alpha,
                act_aware=act_aware,
            )
            setattr(info["father"], info["name"], svd_linear)

            ppl = evaluate_perplexity(model, input_ids, args.n_calib_samples)
            sensitivity_dict[info["full_name"]][param_ratio] = ppl
            print(f"{info['full_name']} {param_ratio} {ppl}")
            pbar.update(1)
        setattr(info["father"], info["name"], raw_linear)
    torch.save(sensitivity_dict, cache_file)
    return sensitivity_dict

@torch.no_grad()
def get_calib_sensitivity(model, calib_loader, args, use_cache=True, step=0.1, act_aware=True):
    model_id = model.config._name_or_path
    
    # TODO: temporary fix. Sensitivity list should be saved in a different directory
    if args.method == "asvd":
        cache_file = f"cache/lists/{model_id.replace('/','_')}_sensitivity_{args.scaling_method}_{args.alpha}_{args.n_calib_samples}_{args.calib_dataset}_step_{args.step}.pt"
    else:
        cache_file = f"cache/lists/{model_id.replace('/','_')}_sensitivity_{args.method}_{args.scaling_method}_{args.alpha}_{args.n_calib_samples}_{args.calib_dataset}_step_{args.step}.pt"
    
    click.secho(f"Search cache_file={cache_file}", fg="yellow")
    if os.path.exists(cache_file) and use_cache:
        click.secho(f"File {cache_file} exist.", fg="green")
        click.secho(f"Load cache_file={cache_file}", fg="yellow")
        sensitivity_dict = torch.load(cache_file, map_location="cpu")
        return sensitivity_dict
    model.eval()
    
    click.secho(f"No cache_file={cache_file}", fg="red")

    click.secho(f"Create sensitivity list...", fg="yellow")

    full_name_dict = {module: name for name, module in model.named_modules()}
    linear_info = {}
    modules = [model]
    while len(modules) > 0:
        submodule = modules.pop()
        for name, raw_linear in submodule.named_children():
            if isinstance(raw_linear, nn.Linear):
                full_name = full_name_dict[raw_linear]
                linear_info[raw_linear] = {
                    "father": submodule,
                    "name": name,
                    "full_name": full_name,
                }
            else:
                modules.append(raw_linear)

    sensitivity_dict = {}
    
    # generate a list in range 0 to 1 with step 0.01
    param_ratio_candidates = np.arange(step, 1.0, step=step).tolist()
    # Round to 2 decimal places
    param_ratio_candidates = [round(_, 2) for _ in param_ratio_candidates]
    
    input_ids = torch.cat([_["input_ids"] for _ in calib_loader], 0)
    print(f"input_ids.shape={input_ids.shape}")
    pbar = tqdm(total=len(linear_info) * len(param_ratio_candidates))
    for raw_linear, info in linear_info.items():
        if info["full_name"] == "lm_head":
            continue
        sensitivity_dict[info["full_name"]] = {}
        for param_ratio in param_ratio_candidates:
            # Different methods implementation
            if args.method == "asvd":
                svd_linear = SVDLinear.from_linear(
                    raw_linear,
                    param_ratio=param_ratio,
                    alpha=args.alpha,
                    act_aware=True,
                )
            elif args.method == "whiten":
                svd_linear = SVDLinear.from_linear_whiten(
                    raw_linear,
                    param_ratio=param_ratio
                )
            setattr(info["father"], info["name"], svd_linear)

            ppl = evaluate_perplexity(model, input_ids, args.n_calib_samples)
            sensitivity_dict[info["full_name"]][param_ratio] = ppl
            print(f"{info['full_name']} {param_ratio} {ppl}")
            pbar.update(1)
        setattr(info["father"], info["name"], raw_linear)
    torch.save(sensitivity_dict, cache_file)
    click.secho(f"Save the sensitivity list to:  {cache_file}", fg="yellow")
    return sensitivity_dict