import argparse
import json
import numpy as np
import optuna


def run_experiments(model_names, dataset_names, experiment_folder, pytorch_experiments, n_trials=20):
    for model_name in model_names:
        for dataset_name in dataset_names:
            with open(f"{experiment_folder}/{model_name}/{dataset_name}.json", "r") as file:
                data = json.load(file)

            # Basic configs
            seeds = data["seeds"]
            data_dir = data["data_dir"]
            output_parent_dir = data["output_parent_dir"]
            lr_scheduler = eval(data["lr_scheduler"])
            num_steps = data["num_steps"]
            print_steps = data["print_steps"]
            batch_size = data["batch_size"]
            metric = data["metric"]

            if model_name in ["LinOSS", "SpikingLinOSS", "SpikingLinOSSwGeLU", "SpikingLinOSSwBN", "SpikingLinOSSwHeterogeneity"]:
                linoss_discretization = data["linoss_discretization"]
            else:
                linoss_discretization = None

            if model_name == "SpikingLinOSSwHeterogeneity":
                heterogeneity_setup = data["heterogeneity_setup"]
            else:                
                heterogeneity_setup = None

            use_presplit = data["use_presplit"]
            T = data["T"]
            if model_name in ["lru", "S5", "S6", "mamba", "LinOSS", "SpikingLinOSS", "SpikingLinOSSwGeLU", "SpikingLinOSSwBN", "SpikingLinOSSwHeterogeneity", "pSpikeSSM","spikingssm","spikformer"]:
                dt0 = None
            else:
                dt0 = float(data["dt0"])

            scale = data["scale"]
            lr = float(data["lr"])
            include_time = data["time"].lower() == "true"
            hidden_dim = int(data["hidden_dim"])

            if model_name in ["log_ncde", "nrde", "ncde"]:
                vf_depth = int(data["vf_depth"])
                vf_width = int(data["vf_width"])
                if model_name in ["log_ncde", "nrde"]:
                    logsig_depth = int(data["depth"])
                    stepsize = int(float(data["stepsize"]))
                else:
                    logsig_depth = 1
                    stepsize = 1
                if model_name == "log_ncde":
                    lambd = float(data["lambd"])
                else:
                    lambd = None
                ssm_dim = None
                num_blocks = None
            else:
                vf_depth = None
                vf_width = None
                logsig_depth = 1
                stepsize = 1
                lambd = None
                ssm_dim = int(data["ssm_dim"])
                num_blocks = int(data["num_blocks"])

            if model_name == "S5":
                ssm_blocks = int(data["ssm_blocks"])
            else:
                ssm_blocks = None

            if dataset_name == "ppg":
                output_step = int(data["output_step"])
            elif dataset_name.startswith("ppg_") and dataset_name[4:].isdigit():
                output_step = int(data["output_step"])
            elif dataset_name == "ETTh1" or dataset_name == "ETTh2" or dataset_name == "ETTm1" or dataset_name == "ETTm2" or dataset_name == "ECL" or dataset_name == "traffic" or dataset_name == "weather" or dataset_name == "electricity" or dataset_name == "metr-la" or dataset_name == "pems-bay" or dataset_name == "solar-energy" or dataset_name == "exchange":
                output_step = int(data["output_step"]) * int(data["prediction_step"])
            else:
                output_step = 1
            
            if model_name == "mamba":
                conv_dim = int(data["convdim"])
                expansion = int(data["expansion"])
            if model_name == "S6":
                conv_dim = None
                expansion = None

            if "dropout" in data:
                dropout = (data["dropout"])
            else:
                dropout = 0.0

            if "kernel" in data:
                kernel = (data["kernel"])
            else:
                kernel = None
            if "early_stopping_steps" in data and (model_name == "mamba" or model_name == "S6"):
                early_stopping_steps = int(data["early_stopping_steps"])
            else: early_stopping_steps = None
            
            if pytorch_experiments:
                from torch_experiments.train import (
                    create_dataset_model_and_train as torch_create_dataset_model_and_train,
                )

                exps_n_samples = {
                    "EigenWorms": 236,
                    "EthanolConcentration": 524,
                    "Heartbeat": 409,
                    "MotorImagery": 378,
                    "SelfRegulationSCP1": 561,
                    "SelfRegulationSCP2": 380,
                    "ppg": 1232,
                    "signature1": 100000,
                    "signature2": 100000,
                    "signature3": 100000,
                    "signature4": 100000,
                }
                n_samples = exps_n_samples[dataset_name]

                model_args = {
                    "num_blocks": num_blocks,
                    "hidden_dim": hidden_dim,
                    "state_dim": ssm_dim,
                    # "lr": lr,
                    # "dropout": dropout,
                    "conv_dim": conv_dim if model_name == "mamba" or model_name == "S6" else None,
                    "expansion": expansion if model_name == "mamba" or model_name == "S6" else None,
                }
                run_args = {
                    "data_dir": data_dir,
                    "output_parent_dir": output_parent_dir,
                    "model_name": model_name,
                    "metric": metric,
                    "batch_size": batch_size,
                    "dataset_name": dataset_name,
                    "n_samples": n_samples,
                    "output_step": output_step,
                    "use_presplit": use_presplit,
                    "include_time": include_time,
                    "num_steps": num_steps,
                    "print_steps": print_steps,
                    "early_stopping_steps": early_stopping_steps if model_name == "mamba" or model_name == "S6" else None,
                    "lr": lr,
                    "model_args": model_args,
                }
                run_fn = torch_create_dataset_model_and_train
            else:
                import diffrax

                from train import create_dataset_model_and_train
                model_args = {
                    "num_blocks": num_blocks,
                    "hidden_dim": hidden_dim,
                    "vf_depth": vf_depth,
                    "vf_width": vf_width,
                    "ssm_dim": ssm_dim,
                    "ssm_blocks": ssm_blocks,
                    "dt0": dt0,
                    "scale": scale,
                    "lambd": lambd,
                }

                run_args = {
                    "data_dir": data_dir,
                    "use_presplit": use_presplit,
                    "dataset_name": dataset_name,
                    "output_step": output_step,
                    "metric": metric,
                    "include_time": include_time,
                    "T": T,
                    "model_name": model_name,
                    "stepsize": stepsize,
                    "logsig_depth": logsig_depth,
                    "scheme": linoss_discretization,
                    "dropout": dropout,
                    "model_args": model_args,
                    "num_steps": num_steps,
                    "print_steps": print_steps,
                    "lr": lr,
                    "lr_scheduler": lr_scheduler,
                    "batch_size": batch_size,
                    "output_parent_dir": output_parent_dir,
                    "kernel": kernel,
                    "heterogeneity_setup": heterogeneity_setup
                }
                run_fn = create_dataset_model_and_train

            # ---------------- OPTUNA OBJECTIVE ---------------- #
            def objective(trial):
                last_test_metrics = []
                last_test_sops = []
                for seed_idx, seed in enumerate(seeds):
                    print(f"[Trial {trial.number}] Seed {seed} - lr={lr}, blocks={num_blocks}, hidden_dim={hidden_dim}, ssm_dim={ssm_dim}, include_time={include_time}, scheme={linoss_discretization}, kernel={kernel}")

                    result = run_fn(seed=seed, **run_args)
                    if len(result) == 3:
                        _, test_metric, test_sops_avg = result
                    else:
                        _, test_metric = result
                        test_sops_avg = 0.0

                    if hasattr(test_metric, "item"):
                        test_metric = float(test_metric)
                    if hasattr(test_sops_avg, "item"):
                        test_sops_avg = float(test_sops_avg)
                    last_test_metrics.append(test_metric)
                    last_test_sops.append(test_sops_avg)

                    # Report after each seed finishes
                    mean_so_far = float(np.mean(last_test_metrics))
                    trial.report(mean_so_far, step=seed_idx)

                    if trial.should_prune():
                        raise optuna.TrialPruned()

                mean_test = float(np.mean(last_test_metrics))
                std_test = float(np.std(last_test_metrics))
                mean_sops = float(np.mean(last_test_sops))
                std_sops = float(np.std(last_test_sops))
                print(f"[Trial {trial.number}] Test metric mean: {mean_test:.6f}, std: {std_test:.6f}")
                print(f"[Trial {trial.number}] Test sops(avg) mean: {mean_sops:.2f}, std: {std_sops:.2f}")
                return mean_test

            # Pick direction automatically
            if metric == "mse":
                direction = "minimize"
            else:
                direction = "maximize"

            study = optuna.create_study(direction=direction)
            study.optimize(objective, n_trials=n_trials)

            print(f"\nBest trial for {model_name} on {dataset_name}:")
            print(study.best_trial)

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--dataset_name", type=str, default='EigenWorms')
    args.add_argument("--model_name", type=str, default='SpikingLinOSS')
    args = args.parse_args()
    
    pytorch_experiments = False

    if args.model_name == "LinOSS" or args.model_name == "mamba" or args.model_name == "S6" or args.model_name == "pSpikeSSM" or args.model_name == "spikingssm" or args.model_name == "spikformer":
        model_name = [args.model_name]
        pytorch_experiments = True
       
    else:
        # model_name = ["SpikingLinOSS"]
        model_name = [args.model_name]
        pytorch_experiments = False

    dataset_names = [
        args.dataset_name
    ]
    experiment_folder = "experiment_configs/repeats"

    run_experiments(model_name, dataset_names, experiment_folder, pytorch_experiments,n_trials=1)