import numpy as np
import time
import torch
import torch.nn as nn
from scipy.stats import spearmanr

from data_provider.experiment_data import ExperimentData
from torch_utils import set_grad_flow_for_nn
from utilz import *


class TEMDecoderInferencer(nn.Module):
    
    def __init__(self, ebm, initial_tensor: torch.tensor):
        super().__init__()
        self.ebm = ebm
        self.y_hat = nn.Parameter(initial_tensor)

    def forward(self, batch_x, batch_x_mark, dec_inp, batch_y_mark, enc_out):
        energy = self.ebm._get_decoded(
            self.y_hat, batch_x, batch_x_mark, dec_inp, batch_y_mark, enc_out
        )
        return energy


class TEMInferencer:
    @classmethod
    def test_ebm_with_optimizer(
        cls,
        passed_ebm,
        device,
        enc_out: torch.Tensor,
        inference_optim_steps: int,  
        inference_optim_lr: float,
        initial_tensor: torch.Tensor,
        batch_x,
        batch_x_mark,
        dec_inp,
        batch_y_mark,
    ):
        
        step_count = inference_optim_steps
        learning_rate = inference_optim_lr

        set_grad_flow_for_nn(passed_ebm, False)

        opt_model = TEMDecoderInferencer(
            passed_ebm, initial_tensor=initial_tensor.clone()
        )
        opt_optimizer = torch.optim.Adam(opt_model.parameters(), lr=learning_rate)

        

        for i in range(step_count):
            opt_optimizer.zero_grad()
            loss = opt_model(batch_x, batch_x_mark, dec_inp, batch_y_mark, enc_out)

            
            

            loss.sum().backward()
            opt_optimizer.step()
            

        y_hat = opt_model.y_hat.detach()
        energy_hat = (
            opt_model(batch_x, batch_x_mark, dec_inp, batch_y_mark, enc_out)
            .squeeze()
            .detach()
            .cpu()
            .numpy()
        )

        set_grad_flow_for_nn(passed_ebm, True)

        return y_hat, energy_hat

    @classmethod
    def generate_many_points(
        cls,
        args,
        passed_ebm,
        device,
        enc_out: torch.Tensor,
        num_of_noise_samples: int,
        
        
        initial_tensor: torch.Tensor,
        batch_x,
        batch_x_mark,
        batch_y,
        dec_inp,
        batch_y_mark,
        sample_stds: [float] = None,
        inference_optim_steps=0,
    ):
        if sample_stds is None:
            sample_stds = [0.1, 0.2, 1.0]

        mse_energy_std_tuple_list = []

        
        for sample_std in sample_stds:
            for sample_idx in range(num_of_noise_samples):
                samples = torch.normal(
                    mean=torch.zeros(batch_y.shape[1], 1),
                    std=sample_std * torch.ones(batch_y.shape[1], 1),
                )
                samples = samples.to(device)
                noisy_batch_y = (
                    batch_y + samples
                )  
                (
                    noisy_optimized_batch_y,
                    energy_for_noisy_batch,
                ) = cls.test_ebm_with_optimizer(
                    passed_ebm=passed_ebm,
                    device=device,
                    enc_out=enc_out,
                    inference_optim_steps=inference_optim_steps,
                    inference_optim_lr=0.05,
                    initial_tensor=noisy_batch_y,
                    batch_x=batch_x,
                    batch_x_mark=batch_x_mark,
                    dec_inp=dec_inp,
                    batch_y_mark=batch_y_mark,
                )

                noisy_batch = (
                    noisy_batch_y
                    if inference_optim_steps == 0
                    else noisy_optimized_batch_y
                )
                mse_for_noisy_batch = torch.square(noisy_batch - batch_y)[
                    :, -args.pred_len :, :
                ].mean(dim=1)
                mse_for_noisy_batch = mse_for_noisy_batch.cpu().numpy()

                mse_energy_std_tuple_list.append(
                    [
                        mse_for_noisy_batch,
                        np.expand_dims(energy_for_noisy_batch, axis=1),
                        np.expand_dims(
                            np.repeat(sample_std, [batch_y.shape[0]]), axis=1
                        ),
                    ]
                )

        mse_stack = np.stack(
            list(map(lambda tuple: tuple[0], mse_energy_std_tuple_list))
        )
        energy_stack = np.stack(
            list(map(lambda tuple: tuple[1], mse_energy_std_tuple_list))
        )
        std_stack = np.stack(
            list(map(lambda tuple: tuple[2], mse_energy_std_tuple_list))
        )
        return mse_stack, energy_stack, std_stack

    @classmethod
    def test_adhoc_energy(
        cls,
        args,
        device,
        path_or_reference_of_model,
        experiment_data: ExperimentData,
        data_loader_name: str = "val",
        do_run_wide_experiments: bool = False,
        
        
        
        
        do_run_energy_optimization_experiments=True,
        override_dataset_tuples=None,
    ):
        print(f"Testing energy model with {data_loader_name} dataset")
        print(f"Dataloader_name = {data_loader_name}")
        
        print(f"Starting energy model evaluation")

        if data_loader_name not in ["train", "val", "test"]:
            raise ValueError(f"data_loader_name '{data_loader_name}' is not supported!")

        iter_count = 0

        if isinstance(path_or_reference_of_model, str):
            model = torch.load(path_or_reference_of_model, map_location=device)
        elif isinstance(path_or_reference_of_model, torch.nn.Module):
            model = path_or_reference_of_model
        else:
            raise ValueError(
                f"Type of passed model {type(path_or_reference_of_model)} is not supported!"
            )

        model.train()

        
        
        
        args.seq_len = model.autoformer.seq_len
        args.pred_len = model.autoformer.pred_len
        args.label_len = model.autoformer.label_len

        print(f"S:{args.seq_len}, P:{args.pred_len}, L:{args.label_len}")

        train_data, train_loader = (
            experiment_data.train_data,
            experiment_data.train_loader,
        )
        vali_data, vali_loader = experiment_data.val_data, experiment_data.val_loader
        test_data, test_loader = experiment_data.test_data, experiment_data.test_loader

        data, data_loader = None, None
        if data_loader_name == "train":
            data, data_loader = train_data, train_loader
        elif data_loader_name == "val":
            data, data_loader = vali_data, vali_loader
        elif data_loader_name == "test":
            data, data_loader = test_data, test_loader
        else:
            raise ValueError(
                f"Data loader '{data_loader_name}' unrecognized. Falling back to 'val'"
            )
            data, data_loader = vali_data, vali_loader

        print(f"<<<<<<<<<<<<<<<<<<<<<<<<<<")
        print(f"Train loader: {len(train_loader)}")
        vali_len = len(vali_loader)
        print(f"Vali loader: {vali_len}")
        test_len = len(test_loader)
        print(f"Test loader: {test_len}")
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>>")

        epoch_time = time.time()

        result_object = {}

        (
            list_of_energy_values_on_ground_truth,
            list_of_y_hat_init_zeros,
            list_of_y_hat_init_ground_truth,
            list_of_y_hat_init_orig_model,
            list_of_energy_hat_init_zeros,
            list_of_energy_hat_ground_truth,
            list_of_energy_hat_orig_model,
            list_of_batch_y,
            list_of_mse_orig,
        ) = ([], [], [], [], [], [], [], [], [])
        (
            list_of_noisy_mse,
            list_of_noisy_energies,
            list_of_noisy_std,
            list_of_noisy_durations,
        ) = ([], [], [], [])
        (
            list_of_noisy_mse_y_hat,
            list_of_noisy_energies_y_hat,
            list_of_noisy_std_y_hat,
            list_of_noisy_durations_y_hat,
        ) = ([], [], [], [])

        list_of_mse_orig_torch, list_of_mse_ebm_torch = [], []

        list_of_heatflex_U_vectors, list_of_heatflex_init_temp = [], []
        list_of_heatflex_flex_ground_5, list_of_heatflex_flex_pred_5 = [], []
        list_of_heatflex_flex_ground_10, list_of_heatflex_flex_pred_10 = [], []

        global ebm_optim_duration_dict  
        (
            ebm_optim_mae_y_hat_dict,
            ebm_optim_mse_dict,
            ebm_optim_energy_dict,
            ebm_optim_duration_dict,
        ) = ({}, {}, {}, {})

        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(data_loader):
            start_of_xy = DateUtils.now()
            iter_count += 1
            
            # Limit iterations in test mode - process just enough data to get meaningful results
            if hasattr(args, 'is_test_mode') and args.is_test_mode == 1 and iter_count > 10:
                print(f"TEST MODE: Limiting to {iter_count-1} data batches for faster execution")
                break

            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            f_dim = -1 if args.features == "MS" else 0
            target_batch_y = batch_y[:, :, f_dim:]

            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)

            
            
            dec_inp_orig = torch.zeros_like(batch_y[:, -args.pred_len :, :]).float()
            dec_inp_orig = (
                torch.cat([batch_y[:, : args.label_len, :], dec_inp_orig], dim=1)
                .float()
                .to(device)
            )

            y_hat_orig = model.autoformer(
                batch_x, batch_x_mark, dec_inp_orig, batch_y_mark
            )
            outputs_orig = y_hat_orig[:, -args.pred_len :, f_dim:]
            batch_y_orig = target_batch_y[:, -args.pred_len :, f_dim:].to(device)

            dec_inp = torch.zeros_like(target_batch_y[:, -args.pred_len :, :]).float()
            dec_inp = (
                torch.cat([target_batch_y[:, : args.label_len, :], dec_inp], dim=1)
                .float()
                .to(device)
            )

            mse_orig_calculated_by_torch = (
                torch.mean(torch.square(outputs_orig - batch_y_orig), axis=1)
                .squeeze()
                .detach()
                .cpu()
                .numpy()
            )
            

            enc_out, attns = model._forward_enc_orig(
                batch_x, batch_x_mark, dec_inp, batch_y_mark
            )
            energy = model._get_decoded(
                target_batch_y, batch_x, batch_x_mark, dec_inp, batch_y_mark, enc_out
            )

            list_of_energy_values_on_ground_truth.append(energy.detach().cpu().numpy())

            
            initial_guess_zeros = torch.zeros_like(target_batch_y).float().detach()
            initial_guess_ground_truth = target_batch_y.float().detach()
            
            
            initial_guess_orig_model = (
                torch.cat(
                    [
                        target_batch_y[:, : -outputs_orig.shape[1], :].to(device),
                        outputs_orig,
                    ],
                    dim=1,
                )
                .float()
                .detach()
            )

            type_of_experiment_name = "y_hat produced by EBM"
            type_of_experiment_name = "ground-truth data"
            type_of_experiment_name = "predictions from Autoformer"

            
            
            

            enc_out = enc_out.detach()

            def inverse_transform_batched_tensor(batched_xy_tensor):
                list_of_unscaled_samples = []
                for sample in batched_xy_tensor:
                    unscaled_sample = data.scaler.inverse_transform(sample.cpu())
                    list_of_unscaled_samples.append(unscaled_sample)
                return np.stack(list_of_unscaled_samples)

            if do_run_wide_experiments:
                start_of_wide_experiments = DateUtils.now()
                
                NUM_OF_NOISE_SAMPLES = 32
                SAMPLE_STDS = [0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
                (
                    noisy_mse_stack,
                    noisy_energy_stack,
                    noisy_std_stack,
                ) = cls.generate_many_points(
                    args=args,
                    passed_ebm=model,
                    device=device,
                    enc_out=enc_out,
                    num_of_noise_samples=NUM_OF_NOISE_SAMPLES,
                    initial_tensor=initial_guess_zeros,
                    batch_x=batch_x,
                    batch_x_mark=batch_x_mark,
                    dec_inp=dec_inp,
                    batch_y=target_batch_y,
                    batch_y_mark=batch_y_mark,
                    sample_stds=SAMPLE_STDS,
                )
                list_of_noisy_mse.append(noisy_mse_stack)
                list_of_noisy_energies.append(noisy_energy_stack)
                list_of_noisy_std.append(noisy_std_stack)
                end_of_noisy_generation = DateUtils.now()
                noisy_generation_batch_y_duration = (
                    end_of_noisy_generation - start_of_wide_experiments
                )
                list_of_noisy_durations.append(noisy_generation_batch_y_duration)

                (
                    noisy_mse_stack_y_hat,
                    noisy_energy_stack_y_hat,
                    noisy_std_stack_y_hat,
                ) = cls.generate_many_points(
                    args=args,
                    passed_ebm=model,
                    device=device,
                    enc_out=enc_out,
                    num_of_noise_samples=NUM_OF_NOISE_SAMPLES,
                    initial_tensor=initial_guess_zeros,
                    batch_x=batch_x,
                    batch_x_mark=batch_x_mark,
                    dec_inp=dec_inp,
                    batch_y=initial_guess_orig_model.detach(),
                    batch_y_mark=batch_y_mark,
                    sample_stds=SAMPLE_STDS,
                )
                list_of_noisy_mse_y_hat.append(noisy_mse_stack_y_hat)
                list_of_noisy_energies_y_hat.append(noisy_energy_stack_y_hat)
                list_of_noisy_std_y_hat.append(noisy_std_stack_y_hat)
                noisy_generation_y_hat_duration = (
                    DateUtils.now() - noisy_generation_batch_y_duration
                )
                list_of_noisy_durations_y_hat.append(noisy_generation_y_hat_duration)

            if do_run_energy_optimization_experiments:
                ENERGY_INFERENCE_OPTIM_LRS = [0.1, 0.01, 0.001]
                ENERGY_INFERENCE_STEPS = [5, 10, 25]
                for optim_lr in ENERGY_INFERENCE_OPTIM_LRS:
                    for steps in ENERGY_INFERENCE_STEPS:
                        start_optim_generation = DateUtils.now()
                        y_ebm_optim, energy_ebm_optim = cls.test_ebm_with_optimizer(
                            passed_ebm=model,
                            device=device,
                            enc_out=enc_out,
                            inference_optim_lr=optim_lr,
                            inference_optim_steps=steps,
                            initial_tensor=initial_guess_orig_model,
                            batch_x=batch_x,
                            batch_x_mark=batch_x_mark,
                            dec_inp=dec_inp,
                            batch_y_mark=batch_y_mark,
                        )
                        optim_generation_duration = (
                            DateUtils.now() - start_optim_generation
                        )
                        y_ebm_optim = y_ebm_optim[:, -args.pred_len :, :]
                        mse_ebm_optim = torch.mean(
                            torch.square(y_ebm_optim - batch_y_orig), dim=1
                        ).squeeze(dim=1)
                        
                        mae_y_hat_ebm_optim = (
                            torch.abs(y_ebm_optim - batch_y_orig)
                        ).squeeze(dim=1)
                        
                        key = f"s{steps}_e{int(np.abs(np.log10(([optim_lr]))))}"

                        if key not in ebm_optim_energy_dict:
                            ebm_optim_energy_dict[key] = []
                            ebm_optim_mae_y_hat_dict[key] = []
                            ebm_optim_mse_dict[key] = []
                            ebm_optim_duration_dict[key] = []

                        ebm_optim_energy_dict[key].append(energy_ebm_optim)
                        ebm_optim_mae_y_hat_dict[key].append(
                            torch.mean(mae_y_hat_ebm_optim, dim=1)
                            .squeeze()
                            .detach()
                            .cpu()
                            .numpy()
                        )
                        ebm_optim_mse_dict[key].append(
                            mse_ebm_optim.squeeze().detach().cpu().numpy()
                        )
                        ebm_optim_duration_dict[key].append(optim_generation_duration)

            y_hat_init_zeros, energy_hat_init_zeros = cls.test_ebm_with_optimizer(
                passed_ebm=model,
                device=device,
                enc_out=enc_out,
                inference_optim_lr=args.ebm_inference_optim_lr * 10,
                inference_optim_steps=25,
                initial_tensor=initial_guess_zeros,
                batch_x=batch_x,
                batch_x_mark=batch_x_mark,
                dec_inp=dec_inp,
                batch_y_mark=batch_y_mark,
            )
            y_hat_init_zeros = y_hat_init_zeros.cpu().numpy()
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            (
                y_hat_init_orig_no_optim,
                energy_hat_init_orig_no_optim,
            ) = cls.test_ebm_with_optimizer(
                passed_ebm=model,
                device=device,
                enc_out=enc_out,
                inference_optim_lr=args.ebm_inference_optim_lr * 10,
                inference_optim_steps=0,
                initial_tensor=initial_guess_orig_model,
                batch_x=batch_x,
                batch_x_mark=batch_x_mark,
                dec_inp=dec_inp,
                batch_y_mark=batch_y_mark,
            )
            y_hat_init_orig_no_optim = y_hat_init_orig_no_optim.cpu().numpy()
            (
                y_hat_init_ground_truth_no_optim,
                energy_hat_init_ground_truth_no_optim,
            ) = cls.test_ebm_with_optimizer(
                passed_ebm=model,
                device=device,
                enc_out=enc_out,
                inference_optim_lr=args.ebm_inference_optim_lr * 10,
                inference_optim_steps=0,
                initial_tensor=initial_guess_ground_truth,
                batch_x=batch_x,
                batch_x_mark=batch_x_mark,
                dec_inp=dec_inp,
                batch_y_mark=batch_y_mark,
            )
            y_hat_init_ground_truth_no_optim = (
                y_hat_init_ground_truth_no_optim.cpu().numpy()
            )

            
            

            enc_out = enc_out.detach()

            list_of_y_hat_init_zeros.append(y_hat_init_zeros)
            list_of_y_hat_init_ground_truth.append(y_hat_init_ground_truth_no_optim)
            list_of_y_hat_init_orig_model.append(y_hat_init_orig_no_optim)

            list_of_energy_hat_init_zeros.append(energy_hat_init_zeros)
            list_of_energy_hat_ground_truth.append(
                energy_hat_init_ground_truth_no_optim
            )
            list_of_energy_hat_orig_model.append(energy_hat_init_orig_no_optim)

            list_of_batch_y.append(target_batch_y.detach().cpu().numpy())
            list_of_mse_orig.append(
                torch.mean(torch.square(outputs_orig - batch_y_orig), dim=1)
                .squeeze()
                .detach()
                .cpu()
                .numpy()
            )
            list_of_mse_orig_torch.append(mse_orig_calculated_by_torch)
            

        if do_run_wide_experiments:
            noisy_energies = (
                np.stack(list_of_noisy_energies).squeeze().transpose(0, 2, 1)
            )
            noisy_mse = np.stack(list_of_noisy_mse).squeeze().transpose(0, 2, 1)
            noisy_std = np.stack(list_of_noisy_std).squeeze().transpose(0, 2, 1)
            noisy_durations = np.array(list_of_noisy_durations)

            noisy_energies = np.concatenate(noisy_energies)
            noisy_mse = np.concatenate(noisy_mse)
            noisy_std = np.concatenate(noisy_std)

            result_object["noisy_energies"] = noisy_energies
            result_object["noisy_mse"] = noisy_mse
            result_object["noisy_std"] = noisy_std
            result_object["noisy_durations"] = noisy_durations

            noisy_energies = np.concatenate(
                np.stack(list_of_noisy_energies_y_hat).squeeze().transpose(0, 2, 1)
            )
            noisy_mse = np.concatenate(
                np.stack(list_of_noisy_mse_y_hat).squeeze().transpose(0, 2, 1)
            )
            noisy_std = np.concatenate(
                np.stack(list_of_noisy_std_y_hat).squeeze().transpose(0, 2, 1)
            )
            noisy_durations = np.array(list_of_noisy_durations_y_hat)

            result_object["noisy_energies_y_hat"] = noisy_energies
            result_object["noisy_mse_y_hat"] = noisy_mse
            result_object["noisy_std_y_hat"] = noisy_std
            result_object["noisy_durations_y_hat"] = noisy_durations

        energy_values_on_ground_truth = np.concatenate(
            list_of_energy_values_on_ground_truth
        )

        y_hats_init_zeros = np.concatenate(list_of_y_hat_init_zeros)
        y_hats_init_ground_truth = np.concatenate(list_of_y_hat_init_ground_truth)
        y_hats_init_orig_model = np.concatenate(list_of_y_hat_init_orig_model)

        energy_hats_init_zeros = np.concatenate(list_of_energy_hat_init_zeros)
        energy_hats_init_ground_truth = np.concatenate(list_of_energy_hat_ground_truth)
        energy_hats_init_orig_model = np.concatenate(list_of_energy_hat_orig_model)

        batch_ys = np.concatenate(list_of_batch_y)

        

        batch_x_feature_shape = batch_x.shape[2]
        inverse_y_hats_init_orig_model = inverse_transform_batched_tensor(
            torch.tile(
                torch.from_numpy(y_hats_init_orig_model),
                dims=[1, 1, batch_x_feature_shape],
            )
        )[:, :, f_dim]
        inverse_batch_ys = inverse_transform_batched_tensor(
            torch.tile(torch.from_numpy(batch_ys), dims=[1, 1, batch_x_feature_shape])
        )[:, :, f_dim]
        inverse_mse = (
            torch.mean(
                torch.square(
                    torch.from_numpy(inverse_y_hats_init_orig_model)
                    - torch.from_numpy(inverse_batch_ys)
                ),
                dim=1,
            )
            .squeeze()
            .numpy()
        )

        result_object["energy_values_on_ground_truth"] = energy_values_on_ground_truth

        result_object["y_hats_init_zeros"] = y_hats_init_ground_truth
        result_object["y_hats_init_ground_truth"] = y_hats_init_ground_truth
        result_object["y_hats_init_orig_model"] = y_hats_init_orig_model

        result_object["energy_hats_init_zeros"] = energy_hats_init_zeros
        result_object["energy_hats_init_ground_truth"] = energy_hats_init_ground_truth
        result_object["energy_hats_init_orig_model"] = energy_hats_init_orig_model

        result_object["batch_ys"] = batch_ys

        result_object["inverse_y_hats_init_orig_model"] = inverse_y_hats_init_orig_model
        result_object["inverse_batch_ys"] = inverse_batch_ys
        result_object["inverse_mse"] = inverse_mse

        
        result_object["mse_init_zeros"] = np.mean(
            np.square(
                batch_ys[:, -args.pred_len :, :]
                - y_hats_init_zeros[:, -args.pred_len :, :]
            ),
            axis=1,
        ).squeeze()
        result_object["mse_init_ground_truth"] = np.mean(
            np.square(
                batch_ys[:, -args.pred_len :, :]
                - y_hats_init_ground_truth[:, -args.pred_len :, :]
            ),
            axis=1,
        ).squeeze()
        result_object["mse_init_orig_model"] = np.mean(
            np.square(
                batch_ys[:, -args.pred_len :, :]
                - y_hats_init_orig_model[:, -args.pred_len :, :]
            ),
            axis=1,
        ).squeeze()

        
        
        
        
        
        
        
        
        
        
        
        
        
        

        result_object["corr_init_orig_model"] = spearmanr(
            result_object["energy_hats_init_orig_model"],
            result_object["mse_init_orig_model"],
        )
        result_object["corr_init_ground_truth"] = spearmanr(
            result_object["energy_hats_init_ground_truth"],
            result_object["mse_init_ground_truth"],
        )
        result_object["corr_init_zeros"] = spearmanr(
            result_object["energy_hats_init_zeros"], result_object["mse_init_zeros"]
        )

        result_object["mse_orig"] = np.concatenate(list_of_mse_orig)
        result_object["metatitle"] = f"Experiments using {type_of_experiment_name}"
        result_object["mse_orig_torch"] = np.concatenate(list_of_mse_orig_torch)
        

        if do_run_energy_optimization_experiments:
            
            ebm_optim_mse_dict = {
                "ebm_optim_mse_" + str(key): np.concatenate(val)
                for key, val in ebm_optim_mse_dict.items()
            }
            ebm_optim_mae_y_hat_dict = {
                "ebm_optim_mae_y_hat_" + str(key): np.concatenate(val)
                for key, val in ebm_optim_mae_y_hat_dict.items()
            }
            ebm_optim_energy_dict = {
                "ebm_optim_energy_" + str(key): np.concatenate(val)
                for key, val in ebm_optim_energy_dict.items()
            }
            ebm_optim_duration_dict = {
                "ebm_optim_duration_" + str(key): np.array(val)
                for key, val in ebm_optim_duration_dict.items()
            }
            result_object.update(ebm_optim_energy_dict)
            result_object.update(ebm_optim_mae_y_hat_dict)
            result_object.update(ebm_optim_mse_dict)
            result_object.update(ebm_optim_duration_dict)

        print("MSE_init_zeros: ", result_object["mse_init_zeros"].mean())
        print("MSE_init_ground_truth: ", result_object["mse_init_ground_truth"].mean())
        print("MSE_init_orig_model: ", result_object["mse_init_orig_model"].mean())

        print("MSE orig: ", result_object["mse_orig_torch"].mean())

        print(f"Finished in: {time.time() - epoch_time}")

        return result_object, data
