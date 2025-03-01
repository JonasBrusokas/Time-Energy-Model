import numpy as np
import time
import torch
import torch.nn as nn
import warnings
from torch import optim

from data_provider.data_factory import get_data_from_provider
from data_provider.experiment_data import ExperimentData
from energy.EnergyModelsV3 import (
    AutoformerNeoEBM_concat,
    InformerNeoEBM_concat,
)
from energy.EnergyModels_fedformer import (
    FedformerNeoEBM_concat,
)
from energy.EnergyModels_patchtst import PatchTSTNeoEBM_concat
from energy.EnergyModels_timesnet import TimesNetNeoEBM_concat
from exp.exp_basic import Exp_Basic
from external_libs.clear_ml_wrapper import SFDB_logger
from models import (
    Autoformer,
    FEDformer,
    Informer,
    PatchTST,
    TimesNet,
)
from models.AbstractModel import AbstractModel
from tem_inferencer import TEMInferencer
from torch_utils import set_grad_flow_for_nn
from utils.tools import EarlyStopping
from utilz import *

warnings.filterwarnings("ignore")


class Exp_Main_Energy(Exp_Basic):
    static_logger = None
    def __init__(self, args, setting):
        super(Exp_Main_Energy, self).__init__(args)

        parsed_args = vars(args)
        self.setting = setting

        test_mode = not self.args.should_log
        sfdb_logger = SFDB_logger(
            args=parsed_args,
            project_name=self.args.name,
            test_mode=test_mode,
            path=os.path.join(
                Path(self.get_full_ebm_parent_path(self.setting)), "log.db"
            ),
        )
        self.logger = sfdb_logger.get_logger()
        self.clear_ml_wrapper = sfdb_logger

        self.test_mode = False
        try:
            self.test_mode = True if (self.args.is_test_mode == 1) else False
            if self.test_mode:
                print(f"[\n[TEST MODE ENABLED]\n]")
                # Reduce epochs for faster testing
                self.args.train_epochs = 3
                self.args.ebm_epochs = 3
                print(f"Reduced training epochs to {self.args.train_epochs} and EBM epochs to {self.args.ebm_epochs}")
        except Exception as e:
            print(
                f"There was an exception setting test_mode. Default value is: {self.test_mode}"
            )

    def set_args(self, args):
        print(f"IMPORTANT! Changing value of the args")
        self.args = args

    def calculate_seed_postfix(self):
        
        seed_postfix = (
            f"_{self.args.ebm_seed}" if self.args.ebm_seed not in [42, 2021] else ""
        )
        return seed_postfix

    def _experiment_path(self, setting) -> str:
        return os.path.join(str(self.args.checkpoints), setting)

    def _model_checkpoint_path(self, setting) -> str:
        
        seed_postfix = self.calculate_seed_postfix()
        model_checkpoint_path = os.path.join(
            self._experiment_path(setting), f"checkpoint{seed_postfix}.pth"
        )
        return model_checkpoint_path

    def _build_model(self):
        model_dict = {
            "Autoformer": Autoformer,
            "Informer": Informer,
            "FEDformer": FEDformer,
            "TimesNet": TimesNet,
            "PatchTST": PatchTST,
        }
        if self.args.model not in model_dict.keys():
            print(f"**WARNING** model with name '{self.args.model}' not found!")
            return None
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(
        self,
        flag,
        override_batch_size=None,
        override_data_path=None,
        override_scaler=None,
        override_target_site_id=None,
        override_dataset_tuples=None,
    ):
        data_set, data_loader = get_data_from_provider(
            args=self.args,
            flag=flag,
            override_batch_size=override_batch_size,
            override_data_path=override_data_path,
            override_scaler=override_scaler,
            override_target_site_id=override_target_site_id,
            override_dataset_tuples=override_dataset_tuples,
        )
        return data_set, data_loader

    def _select_optimizer(self, model=None):
        if model is None:
            model = self.model
        model_optim = optim.Adam(model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                vali_loader
            ):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len :, :]).float()
                dec_inp = (
                    torch.cat([batch_y[:, : self.args.label_len, :], dec_inp], dim=1)
                    .float()
                    .to(self.device)
                )
                
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if "DLinear" in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(
                                    batch_x, batch_x_mark, dec_inp, batch_y_mark
                                )[0]
                            else:
                                outputs = self.model(
                                    batch_x, batch_x_mark, dec_inp, batch_y_mark
                                )
                else:
                    if "DLinear" in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(
                                batch_x, batch_x_mark, dec_inp, batch_y_mark
                            )[0]
                        else:
                            outputs = self.model(
                                batch_x, batch_x_mark, dec_inp, batch_y_mark
                            )
                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[:, -self.args.pred_len :, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting, experiment_data):
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        ORIG_DONE_MARKER_FILE_NAME = f"orig_done{self.calculate_seed_postfix()}.txt"
        orig_done_full_path = os.path.join(path, ORIG_DONE_MARKER_FILE_NAME)
        time_now = time.time()

        train_steps = len(experiment_data.train_loader)
        early_stopping = EarlyStopping(
            patience=self.args.patience,
            verbose=True,
            override_model_path=self._model_checkpoint_path(setting),
        )

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        log_output_model = True

        if self.args.ebm_training_strategy == "cd_only":
            epochs = 0
            print(
                f"TRAIN! CD_only, not training. Setting epochs={epochs} and RETURNING EARLY"
            )
            return self.model
        else:
            epochs = self.args.train_epochs

        before_orig_train = DateUtils.now()
        for epoch in range(epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()

            if epoch == 0:
                print(f"TRAIN loader size: {len(experiment_data.train_loader)}")

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                experiment_data.train_loader
            ):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len :, :]).float()
                dec_inp = (
                    torch.cat([batch_y[:, : self.args.label_len, :], dec_inp], dim=1)
                    .float()
                    .to(self.device)
                )

                

                
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if "DLinear" in self.args.model:
                            outputs = self.model(batch_x)
                            if log_output_model:
                                
                                log_output_model = False
                        else:
                            if self.args.output_attention:
                                outputs = self.model(
                                    batch_x, batch_x_mark, dec_inp, batch_y_mark
                                )[0]
                                if log_output_model:
                                    
                                    log_output_model = False
                            else:
                                outputs = self.model(
                                    batch_x, batch_x_mark, dec_inp, batch_y_mark
                                )
                                if log_output_model:
                                    
                                    log_output_model = False

                        f_dim = -1 if self.args.features == "MS" else 0
                        outputs = outputs[:, -self.args.pred_len :, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(
                            self.device
                        )
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if "DLinear" in self.args.model:
                        outputs = self.model(batch_x)
                        if log_output_model:
                            
                            log_output_model = False
                    else:
                        if self.args.output_attention:
                            outputs = self.model(
                                batch_x, batch_x_mark, dec_inp, batch_y_mark
                            )[0]
                            if log_output_model:
                                
                                log_output_model = False
                        else:
                            outputs = self.model(
                                batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y
                            )
                            if log_output_model:
                                
                                log_output_model = False

                    
                    f_dim = -1 if self.args.features == "MS" else 0
                    outputs = outputs[:, -self.args.pred_len :, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print(
                        "\titers: {0}, epoch: {1}/{2} | loss: {3:.7f}".format(
                            i + 1, epoch + 1, epochs, loss.item()
                        )
                    )
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((epochs - epoch) * train_steps - i)
                    print(
                        "\tspeed: {:.4f}s/iter; left time: {:.4f}s".format(
                            speed, left_time
                        )
                    )

                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)

            self.logger.report_scalar(
                title=f"Training loss",
                series="MSE",
                value=float(train_loss),
                iteration=epoch + 1,
            )

            vali_loss = self.vali(
                experiment_data.val_data, experiment_data.val_loader, criterion
            )
            test_loss = self.vali(
                experiment_data.test_data, experiment_data.test_loader, criterion
            )

            self.logger.report_scalar(
                title=f"Validation loss",
                series="MSE",
                value=float(vali_loss),
                iteration=epoch + 1,
            )

            self.logger.report_scalar(
                title=f"Testing loss",
                series="MSE",
                value=float(test_loss),
                iteration=epoch + 1,
            )

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss
                )
            )
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            print(f"No learning rate adjustment for model  {self.args.model}")

        after_orig_train = DateUtils.now()
        print(f"ORIG TRAIN: {after_orig_train - before_orig_train}")

        
        from torch_utils import count_parameters

        pre_parameters_count = count_parameters(self.model)

        
        best_model_path = self._model_checkpoint_path(setting)
        self.model.load_state_dict(torch.load(best_model_path))

        post_parameters_count = count_parameters(self.model)

        with open(orig_done_full_path, mode="w") as report_file:
            report_file.write(f"{1}")

        return self.model

    
    def get_dec_inp(self, batch_y):
        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len :, :]).float()
        dec_inp = (
            torch.cat([batch_y[:, : self.args.label_len, :], dec_inp], dim=1)
            .float()
            .to(self.device)
        )
        return dec_inp

    def get_deterministic_test_output_full_path(self, setting):
        return os.path.join(self.get_full_ebm_parent_path(setting), "det_test_outputs")

    def test(self, setting, experiment_data: ExperimentData, test=0):
        test_data, test_loader = experiment_data.test_data, experiment_data.test_loader

        if test:
            print("loading model")
            device = torch.device("cuda" if self.args.use_gpu else "cpu")
            self.model.load_state_dict(
                torch.load(self._model_checkpoint_path(setting), map_location=device)
            )

        preds = []
        trues = []
        inputx = []
        folder_path = self.get_deterministic_test_output_full_path(setting)
        FileUtils.create_dir(folder_path)
        log_output_model = True

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                test_loader
            ):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                
                dec_inp = self.get_dec_inp(batch_y)

                
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if "DLinear" in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(
                                    batch_x, batch_x_mark, dec_inp, batch_y_mark
                                )[0]
                            else:
                                outputs = self.model(
                                    batch_x, batch_x_mark, dec_inp, batch_y_mark
                                )
                else:
                    if "DLinear" in self.args.model:
                        outputs = self.model(batch_x)
                        if log_output_model:
                            log_output_model = False
                    else:
                        if self.args.output_attention:
                            outputs = self.model(
                                batch_x, batch_x_mark, dec_inp, batch_y_mark
                            )[0]
                            if log_output_model:
                                log_output_model = False
                        else:
                            outputs = self.model(
                                batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y
                            )
                            if log_output_model:
                                log_output_model = False

                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[:, -self.args.pred_len :, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())

        if self.args.test_flop:
            exit()
        preds = np.array(preds)
        trues = np.array(trues)
        inputx = np.array(inputx)

        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        inputx = inputx.reshape(-1, inputx.shape[-2], inputx.shape[-1])

        from utils.metrics import metric

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        self.logger.report_scalar(
            "metrics",
            "mse",
            mse,
        )
        self.logger.report_scalar(
            "metrics",
            "mae",
            mae,
        )
        self.logger.report_scalar("metrics", "rmse", rmse)
        self.logger.report_scalar("metrics", "mape", mape)
        self.logger.report_scalar("metrics", "mspe", mspe)
        self.logger.report_scalar("metrics", "rse", rse)
        
        

        
        np.save(os.path.join(folder_path, "pred.npy"), preds)
        np.save(os.path.join(folder_path, "true.npy"), trues)
        np.save(os.path.join(folder_path, "x.npy"), inputx)
        
        
        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag="pred")

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + "/" + "checkpoint.pth"
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                pred_loader
            ):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                
                dec_inp = (
                    torch.zeros(
                        [batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]
                    )
                    .float()
                    .to(batch_y.device)
                )
                dec_inp = (
                    torch.cat([batch_y[:, : self.args.label_len, :], dec_inp], dim=1)
                    .float()
                    .to(self.device)
                )
                
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if "DLinear" in self.args.model:
                            outputs = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                outputs = self.model(
                                    batch_x, batch_x_mark, dec_inp, batch_y_mark
                                )[0]
                            else:
                                outputs = self.model(
                                    batch_x, batch_x_mark, dec_inp, batch_y_mark
                                )
                else:
                    if "DLinear" in self.args.model:
                        outputs = self.model(batch_x)
                    else:
                        if self.args.output_attention:
                            outputs = self.model(
                                batch_x, batch_x_mark, dec_inp, batch_y_mark
                            )[0]
                        else:
                            outputs = self.model(
                                batch_x, batch_x_mark, dec_inp, batch_y_mark
                            )
                pred = outputs.detach().cpu().numpy()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        
        folder_path = "./results/" + setting + "/"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + "real_prediction.npy", preds)

        return

    def get_ebm_setting(self) -> str:
        ebm_setting = ""
        args_dict = vars(self.args)
        for key in args_dict:
            if not key.startswith("ebm"):
                continue
            ebm_setting += str(args_dict[key])
        return ebm_setting

    def get_path(self, setting) -> str:
        path = os.path.join(self.args.checkpoints, setting)
        return path

    def get_ebm_parent_path(self, setting) -> str:
        args_dict = vars(self.args)
        ebm_parent_path = os.path.join(self.get_path(setting), args_dict["name"])
        return ebm_parent_path

    def get_full_ebm_path(self, setting):
        ebm_parent_path = self.get_ebm_parent_path(setting)
        ebm_setting = self.get_ebm_setting()
        ebm_path = os.path.join(ebm_parent_path, ebm_setting)
        ebm_full_path = full_ebm_path = os.path.join(ebm_path, "full_ebm.pth")
        return ebm_full_path

    def get_full_ebm_parent_path(self, setting) -> str:
        return str(Path(self.get_full_ebm_path(setting)).parent)

    def get_adjusted_y(self, y):
        f_dim = -1 if self.args.features == "MS" else 0
        return y[:, -self.args.pred_len :, f_dim:]

    def train_energy(
        self,
        setting,
        experiment_data: ExperimentData,
    ):
        pretrained_model: AbstractModel = (
            self.model
        )  

        global ebm
        ebm = None
        if isinstance(pretrained_model, Autoformer.Model):
            if self.args.ebm_model_name == "mlp_concat":
                ebm = AutoformerNeoEBM_concat(setting, self.model, 16, 1)
            else:
                print(
                    f" <<< UNSUPPORTED 'AUTOFORMER' EBM MODEL TYPE {self.args.ebm_model_name} >>> "
                )
                print(f"DEFAULTING TO 'mlp_concat'")
                ebm = AutoformerNeoEBM_concat(setting, self.model, 16, 1)
        elif isinstance(pretrained_model, Informer.Model):
            if self.args.ebm_model_name == "informer_mlp_concat":
                ebm = InformerNeoEBM_concat(setting, self.model, 16, 1)
            else:
                print(
                    f" <<< UNSUPPORTED 'INFORMER' EBM MODEL TYPE {self.args.ebm_model_name} >>> "
                )
                print(f"DEFAULTING TO 'mlp_concat'")
                ebm = InformerNeoEBM_concat(setting, self.model, 16, 1)
        elif isinstance(pretrained_model, FEDformer.Model):
            if self.args.ebm_model_name == "fedformer_mlp_concat":
                ebm = FedformerNeoEBM_concat(setting, self.model, 16, 1)
            else:
                print(
                    f" <<< UNSUPPORTED 'FEDFORMER' EBM MODEL TYPE {self.args.ebm_model_name} >>> "
                )
                print(f"DEFAULTING TO 'mlp_concat'")
                ebm = FedformerNeoEBM_concat(setting, self.model, 16, 1)
        elif isinstance(pretrained_model, TimesNet.Model):
            if self.args.ebm_model_name == "times_net_mlp_concat":
                ebm = TimesNetNeoEBM_concat(setting, self.model, 16, 1)
            else:
                print(
                    f" <<< UNSUPPORTED 'TIMES NET' EBM MODEL TYPE {self.args.ebm_model_name} >>> "
                )
                print(f"DEFAULTING TO 'mlp_concat'")
                ebm = TimesNetNeoEBM_concat(setting, self.model, 16, 1)
        elif isinstance(pretrained_model, PatchTST.Model):
            print(f"PatchTST EBM! Woop Woop!")
            patch_tst_ebm = PatchTSTNeoEBM_concat(setting, self.model, 16, 1)
            if self.args.ebm_model_name == "patch_tst_mlp_concat":
                ebm = patch_tst_ebm
            else:
                ebm = patch_tst_ebm
        else:
            raise ValueError("UNSUPPORTED MODEL TYPE")

        
        
        
        
        

        before_full_ebm_train = DateUtils.now()

        
        
        
        for _, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
            experiment_data.train_loader
        ):
            
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)
            batch_x_mark = batch_x_mark.float().to(self.device)
            batch_y_mark = batch_y_mark.float().to(self.device)
            dec_inp = self.get_dec_inp(batch_y)

            
            
            
            

            
            
            enc_out, attns = ebm._forward_enc_orig(
                batch_x, batch_x_mark, dec_inp, batch_y_mark
            )
            assumed_enc_out_shape = ebm.get_assumed_enc_out_shape(self.args)
            enc_shape_assumption = list(enc_out.shape) == assumed_enc_out_shape

            
            dec_out = ebm._forward_dec_orig(
                batch_x, batch_x_mark, dec_inp, batch_y_mark, enc_out
            )
            assumed_dec_out_shape = ebm.get_assumed_dec_out_shape(self.args)
            dec_shape_assumption = list(dec_out.shape) == assumed_dec_out_shape
            dec_out_2_dim_value = list(dec_out.shape)[1] if self.args.model not in ["PatchTST"] else self.args.enc_in
            
            ebm.setup_y_encoder_and_xy_decoder_(
                seq_len=self.args.seq_len,
                label_len=self.args.label_len,
                pred_len=self.args.pred_len,
                d_model=self.args.d_model,
                c_out=self.args.c_out,
                dec_out_2_dim=dec_out_2_dim_value,
            )
            ebm = ebm.to(self.device)
            ebm.autoformer = ebm.autoformer.to(self.device)
            ebm.y_encoder = ebm.y_encoder.to(self.device)
            ebm.xy_decoder = ebm.xy_decoder.to(self.device)

            
            f_dim = -1 if self.args.features == "MS" else 0
            target_batch_y = batch_y[:, :, f_dim:]

            encoded_y = ebm._forward_y_enc(batch_y=target_batch_y)
            dec_out_for_y_encoder = ebm._forward_dec_orig(
                batch_x, batch_x_mark, dec_inp, batch_y_mark, encoded_y
            )
            ebm._get_decoded(
                target_batch_y, batch_x, batch_x_mark, dec_inp, batch_y_mark, enc_out
            )

            print(f"Setup for Autoformer done!")
            break

        
        
        

        ebm.train()
        set_grad_flow_for_nn(ebm.autoformer, False)
        set_grad_flow_for_nn(ebm.y_encoder, True)
        set_grad_flow_for_nn(ebm.xy_decoder, False)

        path = self.get_path(setting)
        if not os.path.exists(path):
            os.makedirs(path)

        ebm_setting = self.get_ebm_setting()
        encoded_file_name_bytes = base64.urlsafe_b64encode(ebm_setting.encode("utf-8"))
        encoded_file_name = encoded_file_name_bytes.decode("utf-8")

        
        args_dict = vars(self.args)
        ebm_parent_path = self.get_ebm_parent_path(setting)
        if not os.path.exists(ebm_parent_path):
            os.makedirs(ebm_parent_path)

        print(f">>> Experiment parent path: {ebm_parent_path}")

        ebm_path = os.path.join(ebm_parent_path, ebm_setting)
        if not os.path.exists(ebm_path):
            os.makedirs(ebm_path)

        print(f">>> Experiment full path: {ebm_path}")

        if self.args.ebm_training_strategy == "one_two_one_two_on_top_of_orig":
            print("Overriding 'one_two_one_two_on_top_of_orig' for COMPATIBILITY!")
            self.args.ebm_training_strategy = "neo_ebm_one_two_three_cd"

        # Set up ebm training strategy handling
        if self.args.ebm_training_strategy == "neo_ebm_one_two_three_cd":
            
            y_encoder_train_epochs_initial = self.args.ebm_epochs
            y_encoder_train_epochs = y_encoder_train_epochs_initial  

            Y_ENC_DONE_MARKER_FILE_NAME = "y_enc_done.txt"
            y_enc_done_full_path = os.path.join(ebm_path, Y_ENC_DONE_MARKER_FILE_NAME)
            if Path(y_enc_done_full_path).exists():
                print("FOUND TRAINED Y_ENC MODEL!")
                
                
                

            ebm_full_path = full_ebm_path = os.path.join(ebm_path, "full_ebm.pth")
            if Path(ebm_full_path).exists() and not self.args.force_retrain_y_enc:
                print("FOUND TRAINED XY_EBM MODEL! Will not train y_enc")
                y_encoder_train_epochs = 0

            time_now = time.time()

            train_steps = len(experiment_data.train_loader)
            early_stopping = EarlyStopping(
                patience=self.args.patience, verbose=True, model_name_postfix="y_enc"
            )

            model = ebm.y_encoder
            model_optim = self._select_optimizer(model)
            criterion = self._select_criterion()

            if self.args.use_amp:
                scaler = torch.cuda.amp.GradScaler()

            log_output_model = True

            def validate_y_encoder(data_loader):
                y_encoder_validation_loss_list = []
                before_validation = time.time()

                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                    data_loader
                ):
                    model_optim.zero_grad()
                    batch_x = batch_x.float().to(self.device)

                    batch_y = batch_y.float().to(self.device)
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                    
                    dec_inp = torch.zeros_like(
                        batch_y[:, -self.args.pred_len :, :]
                    ).float()
                    dec_inp = (
                        torch.cat(
                            [batch_y[:, : self.args.label_len, :], dec_inp], dim=1
                        )
                        .float()
                        .to(self.device)
                    )

                    encoded_y = ebm._forward_y_enc(batch_y)
                    outputs = ebm._forward_dec_orig(
                        batch_x, batch_x_mark, dec_inp, batch_y_mark, encoded_y
                    )

                    f_dim = -1 if self.args.features == "MS" else 0
                    outputs = outputs[:, -self.args.pred_len :, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    y_encoder_validation_loss_list.append(loss.item())

                y_encoder_validation_loss = np.average(y_encoder_validation_loss_list)
                return y_encoder_validation_loss

            

            for epoch in range(y_encoder_train_epochs):
                
                
                iter_count = 0
                train_loss = []

                model.train()
                epoch_time = time.time()

                if epoch == 0:
                    print(f"Y ENCODER TRAIN loader size: {len(y_encoder_train_epochs)}")

                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                    experiment_data.train_loader
                ):
                    iter_count += 1
                    model_optim.zero_grad()
                    batch_x = batch_x.float().to(self.device)

                    batch_y = batch_y.float().to(self.device)
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                    
                    dec_inp = torch.zeros_like(
                        batch_y[:, -self.args.pred_len :, :]
                    ).float()
                    dec_inp = (
                        torch.cat(
                            [batch_y[:, : self.args.label_len, :], dec_inp], dim=1
                        )
                        .float()
                        .to(self.device)
                    )

                    encoded_y = ebm._forward_y_enc(batch_y)
                    outputs = ebm._forward_dec_orig(
                        batch_x, batch_x_mark, dec_inp, batch_y_mark, encoded_y
                    )

                    

                    
                    f_dim = -1 if self.args.features == "MS" else 0
                    outputs = outputs[:, -self.args.pred_len :, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                    if (i + 1) % 100 == 0:
                        print(
                            "\ty_encoder | iters: {0}, epoch: {1} | loss: {2:.7f}".format(
                                i + 1, epoch + 1, loss.item()
                            )
                        )
                        speed = (time.time() - time_now) / iter_count
                        left_time = speed * (
                            (self.args.ebm_epochs - epoch) * train_steps - i
                        )
                        print(
                            "\ty_encoder | speed: {:.4f}s/iter; left time: {:.4f}s".format(
                                speed, left_time
                            )
                        )

                        iter_count = 0
                        time_now = time.time()

                    loss.backward()
                    model_optim.step()

                

                print(
                    "y_encoder | epoch: {} cost time: {}".format(
                        epoch + 1, time.time() - epoch_time
                    )
                )
                train_loss = np.average(train_loss)

                self.logger.report_scalar(
                    title=f"Y_ENCODER Training loss",
                    series="MSE",
                    value=float(train_loss),
                    iteration=epoch + 1,
                )

                vali_loss = validate_y_encoder(experiment_data.val_loader)
                test_loss = validate_y_encoder(experiment_data.test_loader)

                self.logger.report_scalar(
                    title=f"Y_ENCODER Validation loss",
                    series="MSE",
                    value=float(vali_loss),
                    iteration=epoch + 1,
                )

                self.logger.report_scalar(
                    title=f"Y_ENCODER Testing loss",
                    series="MSE",
                    value=float(test_loss),
                    iteration=epoch + 1,
                )

                print(
                    "Y_ENCODER | Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {"
                    "4:.7f}".format(
                        epoch + 1, train_steps, train_loss, vali_loss, test_loss
                    )
                )
                early_stopping(vali_loss, model, ebm_path)
                if early_stopping.early_stop:
                    print("Y_ENCODER | Early stopping")
                    break

            final_y_encoder_vali_loss = validate_y_encoder(experiment_data.val_loader)
            final_y_encoder_test_loss = validate_y_encoder(experiment_data.test_loader)

            self.logger.report_scalar(
                title=f"Y_ENCODER Validation loss",
                series="MSE",
                value=float(final_y_encoder_vali_loss),
                iteration=y_encoder_train_epochs_initial,
            )

            self.logger.report_scalar(
                title=f"Y_ENCODER Testing loss",
                series="MSE",
                value=float(final_y_encoder_test_loss),
                iteration=y_encoder_train_epochs_initial,
            )

            with open(y_enc_done_full_path, mode="w") as report_file:
                report_file.write(f"{1}")
        elif self.args.ebm_training_strategy == "train_y_and_xy_together":
            print("NOT training Y_ENCODER using the supervised loss")
        elif self.args.ebm_training_strategy == "cd_only":
            print("Training 'cd_only'")
        else:
            raise ValueError(
                f"Selected training strategy '{self.args.ebm_training_strategy}' UNSUPPORTED!"
            )

        ebm.train()
        set_grad_flow_for_nn(ebm.autoformer, False)
        set_grad_flow_for_nn(
            ebm.y_encoder, self.args.ebm_training_strategy == "train_y_and_xy_together"
        )
        set_grad_flow_for_nn(ebm.xy_decoder, True)

        if self.args.ebm_training_strategy == "cd_only":
            set_grad_flow_for_nn(ebm.autoformer, True)
            set_grad_flow_for_nn(ebm.y_encoder, True)
            set_grad_flow_for_nn(ebm.xy_decoder, True)

        print(f"Train Y and XY together with CD: {self.args.ebm_training_strategy == 'train_y_and_xy_together'}")

        early_stopping_ebm = EarlyStopping(
            patience=self.args.patience, verbose=True, model_name_postfix="xy_ebm"
        )

        model = ebm.xy_decoder
        optimizer_for_xy_decoder = self._select_optimizer(model=model)

        
        xy_decoder_train_epochs = self.args.ebm_epochs

        ebm_full_path = full_ebm_path = os.path.join(ebm_path, "full_ebm.pth")

        def sample_langevin(
            enc_out,
            batch_y,
            stepsize: float,
            n_steps: int,
            batch_x,
            batch_x_mark,
            dec_inp,
            batch_y_mark,
            noise_scale=None,
            stepsize_sched_rate: float = 1.0,
            intermediate_samples=False,
        ):
            from torch import autograd

            enc_x = enc_out

            if noise_scale is None:
                noise_scale = torch.sqrt(torch.tensor(stepsize * 2))

            
            batch_y.requires_grad = True
            for _ in range(n_steps):
                noise = torch.randn_like(batch_y) * noise_scale
                out = ebm._get_decoded(
                    batch_y, batch_x, batch_x_mark, dec_inp, batch_y_mark, enc_out
                )
                grad = autograd.grad(
                    out.sum(), batch_y, only_inputs=True, allow_unused=True
                )[0]
                dynamics = stepsize * grad + noise
                batch_y = batch_y + dynamics
                stepsize = stepsize * stepsize_sched_rate

            return batch_y

        def some_random_method(
            enc_out,
            batch_y,
            batch_x,
            batch_x_mark,
            dec_inp,
            batch_y_mark,
            stepsize=0.1,
            n_steps=100,
            alpha=1.0,
            cd_sched_rate=1.0,
            device=torch.device("cpu"),
            margin_for_square_square=None,
        ):
            

            
            
            
            batch_x = batch_x.detach()
            batch_x_mark = batch_x_mark.detach()
            dec_inp = dec_inp.detach()
            batch_y_mark = batch_y_mark.detach()

            local_batch_y = torch.clone(batch_y)

            neg_y = torch.randn_like(local_batch_y)
            neg_y = sample_langevin(
                enc_out=enc_out,
                batch_y=neg_y,
                stepsize=stepsize,
                n_steps=n_steps,
                batch_x=batch_x,
                batch_x_mark=batch_x_mark,
                dec_inp=dec_inp,
                batch_y_mark=batch_y_mark,
                intermediate_samples=False,
            ).to(device)

            pos_out = -ebm._get_decoded(
                local_batch_y, batch_x, batch_x_mark, dec_inp, batch_y_mark, enc_out
            )
            neg_out = -ebm._get_decoded(
                neg_y, batch_x, batch_x_mark, dec_inp, batch_y_mark, enc_out
            )

            loss = (pos_out - neg_out) + alpha * (pos_out**2 + neg_out**2)
            if margin_for_square_square != -1.0:
                zeros = torch.zeros_like(pos_out)
                margin_tensor = torch.tensor(margin_for_square_square)
                loss += torch.square(pos_out) - torch.square(
                    torch.maximum(zeros, (neg_out - margin_tensor))
                )
            loss = loss.mean()
            return loss

        def validate_full_ebm(data_loader):
            validation_loss_list = []

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                data_loader
            ):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len :, :]).float()
                dec_inp = (
                    torch.cat([batch_y[:, : self.args.label_len, :], dec_inp], dim=1)
                    .float()
                    .to(self.device)
                )

        ebm_margin_loss = self.args.ebm_margin_loss
        def loss_method(
            enc_out,
            batch_y,
            batch_x,
            batch_x_mark,
            dec_inp,
            batch_y_mark,
        ):
            return some_random_method(
                enc_out=enc_out,
                batch_y=batch_y,
                batch_x=batch_x,
                batch_x_mark=batch_x_mark,
                dec_inp=dec_inp,
                batch_y_mark=batch_y_mark,
                stepsize=self.args.ebm_cd_step_size,
                n_steps=self.args.ebm_cd_num_steps,
                alpha=self.args.ebm_cd_alpha,
                cd_sched_rate=self.args.ebm_cd_sched_rate,
                device=self.device,
                margin_for_square_square=ebm_margin_loss,
            )

        for epoch in range(xy_decoder_train_epochs):
            iter_count = 0
            train_loss = []

            model.train()
            epoch_time = time.time()
            ebm_loss_list = []

            if epoch == 0:
                print(
                    f"XY DECODER TRAIN loader size: {len(experiment_data.train_loader)}"
                )

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                experiment_data.train_loader
            ):
                iter_count += 1
                optimizer_for_xy_decoder.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                f_dim = -1 if self.args.features == "MS" else 0
                target_batch_y = batch_y[:, :, f_dim:]

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                
                dec_inp = torch.zeros_like(
                    target_batch_y[:, -self.args.pred_len :, :]
                ).float()
                dec_inp = (
                    torch.cat(
                        [target_batch_y[:, : self.args.label_len, :], dec_inp], dim=1
                    )
                    .float()
                    .to(self.device)
                )

                enc_out, attns = ebm._forward_enc_orig(
                    batch_x, batch_x_mark, dec_inp, batch_y_mark
                )
                loss_ebm = loss_method(
                    enc_out=enc_out,
                    batch_y=target_batch_y,
                    batch_x=batch_x,
                    batch_x_mark=batch_x_mark,
                    dec_inp=dec_inp,
                    batch_y_mark=batch_y_mark,
                )
                ebm_loss_list.append(loss_ebm.cpu().detach().numpy())

                
                loss_ebm.backward()
                optimizer_for_xy_decoder.step()

            if (epoch + 1) % self.args.ebm_validate_during_training_step == 0:
                validation_results_object, _ = TEMInferencer.test_adhoc_energy(
                    args=self.args,
                    device=self.device,
                    path_or_reference_of_model=ebm,
                    experiment_data=experiment_data,
                    data_loader_name="val",
                )

            ebm_epoch_loss = np.mean(np.stack(ebm_loss_list))
            try:
                self.logger.report_scalar(
                    title=f"Training loss (loss_ebm)",
                    series="Training loss",
                    value=float(
                        ebm_epoch_loss.mean()
                    ),  
                    iteration=epoch + 1,
                )
            except Exception as e:
                print(e)

            print(
                "xy_decoder | epoch: {} cost time: {}".format(
                    epoch + 1, time.time() - epoch_time
                )
            )
            print("xy_decoder | loss: {}".format(ebm_epoch_loss))

            early_stopping_ebm(ebm_epoch_loss, model, ebm_path)
            if early_stopping_ebm.early_stop:
                print("XY_DECODER | Early stopping")
                break

        torch.save(ebm, full_ebm_path)
        full_ebm = torch.load(full_ebm_path, map_location=self.device)

        try:
            import pickle

            ebm_params_obj = os.path.join(ebm_path, "ebm_args.pkl")

            ebm_param_dict = dict()
            
            for key, value in args_dict.items():
                
                if key.startswith("ebm_"):
                    ebm_param_dict[key] = value

            with open(ebm_params_obj, "wb") as ebm_param_dict_file:
                pickle.dump(ebm_param_dict, ebm_param_dict_file)

            with open(ebm_params_obj, "rb") as ebm_param_dict_file:
                pickle.load(ebm_param_dict_file)
        except Exception as e:
            print(f"Exception occured during ebm param serialization. Error: {e}")

        try:
            TEMInferencer.test_adhoc_energy(
                args=self.args,
                device=self.device,
                path_or_reference_of_model=full_ebm_path,
                experiment_data=experiment_data,
                do_run_wide_experiments=True,
                
            )

        except Exception as e:
            print(f"Exception: {str(e)}")

        after_full_ebm_train = DateUtils.now()
        print(f"Finished EBM training, took: ({after_full_ebm_train - before_full_ebm_train})")

        return full_ebm, full_ebm_path

    def get_train_data(
        self,
        override_data_path=None,
        override_scaler=None,
        override_target_site_id=None,
        override_flag=None,
    ):
        if override_flag is not None:
            print(f"Fetching train data with override flag: '{override_flag}'")
        train_data, train_loader = self._get_data(
            flag="train" if override_flag is None else override_flag,
            override_batch_size=self.args.ebm_inference_batch_size,
            override_data_path=override_data_path,
            override_scaler=override_scaler,
            override_target_site_id=override_target_site_id,
        )
        return train_data

    def test_adhoc_energy(
        self,
        path_or_reference_of_model,
        experiment_data: ExperimentData,
        args=None,
        device=None,
        data_loader_name: str = "val",
        do_run_wide_experiments: bool = False,
        do_run_energy_optimization_experiments=True,
        override_dataset_tuples=None,
    ):
        
        print(f"[COMPAT] Running exp_main_energy.test_adhoc_energy!")
        return TEMInferencer.test_adhoc_energy(
            args=self.args if args is None else args,
            device=self.device if device is None else device,
            path_or_reference_of_model=path_or_reference_of_model,
            experiment_data=experiment_data,
            data_loader_name=data_loader_name,
            do_run_wide_experiments=do_run_wide_experiments,
            do_run_energy_optimization_experiments=do_run_energy_optimization_experiments,
            override_dataset_tuples=override_dataset_tuples,
        )
