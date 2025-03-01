import os

import torch


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        build_model = self._build_model()
        if build_model is not None:
            self.model = build_model.to(self.device)
        else:
            self.model = None

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            if torch.cuda.is_available():
                os.environ["CUDA_VISIBLE_DEVICES"] = (
                    str(self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
                )
                device = torch.device("cuda:{}".format(self.args.gpu))
                print("Use GPU: cuda:{}".format(self.args.gpu))
            elif torch.backends.mps.is_available():
                print(f"Using MPS!")
                device = torch.device("mps")
            else:
                
                print(f"Fallback to CPU! Did not find compatible torch GPU backend")
                device = torch.device("cpu")
        else:
            device = torch.device("cpu")
            print("Use CPU")
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
