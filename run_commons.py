class ExperimentConstants:
    fix_seed = 2024

    # Settings prefixes
    SETTINGS_PREFIX = "v3"

    # Result object prefixes
    RESULT_OBJECT_VERSION_POSTFIX = (
        "20240919_v1"
    )

class TorchDeviceUtils:

    @staticmethod
    def check_if_should_use_gpu(args) -> (bool, bool):
        import torch
        should_use_gpu = True if torch.cuda.is_available() or args.use_gpu else False
        if (should_use_gpu):
            # If CUDA is not found, check for the MPS backend
            try:
                mps_available = torch.backends.mps.is_available()
                if (mps_available):
                    print("USING MPS !!!")
                    should_use_gpu = mps_available
            except Exception as e:
                print(f"Exception occurred while trying to check for 'mps' availability : {e}")
                mps_available = False
        else:
            mps_available = False
        print(f">> USING GPU: {should_use_gpu}")

        if should_use_gpu and not mps_available:
            cuda_device_count = torch.cuda.device_count()
            print(f">> Number of available CUDA GPUs: {cuda_device_count}")
            if cuda_device_count > 1:
                print(
                    f"***\n***\nWARNING: more than 1 CUDA GPU is available in this scripts scope\n***\n***"
                )
        return (should_use_gpu, mps_available)
