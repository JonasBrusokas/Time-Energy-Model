import os
from pathlib import Path
from torch.utils.data import DataLoader

from data_provider.data_loader import (
    Dataset_Custom,
    Dataset_ETT_hour,
    Dataset_ETT_minute,
    Dataset_Pred,
)

data_dict = {
    "ETTh1": Dataset_ETT_hour,
    "ETTh2": Dataset_ETT_hour,
    "ETTm1": Dataset_ETT_minute,
    "ETTm2": Dataset_ETT_minute,
    "custom": Dataset_Custom,
}

def extract_data_and_data_path_tuples(
    multi_data_string: str, multi_data_path_string: str
):
    data_list = list(
        map(lambda data_str: data_str.strip(), multi_data_string.split(","))
    )
    data_path_list = list(
        map(
            lambda data_path_str: data_path_str.strip(),
            multi_data_path_string.split(","),
        )
    )
    if len(data_path_list) != len(data_list):
        raise ValueError(f"The length of data_list and multi_data_path does not match!")
    return list(zip(data_list, data_path_list))


def data_provider(
    args,
    flag,
    override_batch_size=None,
    override_data_path=None,
    override_scaler=None,
    override_target_site_id=None,
    override_dataset_tuples=None,
):
    

    def fetch_data_to_use():
        return data_dict[args.data]

    Data = fetch_data_to_use()
    timeenc = 0 if args.embed != "timeF" else 1

    if override_data_path is not None:
        if not Path(os.path.join(args.root_path, override_data_path)).exists():
            raise ValueError(f"Data at path {override_data_path} does not exist!")
        data_path = override_data_path
    else:
        data_path = args.data_path

    if flag == "test":
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    elif flag == "pred":
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        Data = Dataset_Pred
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    if args.data in ["ETTh1", "ETTh2"]:
        data_set = Data(
            root_path=args.root_path,
            data_path=data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            override_scaler=override_scaler,
        )
    elif args.data in ["custom"]:
        data_set = Data(
            root_path=args.root_path,
            data_path=data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
        )
    else:
        raise ValueError(f"Unknown dataset type: {args.data}")
        data_set = Data(
            root_path=args.root_path,
            data_path=data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
        )

    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size if override_batch_size is None else override_batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last,
    )
    return data_set, data_loader


def get_data_from_provider(
    args,
    flag,
    override_batch_size=None,
    override_data_path=None,
    override_scaler=None,
    override_target_site_id=None,
    override_dataset_tuples=None,
):
    data_set, data_loader = data_provider(
        args=args,
        flag=flag,
        override_batch_size=override_batch_size,
        override_data_path=override_data_path,
        override_scaler=override_scaler,
        override_target_site_id=override_target_site_id,
        override_dataset_tuples=override_dataset_tuples,
    )
    return data_set, data_loader