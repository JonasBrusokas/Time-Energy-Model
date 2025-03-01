import hashlib
from sklearn.preprocessing import StandardScaler

from data_provider.data_factory import get_data_from_provider


class ExperimentData:
    cache = {}

    def __init__(
        self,
        train_data,
        train_loader,
        val_data,
        val_loader,
        test_data,
        test_loader,
        unique_identifier=None,
        short_unique_identifier=None,
    ):
        self.train_data = train_data
        self.train_loader = train_loader
        self.val_data = val_data
        self.val_loader = val_loader
        self.test_data = test_data
        self.test_loader = test_loader
        self.id = unique_identifier
        self.short_id = short_unique_identifier

    @classmethod
    def from_args(
        cls,
        args,
        train_flag="train",
        val_flag="val",
        test_flag="test",
        override_batch_size=None,
        override_data_path=None,
        override_scaler=None,
        override_target_site_id=None,
        override_dataset_tuples=None,
        override_id=None,
        override_short_id=None,
    ):
        
        def generate_id_components(
            is_short=False,
        ):
            def hash_value(
                value,
            ):
                return hashlib.shake_256(str(value).encode()).hexdigest(14 // 2)

            components = [
                f"{train_flag}" if train_flag != "train" else "",
                f"{val_flag}" if val_flag != "val" else "",
                f"{test_flag}" if test_flag != "test" else "",
            ]

            if is_short:
                components_to_be_shortened = []
                if override_batch_size is not None:
                    components_to_be_shortened.append(
                        f"o_batch={(override_batch_size)}"
                    )
                if override_data_path is not None:
                    components_to_be_shortened.append(
                        f"o_d_path={(override_data_path)}"
                    )
                if override_scaler is not None:
                    components_to_be_shortened.append(
                        f"o_scaler={(ExperimentData._format_scaler_id(override_scaler))}"
                    )
                components.append(f"h={hash_value(components_to_be_shortened)}")
            else:
                if override_batch_size is not None:
                    components.append(f"o_batch={override_batch_size}")
                if override_data_path is not None:
                    components.append(f"o_d_path={override_data_path}")
                if override_scaler is not None:
                    components.append(
                        f"o_scaler={ExperimentData._format_scaler_id(override_scaler)}"
                    )

            if override_target_site_id is not None:
                components.append(f"o_site_id={override_target_site_id}")
            if override_dataset_tuples is not None:
                components.append(f"o_d_tuples={hash_value(override_dataset_tuples)}")

            return components

        generated_id = "ExpData" + "_".join(
            filter(
                lambda s: len(s) != 0,
                generate_id_components(
                    is_short=False,
                ),
            )
        ) if override_id is None else override_id
        generated_short_id = "ExpData" + "_".join(
            filter(
                lambda s: len(s) != 0,
                generate_id_components(
                    is_short=True,
                ),
            )
        ) if override_short_id is None else override_short_id

        
        if generated_id in cls.cache:
            return cls.cache[generated_id]

        train_data, train_loader = get_data_from_provider(
            args=args,
            flag=train_flag,
            override_batch_size=override_batch_size,
            override_data_path=override_data_path,
            override_scaler=override_scaler,
            override_target_site_id=override_target_site_id,
            override_dataset_tuples=override_dataset_tuples,
        )
        val_data, val_loader = get_data_from_provider(
            args=args,
            flag=val_flag,
            override_batch_size=override_batch_size,
            override_data_path=override_data_path,
            override_scaler=override_scaler,
            override_target_site_id=override_target_site_id,
            override_dataset_tuples=override_dataset_tuples,
        )
        test_data, test_loader = get_data_from_provider(
            args=args,
            flag=test_flag,
            override_batch_size=override_batch_size,
            override_data_path=override_data_path,
            override_scaler=override_scaler,
            override_target_site_id=override_target_site_id,
            override_dataset_tuples=override_dataset_tuples,
        )

        experiment_data = cls(
            train_data,
            train_loader,
            val_data,
            val_loader,
            test_data,
            test_loader,
            unique_identifier=generated_id,
            short_unique_identifier=generated_short_id,
        )

        
        cls.cache[generated_id] = experiment_data

        return experiment_data

    @staticmethod
    def _format_scaler_id(scaler):
        if scaler is not None and isinstance(scaler, StandardScaler):
            return f"scaler_scale_{scaler.scale_}_mean_{scaler.mean_}_var_{scaler.var_}"
        return str(scaler)

    def __str__(self):
        return self.short_id
