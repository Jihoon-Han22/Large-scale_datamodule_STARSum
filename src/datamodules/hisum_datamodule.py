from typing import Any, Dict, Optional

import os
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from src.datamodules.sum_dataset import SumDataset, BatchCollator

class SumDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        exp_type: str,
        dataset_type: str,
        h5_filename: str,
        try_num:int,
        split_id: int,
        batch_size: int,
        num_workers: int,
        prefetch_factor: int,
        pin_memory: bool,
        uboco_model_dir: str,
        uboco_hparams: DictConfig,
        tvsum_tsm_parser: DictConfig,
        summe_tsm_parser: DictConfig,
        mrhighlight_tsm_parser: DictConfig
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self._data_train: Optional[Dataset] = None
        self._data_val: Optional[Dataset] = None
        self._data_test: Optional[Dataset] = None
        
        self.exp_type = self.hparams.exp_type
        self.dataset_type = dataset_type

        self.h5_filename = self.hparams.h5_filename
        self.try_num = self.hparams.try_num
        
        if self.exp_type == 'original':
            self.h5_file_path = '/data2/projects/summarization/summarization/summarization_dataset/mrsum_with_features.h5'
        else:
            NotImplementedError
        
        self.split_id = 0
        self.train_test_json = '/data2/projects/summarization/summarization/summarization_dataset/mrsum_split.json'

        
        self.bd_model_path = os.path.join(self.hparams.uboco_model_dir, f"{self.dataset_type}{self.split_id}_bd_model.pt")
        self.uboco_cfg = self.hparams.uboco_hparams
        
        if self.dataset_type == "tvsum":
            self.parser_cfg = self.hparams.tvsum_tsm_parser
        elif self.dataset_type == "summe":
            self.parser_cfg = self.hparams.summe_tsm_parser
        elif self.dataset_type == "mrhighlight":
            self.parser_cfg = self.hparams.mrhighlight_tsm_parser
        else:
            NotImplementedError

    @property
    def train_ds(self):
        return self._data_train

    @property
    def val_ds(self):
        return self._data_val

    @property
    def test_ds(self):
        return self._data_test
    

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        if (stage == 'fit' or stage is None) and not self._data_train:
            self._data_train = SumDataset(
                                exp_type = self.exp_type,
                                h5_file_path = self.h5_file_path,
                                train_test_json = self.train_test_json,
                                mode = 'train',
                                split_id = self.split_id,
                                bd_model_path = self.bd_model_path,
                                uboco_config = self.uboco_cfg,
                                tsm_parser_config = self.parser_cfg
                                )
            
            self._data_val = SumDataset(
                                exp_type = self.exp_type,
                                h5_file_path = self.h5_file_path,
                                train_test_json = self.train_test_json,
                                mode = 'val',
                                split_id = self.split_id,
                                bd_model_path = self.bd_model_path,
                                uboco_config = self.uboco_cfg,
                                tsm_parser_config = self.parser_cfg
                                )

            self._data_test = SumDataset(
                                exp_type = self.exp_type,
                                h5_file_path = self.h5_file_path,
                                train_test_json = self.train_test_json,
                                mode = 'test',
                                split_id = self.split_id,
                                bd_model_path = self.bd_model_path,
                                uboco_config = self.uboco_cfg,
                                tsm_parser_config = self.parser_cfg
                                )
        
        if (stage == 'test' or stage is None) and not self._data_test:
            self._data_test = SumDataset(
                                exp_type = self.exp_type,
                                h5_file_path = self.h5_file_path,
                                train_test_json = self.train_test_json,
                                mode = 'test',
                                split_id = self.split_id,
                                bd_model_path = self.bd_model_path,
                                uboco_config = self.uboco_cfg,
                                tsm_parser_config = self.parser_cfg
                                )

    def train_dataloader(self):
        return DataLoader(
            dataset=self._data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            prefetch_factor=self.hparams.prefetch_factor,
            collate_fn = BatchCollator()
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self._data_val,
            batch_size=1, #self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            prefetch_factor=self.hparams.prefetch_factor,
            collate_fn = BatchCollator()
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self._data_test,
            batch_size=1, #self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            prefetch_factor=self.hparams.prefetch_factor,
            collate_fn = BatchCollator()
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "datamodule" / "summe.yaml")
    cfg.data_dir = str(root / "data")
    _ = hydra.utils.instantiate(cfg)