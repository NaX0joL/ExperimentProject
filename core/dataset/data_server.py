from collections.abc import Generator

import pandas as pd

from torch.utils.data import DataLoader

from .dataset_config import DatasetConfig
from .new_dataset import ExperimentDataset, ExperimentDatasetFactory
from .new_datamodule import DataModule
from .preprocessing.splitter.leave_one_out import LeaveOneOut



class DataServer():
    
    def __init__(self, dataset_config:DatasetConfig) -> None:
        self.dataset_config = dataset_config
        self.data = self._get_preprocessed_data()
        return
    
    def serve_datamodule(self, n_fold:int) -> Generator[DataModule]:
        splitter = LeaveOneOut(self.data)
        
        for train_fold, test_fold in splitter.split_data(n_fold):
            datamodule = self._get_datamodule(
                train_df = train_fold,
                validation_df = None,
                test_df = test_fold,
            )
            yield datamodule
        
        return
        
    def _get_preprocessed_data(self) -> pd.DataFrame:
        processes = self.dataset_config.pipeline
        
        data = processes.loader.load_data(parallelized=True)
        if processes.pre_filter is not None:
            data = processes.pre_filter.filter_data(data)
        data = processes.grouper.group_data(data)
        if processes.regrouper is not None:
            data = processes.regrouper.regroup_data(data)
        data = processes.segmenter.segment_data(data)
        if processes.post_filter is not None:
            data = processes.post_filter.filter_data(data)
        if processes.transformer is not None:
            data = processes.transformer.transform_data(data)
        
        return data
    
    def _get_datamodule(self, train_df:pd.DataFrame, validation_df:pd.DataFrame, test_df:pd.DataFrame) -> DataModule:
        train_dataset = self._build_dataset(train_df)
        validation_dataset = self._build_dataset(validation_df)
        test_dataset = self._build_dataset(test_df)
        
        train_dataloader = self._build_dataloader(train_dataset)
        validation_dataloader = self._build_dataloader(validation_dataset)
        test_datalaoder = self._build_dataloader(test_dataset)
        
        datamodule = DataModule(
            train_dataloader = train_dataloader,
            validation_dataloader = validation_dataloader,
            test_dataloader = test_datalaoder,
        )
        return datamodule
    
    def _build_dataset(self, dataframe:pd.DataFrame) -> ExperimentDataset:
        dataset = ExperimentDatasetFactory.create(dataframe)
        return dataset
    
    def _build_dataloader(self, dataset:ExperimentDataset) -> DataLoader:
        if not dataset or len(dataset) == 0:
            return None
        
        dataloader = DataLoader(
            dataset = dataset,
            batch_size = self.dataset_config.batch_size,
            shuffle = self.dataset_config.shuffle,
            drop_last = self.dataset_config.drop_last,
        )
        return dataloader