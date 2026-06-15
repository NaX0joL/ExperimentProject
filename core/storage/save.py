import os
import json
import pickle
from pathlib import Path
from pprint import pformat
from datetime import datetime
import pickle

import torch
from torch import nn, Tensor

from core.schema import ExperimentState
from core.master_config import MasterConfig
from core.dataset.datamodule import DataModule
from modules.plotter import PlotConfig, plot_timeseries, plot_subplots, save_figures_to_pdf
from modules.tensor_normalization import normalize_tensor
from modules.device_resolve import get_model_device



SAVE_DIR = Path("savefolder/mpkg")
NUM_PLOT = 40



class SaveService():
    
    def __init__(self, experiment_state:ExperimentState):
        self.experiment_state = experiment_state
        return
    
    ### public functions
    
    def put_in_marker(self, **content:any) -> None:
        self.marker_content = content
        return
    
    def save(self, path:Path=None, verbose:bool=True) -> None:
        path = self._resolve_path(path)
        self._create_mpkg_folder(path)
        
        if verbose:
            print("saving mpkg...")
        
        marker_file.save(getattr(self, "marker_content", None), path)
        config_file.save(self.experiment_state.master_config, path)
        model_state_file.save(self.experiment_state.model, path)
        loss_figure_file.save(self.experiment_state.loss_log, path)
        metrics_file.save(self.experiment_state.metrics_result, path)
        model_plot.save(self.experiment_state.model, self.experiment_state.datamodule, path)
        
        if verbose:
            print(f"mpkg saved to {path}")
        return
        
    ### private functions
    
    def _resolve_path(self, path:Path) -> Path:
        if path is None:
            path = SAVE_DIR / "tmp" / self._generate_tmp_mpkg_name()
        return Path(path)
    
    def _generate_tmp_mpkg_name(self) -> str:
        current_time = datetime.now().strftime("%Y_%m_%d-%H%M%S")
        tmp_mpkg_name = f"mpkg-tmp_exp-{current_time}"
        return tmp_mpkg_name
    
    def _create_mpkg_folder(self, path:Path) -> None:
        os.makedirs(name=path, exist_ok=True)
        return



class marker_file():
    
    @staticmethod
    def save(marker_content:dict, path:Path) -> None:
        init_file_path = path / "__mpkg__.py"
        with open(init_file_path, "w") as file:
            if marker_content is not None:
                for key, value in marker_content.items():
                    file.write(f"{key}={repr(value)}\n")
        return



class config_file():
    
    @staticmethod
    def save(config:MasterConfig, path:Path) -> None:
        config_file._create_pkl(config, path)
        config_file._create_txt(config, path)
        return
    
    @staticmethod
    def _create_pkl(config:MasterConfig, path:Path) -> None:
        pickle_path = path / "config.pkl"
        with open(pickle_path, 'wb') as file:
            pickle.dump(config, file)
        return
    
    @staticmethod
    def _create_txt(config:MasterConfig, path:Path) -> None:
        txt_path = path / "config.txt"
        with open(txt_path, 'w') as file:
            file.write(pformat(config, indent = 2, width = 120,))
        return



class model_state_file():
    
    @staticmethod
    def save(model:nn.Module, path:Path) -> None:
        model_state_path = path / "model.pth"
        torch.save(model.state_dict(), model_state_path)
        return



class loss_figure_file():
    
    @staticmethod
    def save(loss_log:dict[str, list], path:Path) -> None:
        plot_timeseries(
            loss_log.get("train"), loss_log.get("validation"),
            labels = ["train loss", "validation loss"],
            cfg = PlotConfig(
                title = None,
                figsize = (9, 5),
                xlabel = "epoch",
                ylabel = "loss",
                save_path = f"{path}/loss.png",
            ),
            show = False,
        )
        return



class metrics_file():
    
    @staticmethod
    def save(metrics_result:dict[str, dict], path:Path) -> None:
        metrics_file_path = path / "metrics.json"
        with open(metrics_file_path, 'w', encoding='utf-8') as file:
            json.dump(metrics_result, file, indent=4)
        return



class model_plot():
    
    @staticmethod
    def save(model:nn.Module, datamodule:DataModule, path:Path, num_plot:int=None) -> None:
        folder_path = model_plot._create_folder(path)
        num_plot = model_plot._resolve_num_plot(num_plot)
        model_plot._create_plot(model, datamodule, num_plot, folder_path)
        return
    
    @staticmethod
    def _create_folder(path:Path) -> Path:
        folder_path = path / "model_plot"
        os.makedirs(name=folder_path, exist_ok=True)
        return folder_path
    
    @staticmethod
    def _resolve_num_plot(num_plot:int) -> int:
        if num_plot is None:
            num_plot = NUM_PLOT
        return num_plot
    
    @staticmethod
    def _create_plot(model:nn.Module, datamodule:DataModule, num_plot:int, path:Path) -> None:
        train, _, test = datamodule.unwrap()
        device = get_model_device(model)
        
        plot_queue = {
            "train": train,
            "test": test,
        }
        
        for type, dataloader in plot_queue.items():

            cnt = 0
            figs = []
            for batch in dataloader:
                batch: dict[str, Tensor|list]
                
                input = batch.get("value").float().to(device)
                label = batch.get("ground_truth").float().to(device)
                output = model(input)
                
                input = normalize_tensor(input).cpu().detach().numpy()
                label = normalize_tensor(label).cpu().detach().numpy()
                output = normalize_tensor(output).cpu().detach().numpy()
                
                for index in range(min(input.shape[0], label.shape[0], output.shape[0])):
                    if cnt >= num_plot:
                        break
                    cnt += 1
                    
                    fig, _ = plot_subplots(
                        input[index], label[index], output[index],
                        labels = ["input", "ground_truth", "output"],
                        cfg = PlotConfig(
                            title = None,
                            figsize = (9, 3),
                            xlabel = "time step",
                            ylabel = None,
                        ),
                        show = False,
                    )
                    figs.append(fig)
            
            save_figures_to_pdf(figs, f"{path}/{type}.pdf")
        
        return