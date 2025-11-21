from pathlib import Path

import pickle
from omegaconf import OmegaConf
import pandas as pd

import deeplabcut
from deeplabcut.core.engine import Engine
from utils import copy_template_project, update_pose_config


class TestCase():
    """
    Class for handling training configurations, results, and expected output. 
    A testcase configuration consists of: 
        - name
        - ref_output_path

    and the following training-specific parameters: 
        - dataset
        - net_type
        - detector
        - training (hyperparameters)

    By default the configuration and ref_output are validated at initialization.
    """ 
    def __init__(
        self,
        config,
        validate=True,
    ):
        self.config = config
        self.name = OmegaConf.select(config,'testcase.name')
        self.ref_output_path = Path(OmegaConf.select(config,'testcase.ref_output') or '')
        self._ref_output: dict | None = None
        self._results = dict()
        
        if validate:
            self.validate_config()

    def validate_config(self):
        assert self.name is not None and self.ref_output_path.exists(), (
            "Invalid testcase configuration. Make sure to pass a valid testcase "
            "yaml that points to an existing reference output file. See template "
            "tests/functional_testing/config/testcase/_template.yaml. "
            "example usage: python run_testcase testcase=fly-resnet_50-sddlite"
        )
        ref_output = self.ref_output
        pass # TODO implement validation of ref_output

    @classmethod
    def load_pickle(cls, fn: str | Path) -> dict:
        with open(fn, 'rb') as f:
            ref_output = pickle.load(f)
        return ref_output

    @classmethod
    def save_pickle(cls, data: dict, fn: str | Path):
        with open(fn, 'wb') as f:
            pickle.dump(data, f)

    @property
    def ref_output(self) -> dict:
        if self._ref_output is None:
            self._ref_output = self.load_pickle(self.ref_output_path)
        return self._ref_output


    # TODO: add summaries of results
    def store_train_results(self, project_path: str | Path):
        _, _, train_folder = deeplabcut.return_train_network_path(
            config=project_path / 'config.yaml',
            shuffle=1,
            trainingsetindex=0,
        )
        train_results = pd.read_csv(train_folder/ 'learning_stats.csv')
        self._results['training'] = train_results

    def store_eval_results(
        self,
        project_path: str | Path,
        trainingsetindex: int=0
    ):
        csv_file = Path(
            project_path,
            'evaluation-results-pytorch',
            f'iteration-{trainingsetindex}',
            'CombinedEvaluation-results.csv',
        )
        eval_results = pd.read_csv(csv_file)
        self._results['eval'] = eval_results

    def store_analysis_results(
        self,
        project_path: str | Path,
        video_name: str,
        scorer: str,
    ):
        csv_file = Path(
            project_path,
            'video_analysis',
            f'{video_name}{scorer}.csv',
        )
        analysis_results = pd.read_csv(csv_file)
        self._results['analysis'] = analysis_results
        pass

    def compare_performance(self):
        report = dict()
        for stage, metric in [
            ('training', 'losses/train.total_loss'),
            ('eval', 'test rmse'),
            # ('analysis', '')
        ]:
            report[stage] = self._results[stage][metric] - self.ref_output[stage][metric]
        return report


def init_project_for_testcase(
    project_path: str | Path, 
    template_dataset: Path | str,
    net_type: str,
    detector_type: str | None = None,
    epochs: int | None = None,
    detector_epochs: int | None = None,
    seed: int | None = None,
    batch_size: int | None = None, 
) -> Path:
    """
    Create a new train-ready deeplabcut project for a testcase
    configuration. 

    Args: 
        project_path (Path): the output directory where the project is initialized
            (and all training results are stored).
        template_dataset (Path): path to template dataset folder with subfolder
            labeled-data and a template project config (config.yaml). 
        net_type (str): the network type to initialize a training run for. Must be
            one of pose_estimation_pytorch.available_models().
        detector_type (str): the type of detector that will be used. Must be
            one of pose_estimation_pytorch.available_detectors(). If None, the 
            default detector type ``ssdlite`` will be used.
        epochs (int): the number of training epochs for the pose model.
        detector_epochs (int): the number of training epochs for the detector.
        seed (int): the RNG seed used for training.
        batch_size (int): the minibatch size used for training.

    Returns: 
        config_path (Path): the path of the dlc project config.yaml inside the 
            project folder. 
    """

    # Copy template project to output project path
    copy_template_project(template_dataset, project_path)

    # Edit the project config
    config_path = project_path / 'config.yaml'

    # Create training dataset for net_type and detector_type
    deeplabcut.create_training_dataset(
        config_path,
        net_type=net_type,
        detector_type=detector_type,
        engine=Engine.PYTORCH,
        num_shuffles=1,
    )

    # Edit the pose_cfg
    pose_cfg, _, _ = deeplabcut.return_train_network_path(
        config_path,
        shuffle=1,
        trainingsetindex=0,
    )

    update_pose_config(
        pose_cfg=pose_cfg,
        epochs=epochs,
        detector_epochs=detector_epochs,
        seed=seed,
        batch_size=batch_size, 
    )

    return config_path
