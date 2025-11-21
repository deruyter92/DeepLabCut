from pathlib import Path

from utils import copy_template_project, update_pose_config
import deeplabcut
from deeplabcut.core.engine import Engine
import deeplabcut.pose_estimation_pytorch as dlc_torch


def train_with_template_dataset(
    project_path: Path | str,
    template_dataset: Path | str,
    net_type: str,
    detector_type: str | None = None,
    epochs: int | None = None,
    detector_epochs: int | None = None,
    seed: int | None = None,
    batch_size: int | None = None, 
): 
    """
    Train a new pose network using a template dataset.

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

        adjust_config (dict, optional): optional mapping of parameter adjustments
            in the project config
        adjust_pose_config (dict, optional): mapping of optional adjustments
            to the pytorch_config model configuration.

        epochs (int, optional): the number of 
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

    # Train network
    dlc_torch.train_network(
        config=config_path,
        shuffle=1,
        trainingsetindex=0,
    )