from pathlib import Path
import yaml


def copy_template_project(
    template_folder: Path | str,
    destination_folder: Path | str, 
) -> Path:
    template_folder = Path(template_folder).resolve()
    destination_folder = Path(destination_folder).resolve()
    template_config = template_folder / 'config.yaml'

    print('template dataset: ', template_folder)
    print('destination folder: ', destination_folder)
    assert destination_folder.exists(), "Invalid destination folder" 
    assert (
        template_config.exists() and
        (template_folder / 'labeled-data').exists()
    ), "Invalid template folder"

    # Create symlinks to the dataset folders
    (destination_folder / 'labeled-data').symlink_to(
        template_folder / 'labeled-data'
    )
    
    # Read template project configuration
    with template_config.open('r') as f:
        project_config = yaml.safe_load(f)

    # Write config to new folder (with correct field 'project_path')
    project_config['project_path'] = str(destination_folder.resolve())
    with (destination_folder / 'config.yaml').open('w') as f:
        yaml.safe_dump(project_config, f)
    return destination_folder


def update_pose_config(
    pose_cfg: str | Path,
    epochs: int | None = None,
    detector_epochs: int | None = None,
    seed: int | None = None,
    batch_size: int | None = None, 
):
    """
    Helper function to update specific fields in the nested config dictionary 
    for the pose estimation model. 

    args: 
        pose_cfg (str): path to the pytorch_config.yaml file to be updated.
        epochs (int, optional): update value for number of training epochs
        detector_epochs (int, optional): update value for number of detector_epochs
        seed (int, optional): update value for rng seed
        batch_size (int, optional): update value for the training batch_size
    """
    with Path(pose_cfg).open('r') as f:
        cfg = yaml.safe_load(f)
    
    with Path(pose_cfg).open('w') as f:
        if epochs is not None:
            cfg['train_settings']['epochs'] = epochs
        if seed is not None:
            cfg['train_settings']['seed'] = seed
        if batch_size is not None:
            cfg['train_settings']['batch_size'] = batch_size
        if 'train_settings' in cfg.get('detector', {}) and detector_epochs is not None:
            cfg['detector']['train_settings']['epochs'] = detector_epochs
        yaml.safe_dump(cfg, f)