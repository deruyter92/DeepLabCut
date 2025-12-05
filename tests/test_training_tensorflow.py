#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Please see AUTHORS for contributors.
# https://github.com/DeepLabCut/DeepLabCut/blob/master/AUTHORS
#
# Licensed under GNU Lesser General Public License v3.0
#
import os
from pathlib import Path
from unittest.mock import Mock, patch
import pytest

import deeplabcut
from deeplabcut.core.engine import Engine
from deeplabcut.pose_estimation_tensorflow.training import train_network


@pytest.fixture
def mock_config(tmp_path):
    """Create a mock config file."""
    config_path = tmp_path / "config.yaml"
    config_data = {
        "project_path": str(tmp_path),
        "TrainingFraction": [0.95],
    }
    return config_path, config_data


@pytest.fixture
def mock_pose_config_file(tmp_path):
    """Create a mock pose_cfg.yaml file."""
    model_folder = tmp_path / "DLC_model" / "train"
    model_folder.mkdir(parents=True)
    pose_config_file = model_folder / "pose_cfg.yaml"
    pose_config_file.write_text("dataset_type: single-animal\n")
    return pose_config_file


@pytest.fixture
def mock_multi_animal_pose_config_file(tmp_path):
    """Create a mock multi-animal pose_cfg.yaml file."""
    model_folder = tmp_path / "DLC_model" / "train"
    model_folder.mkdir(parents=True)
    pose_config_file = model_folder / "pose_cfg.yaml"
    pose_config_file.write_text("dataset_type: multi-animal\n")
    return pose_config_file


@patch("deeplabcut.pose_estimation_tensorflow.core.train.train")
@patch("deeplabcut.utils.auxiliaryfunctions.get_model_folder")
@patch("deeplabcut.utils.auxiliaryfunctions.read_plainconfig")
@patch("deeplabcut.utils.auxiliaryfunctions.read_config")
@patch("importlib.reload")
@patch("logging.shutdown")
def test_train_network_initialization_single_animal(
    mock_logging_shutdown,
    mock_reload,
    mock_read_config,
    mock_read_plainconfig,
    mock_get_model_folder,
    mock_train,
    mock_config,
    mock_pose_config_file,
    tmp_path,
):
    """Test initialization of train_network for single-animal training."""
    config_path, config_data = mock_config
    
    # Setup mocks
    mock_read_config.return_value = config_data
    mock_get_model_folder.return_value = "DLC_model"
    mock_read_plainconfig.return_value = {"dataset_type": "single-animal"}
    
    # Ensure pose config file exists
    model_folder = tmp_path / "DLC_model" / "train"
    model_folder.mkdir(parents=True, exist_ok=True)
    pose_config_file = model_folder / "pose_cfg.yaml"
    pose_config_file.write_text("dataset_type: single-animal\n")
    
    # Call train_network
    train_network(
        str(config_path),
        shuffle=1,
        trainingsetindex=0,
        max_snapshots_to_keep=5,
        allow_growth=True,
        gputouse=None,
    )
    
    # Verify initialization calls
    mock_read_config.assert_called_once_with(str(config_path))
    mock_get_model_folder.assert_called_once_with(
        config_data["TrainingFraction"][0], 1, config_data, modelprefix=""
    )
    mock_read_plainconfig.assert_called_once()
    assert str(pose_config_file) in str(mock_read_plainconfig.call_args[0][0])
    
    # Verify train was called with correct arguments
    mock_train.assert_called_once()
    call_args = mock_train.call_args
    assert str(pose_config_file) == call_args[0][0]
    assert call_args[1]["max_to_keep"] == 5
    assert call_args[1]["keepdeconvweights"] is True
    assert call_args[1]["allow_growth"] is True


@patch("deeplabcut.pose_estimation_tensorflow.core.train_multianimal.train")
@patch("deeplabcut.utils.auxiliaryfunctions.get_model_folder")
@patch("deeplabcut.utils.auxiliaryfunctions.read_plainconfig")
@patch("deeplabcut.utils.auxiliaryfunctions.read_config")
@patch("importlib.reload")
@patch("logging.shutdown")
def test_train_network_initialization_multi_animal(
    mock_logging_shutdown,
    mock_reload,
    mock_read_config,
    mock_read_plainconfig,
    mock_get_model_folder,
    mock_train,
    mock_config,
    mock_multi_animal_pose_config_file,
    tmp_path,
):
    """Test initialization of train_network for multi-animal training."""
    config_path, config_data = mock_config
    
    # Setup mocks
    mock_read_config.return_value = config_data
    mock_get_model_folder.return_value = "DLC_model"
    mock_read_plainconfig.return_value = {"dataset_type": "multi-animal"}
    
    # Ensure pose config file exists
    model_folder = tmp_path / "DLC_model" / "train"
    model_folder.mkdir(parents=True, exist_ok=True)
    pose_config_file = model_folder / "pose_cfg.yaml"
    pose_config_file.write_text("dataset_type: multi-animal\n")
    
    # Call train_network
    train_network(
        str(config_path),
        shuffle=1,
        trainingsetindex=0,
        max_snapshots_to_keep=3,
        allow_growth=False,
        gputouse=0,
    )
    
    # Verify initialization calls
    mock_read_config.assert_called_once_with(str(config_path))
    mock_get_model_folder.assert_called_once_with(
        config_data["TrainingFraction"][0], 1, config_data, modelprefix=""
    )
    mock_read_plainconfig.assert_called_once()
    
    # Verify multi-animal train was called
    mock_train.assert_called_once()
    call_args = mock_train.call_args
    assert str(pose_config_file) == call_args[0][0]
    assert call_args[1]["max_to_keep"] == 3
    assert call_args[1]["keepdeconvweights"] is True
    assert call_args[1]["allow_growth"] is False


@patch("deeplabcut.pose_estimation_tensorflow.core.train.train")
@patch("deeplabcut.create_project.new.VideoReader")
def test_full_pipeline_tensorflow(
    mock_video_reader,
    mock_train,
    tmp_path,
):
    """Test the full functional training pipeline from project creation to training.
    
    This test:
    1. Creates mock video files
    2. Creates an actual project with real config
    3. Creates mock labeled data (frames and annotations)
    4. Creates training dataset
    5. Mocks the training function to avoid actual training
    """
    import numpy as np
    import pandas as pd
    from PIL import Image
    
    # Setup mock video reader
    mock_video = Mock()
    mock_video.get_bbox.return_value = [0, 640, 0, 480]  # width, height bbox
    mock_video_reader.return_value = mock_video
    
    # Create mock video directory and video files
    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    
    video_1 = video_dir / "test_video_1.mp4"
    video_2 = video_dir / "test_video_2.mp4"
    
    # Create mock video files (just empty files with .mp4 extension)
    video_1.write_bytes(b"fake video content")
    video_2.write_bytes(b"fake video content")
    
    # Create actual project with real config
    working_dir = tmp_path / "projects"
    working_dir.mkdir()
    
    config_path = deeplabcut.create_new_project(
        project="test_project",
        experimenter="tester",
        videos=[str(video_1), str(video_2)],
        working_directory=str(working_dir),
        copy_videos=False,
        videotype="",
        multianimal=False,
        individuals=None,
    )
    
    # Verify project was created
    assert config_path != "nothingcreated"
    assert Path(config_path).exists()
    
    # Read the config to get project path
    from deeplabcut.utils import auxiliaryfunctions
    cfg = auxiliaryfunctions.read_config(config_path)
    project_path = Path(cfg["project_path"])
    
    # Create mock labeled data (frames and annotations)
    # This simulates the output of extract_frames and manual labeling
    video_name_1 = "test_video_1"
    video_name_2 = "test_video_2"
    
    labeled_data_dir = project_path / "labeled-data"
    video_data_dir_1 = labeled_data_dir / video_name_1
    video_data_dir_2 = labeled_data_dir / video_name_2
    
    video_data_dir_1.mkdir(parents=True, exist_ok=True)
    video_data_dir_2.mkdir(parents=True, exist_ok=True)
    
    # Create mock frame images
    num_frames = 5
    bodyparts = cfg["bodyparts"]
    scorer = cfg["scorer"]
    
    for frame_idx in range(num_frames):
        # Create mock frame images
        frame_img_1 = Image.new('RGB', (640, 480), color='gray')
        frame_img_1.save(video_data_dir_1 / f"img{frame_idx:03d}.png")
        
        frame_img_2 = Image.new('RGB', (640, 480), color='gray')
        frame_img_2.save(video_data_dir_2 / f"img{frame_idx:03d}.png")
    
    # Create mock annotation DataFrames (CollectedData files)
    for video_name, video_data_dir in [(video_name_1, video_data_dir_1), (video_name_2, video_data_dir_2)]:
        # Create DataFrame with mock annotations
        frames = sorted([f"labeled-data/{video_name}/img{i:03d}.png" for i in range(num_frames)])
        
        data_frames = []
        for bodypart in bodyparts:
            columnindex = pd.MultiIndex.from_product(
                [[scorer], [bodypart], ["x", "y"]],
                names=["scorer", "bodyparts", "coords"]
            )
            # Create mock coordinates (diagonal pattern for testing)
            coords = np.ones((num_frames, 2)) * 100 + np.arange(num_frames).reshape(-1, 1) * 50
            frame = pd.DataFrame(
                coords,
                columns=columnindex,
                index=frames,
            )
            data_frames.append(frame)
        
        # Combine all bodyparts
        dataFrame = pd.concat(data_frames, axis=1)
        
        # Save as CSV and H5 (both formats used by DLC)
        dataFrame.to_csv(video_data_dir / f"CollectedData_{scorer}.csv")
        dataFrame.to_hdf(
            video_data_dir / f"CollectedData_{scorer}.h5",
            key="df_with_missing",
            format="table",
            mode="w",
        )
    
    # Create training dataset
    deeplabcut.create_training_dataset(
        config=config_path,
        num_shuffles=1,
        net_type='resnet_50',
        engine=Engine.TF,
        userfeedback=False,  # Avoid prompts
    )
    
    # Train network (mocked to avoid actual training)
    deeplabcut.train_network(
        config=config_path,
        shuffle=1,
        epochs=1,
    )
    
    # Verify train was called (mocked, so no actual training)
    mock_train.assert_called_once()
