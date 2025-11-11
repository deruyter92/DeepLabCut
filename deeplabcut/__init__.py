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

from __future__ import annotations

import importlib
from typing import Any, Dict, Tuple
import os

DEBUG = True and "DEBUG" in os.environ and os.environ["DEBUG"]
from deeplabcut.version import __version__, VERSION

print(f"Loading DLC {VERSION}...")

# Lazy public API to speed up `import deeplabcut`.
# Attributes are imported on first access via __getattr__ (PEP 562).
_LAZY_ATTRS: Dict[str, Tuple[str, str]] = {
    # Core engine
    "Engine": ("deeplabcut.core.engine", "Engine"),
    
    # Create project
    "create_new_project": ("deeplabcut.create_project", "create_new_project"),
    "create_new_project_3d": ("deeplabcut.create_project", "create_new_project_3d"),
    "add_new_videos": ("deeplabcut.create_project", "add_new_videos"),
    "load_demo_data": ("deeplabcut.create_project", "load_demo_data"),
    "create_pretrained_project": ("deeplabcut.create_project", "create_pretrained_project"),
    "create_pretrained_human_project": ("deeplabcut.create_project", "create_pretrained_human_project"),
    
    # Generate training dataset
    "check_labels": ("deeplabcut.generate_training_dataset", "check_labels"),
    "create_training_dataset": ("deeplabcut.generate_training_dataset", "create_training_dataset"),
    "extract_frames": ("deeplabcut.generate_training_dataset", "extract_frames"),
    "mergeandsplit": ("deeplabcut.generate_training_dataset", "mergeandsplit"),
    "create_training_dataset_from_existing_split": (
        "deeplabcut.generate_training_dataset",
        "create_training_dataset_from_existing_split",
    ),
    "create_training_model_comparison": (
        "deeplabcut.generate_training_dataset",
        "create_training_model_comparison",
    ),
    "create_multianimaltraining_dataset": (
        "deeplabcut.generate_training_dataset",
        "create_multianimaltraining_dataset",
    ),
    "dropannotationfileentriesduetodeletedimages": (
        "deeplabcut.generate_training_dataset",
        "dropannotationfileentriesduetodeletedimages",
    ),
    "comparevideolistsanddatafolders": (
        "deeplabcut.generate_training_dataset",
        "comparevideolistsanddatafolders",
    ),
    "dropimagesduetolackofannotation": (
        "deeplabcut.generate_training_dataset",
        "dropimagesduetolackofannotation",
    ),
    "adddatasetstovideolistandviceversa": (
        "deeplabcut.generate_training_dataset",
        "adddatasetstovideolistandviceversa",
    ),
    "dropduplicatesinannotatinfiles": (
        "deeplabcut.generate_training_dataset",
        "dropduplicatesinannotatinfiles",
    ),
    "dropunlabeledframes": (
        "deeplabcut.generate_training_dataset",
        "dropunlabeledframes",
    ),
    
    # Utils
    "create_labeled_video": ("deeplabcut.utils", "create_labeled_video"),
    "create_video_with_all_detections": ("deeplabcut.utils", "create_video_with_all_detections"),
    "plot_trajectories": ("deeplabcut.utils", "plot_trajectories"),
    "auxiliaryfunctions": ("deeplabcut.utils", "auxiliaryfunctions"),
    "convert2_maDLC": ("deeplabcut.utils", "convert2_maDLC"),
    "convertcsv2h5": ("deeplabcut.utils", "convertcsv2h5"),
    "analyze_videos_converth5_to_csv": ("deeplabcut.utils", "analyze_videos_converth5_to_csv"),
    "analyze_videos_converth5_to_nwb": ("deeplabcut.utils", "analyze_videos_converth5_to_nwb"),
    "auxfun_videos": ("deeplabcut.utils", "auxfun_videos"),
    
    # Utils auxfun_videos
    "ShortenVideo": ("deeplabcut.utils.auxfun_videos", "ShortenVideo"),
    "DownSampleVideo": ("deeplabcut.utils.auxfun_videos", "DownSampleVideo"),
    "CropVideo": ("deeplabcut.utils.auxfun_videos", "CropVideo"),
    "check_video_integrity": ("deeplabcut.utils.auxfun_videos", "check_video_integrity"),
    
    # Pose tracking pytorch (optional)
    "transformer_reID": ("deeplabcut.pose_tracking_pytorch", "transformer_reID"),
    
    # Pose estimation 3D
    "calibrate_cameras": ("deeplabcut.pose_estimation_3d", "calibrate_cameras"),
    "check_undistortion": ("deeplabcut.pose_estimation_3d", "check_undistortion"),
    "triangulate": ("deeplabcut.pose_estimation_3d", "triangulate"),
    "create_labeled_video_3d": ("deeplabcut.pose_estimation_3d", "create_labeled_video_3d"),
    
    # Refine training dataset
    "stitch_tracklets": ("deeplabcut.refine_training_dataset.stitch", "stitch_tracklets"),
    "extract_outlier_frames": ("deeplabcut.refine_training_dataset", "extract_outlier_frames"),
    "merge_datasets": ("deeplabcut.refine_training_dataset", "merge_datasets"),
    "find_outliers_in_raw_data": ("deeplabcut.refine_training_dataset", "find_outliers_in_raw_data"),
    
    # Post processing
    "filterpredictions": ("deeplabcut.post_processing", "filterpredictions"),
    "analyzeskeleton": ("deeplabcut.post_processing", "analyzeskeleton"),
    
    # GUI features (require Qt binding)
    "refine_tracklets": ("deeplabcut.gui.tracklet_toolbox", "refine_tracklets"),
    "label_frames": ("deeplabcut.gui.tabs.label_frames", "label_frames"),
    "refine_labels": ("deeplabcut.gui.tabs.label_frames", "refine_labels"),
    "SkeletonBuilder": ("deeplabcut.gui.widgets", "SkeletonBuilder"),
    "launch_dlc": ("deeplabcut.gui.launch_script", "launch_dlc"),

    # TensorFlow-compat entry points
    "train_network": ("deeplabcut.compat", "train_network"),
    "return_train_network_path": ("deeplabcut.compat", "return_train_network_path"),
    "evaluate_network": ("deeplabcut.compat", "evaluate_network"),
    "return_evaluate_network_data": (
        "deeplabcut.compat",
        "return_evaluate_network_data",
    ),
    "analyze_videos": ("deeplabcut.compat", "analyze_videos"),
    "create_tracking_dataset": ("deeplabcut.compat", "create_tracking_dataset"),
    "analyze_images": ("deeplabcut.compat", "analyze_images"),
    "analyze_time_lapse_frames": (
        "deeplabcut.compat",
        "analyze_time_lapse_frames",
    ),
    "convert_detections2tracklets": (
        "deeplabcut.compat",
        "convert_detections2tracklets",
    ),
    "extract_maps": ("deeplabcut.compat", "extract_maps"),
    "visualize_scoremaps": ("deeplabcut.compat", "visualize_scoremaps"),
    "visualize_locrefs": ("deeplabcut.compat", "visualize_locrefs"),
    "visualize_paf": ("deeplabcut.compat", "visualize_paf"),
    "extract_save_all_maps": ("deeplabcut.compat", "extract_save_all_maps"),
    "export_model": ("deeplabcut.compat", "export_model"),
    
    # Model zoo (may pull torch/huggingface/timm)
    "video_inference_superanimal": (
        "deeplabcut.modelzoo.video_inference",
        "video_inference_superanimal",
    ),
}


__all__ = ["__version__", "VERSION", "DEBUG"] + sorted(list(_LAZY_ATTRS.keys()))


def __getattr__(name: str) -> Any:
    if name in _LAZY_ATTRS:
        module_path, attr_name = _LAZY_ATTRS[name]
        try:
            module = importlib.import_module(module_path)
        except ModuleNotFoundError as exc:
            if module_path.startswith("deeplabcut.gui"):
                raise ModuleNotFoundError(
                    "DLC loaded in light mode; you cannot use any GUI (labeling, relabeling and standalone GUI). "
                    "Install extras `pip install deeplabcut[gui]` "
                    "and ensure a Qt binding (PySide6) is installed."
                ) from exc
            if module_path.startswith("deeplabcut.compat"):
                raise ModuleNotFoundError(
                    "TensorFlow features are unavailable. Install TF extras "
                    "e.g. `pip install deeplabcut[tf]` (platform-specific)."
                ) from exc
            if module_path.startswith("deeplabcut.pose_tracking_pytorch"):
                import warnings
                warnings.warn(
                    "As PyTorch is not installed, unsupervised identity learning will not be available. "
                    "Please run `pip install torch`, or ignore this warning."
                )
                raise
            raise
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'deeplabcut' has no attribute '{name}'")


def __dir__() -> list[str]:
    # Return public attributes (not starting with _) and all lazy attributes
    # Include special attributes like __version__
    public_attrs = [
        k for k in globals().keys() 
        if not k.startswith("_") or k == "__version__"
    ]
    return sorted(set(public_attrs) | set(_LAZY_ATTRS.keys()))