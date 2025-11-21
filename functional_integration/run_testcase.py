#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Please see AUTHORS for contributors.
# https://github.com/DeepLabCut/DeepLabCut/blob/main/AUTHORS
#
# Licensed under GNU Lesser General Public License v3.0
#

"""
Script for running a representative training instance to compare
functional performance with reference values. Used to detect unintended
regression of performance across commits.  
"""
from pathlib import Path
import logging
import pickle

import hydra
from omegaconf import OmegaConf
import pickle

from init_testcase import TestCase, init_project_for_testcase
import deeplabcut.pose_estimation_pytorch as dlc_torch


@hydra.main(
    version_base=None,
    config_path='config/',
    config_name='config',
)
def main(config):
    """
    Perform a training run for a test-case configured using Hydra. 
    Example usage (CLI):
        python tests/functional_testing/run_testcase.py testcase=fly-resnet_50-sddlite
    """
    # Get the logger for the current file (created by hydra)
    log = logging.getLogger(__name__)
    log.info(f"Initialing run for testcase {OmegaConf.select(config,'testcase.name')}")

    # Validate the test case before running (checks for valid config and ref_output)
    test_case = TestCase(config, validate=config.validate)

    # Use the output dir created by hydra
    hydra_config = hydra.core.hydra_config.HydraConfig.get()
    project_path = Path(hydra_config.run.dir) / test_case.name
    project_path.mkdir(parents=True, exist_ok=False)
    
    # Perform training run for this testcase
    cfg = config.testcase
    init_project_for_testcase(
        project_path,
        template_dataset=cfg.dataset.data_path,
        net_type=cfg.net_type.name,
        detector_type=cfg.detector.name,
        epochs=cfg.training.epochs,
        detector_epochs=cfg.training.detector_epochs,
        seed=cfg.training.seed,
        batch_size=cfg.training.batch_size,
    )

    # Train network
    dlc_torch.train_network(
        config=str(project_path / 'config.yaml'),
        shuffle=1,
        trainingsetindex=0,
    )
    test_case.store_train_results(project_path=project_path)

    log.info(f"Starting evaluation")
    dlc_torch.evaluate_network(
        config=str(project_path / 'config.yaml'),
        shuffles=[1],
        trainingsetindex=0,
    )
    test_case.store_eval_results(project_path=project_path)

    log.info(f"Analyzing example video: {cfg.dataset.eval_video}")
    scorer = dlc_torch.analyze_videos(
        config=str(project_path / "config.yaml"),
        videos=[cfg.dataset.eval_video],
        shuffle=1,
        save_as_csv=True,
        destfolder=str(project_path / 'video_analysis'),
    )
    test_case.store_analysis_results(
        project_path=project_path,
        video_name=Path(cfg.dataset.eval_video).stem,
        scorer=scorer,
    )


    output_fn = Path(project_path).parent / f'{test_case.name}.pckl'
    log.info(f"Saving functional performance as: {output_fn}")
    test_case.save_pickle(data=test_case._results, fn=output_fn) 

    test_result = test_case.compare_performance()


if __name__ == '__main__':
    main()