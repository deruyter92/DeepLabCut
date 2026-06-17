import pytest

from deeplabcut.pose_estimation_pytorch.config.make_pose_config import make_pytorch_pose_config
from deeplabcut.pose_estimation_pytorch.config.pose import PoseConfig


@pytest.mark.parametrize(
    "project_config",
    [
        "examples/openfield-Pranav-2018-10-30/config.yaml",
        "/home/jaap/Projects/DLC-Jaap/examples/Reaching-Mackenzie-2018-08-30/config.yaml",
    ],
)
@pytest.mark.parametrize("net_type", ["resnet_50", "resnet_101", "hrnet_w18", "hrnet_w32", "hrnet_w48"])
def test_for_project_matches_legacy_make_pytorch_pose_config(project_config: str, net_type: str):
    old = make_pytorch_pose_config(project_config, "cfg.yaml", net_type=net_type)
    new = PoseConfig.build(project_config, "cfg.yaml", net_type=net_type, top_down=False)
    assert old.to_dict() == new.to_dict()
