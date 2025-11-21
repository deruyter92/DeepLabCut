## Functional integration testing with DeepLabCut

TL;DR This submodule can be used to test functional performance of DLC across different commits, to spot if there are 
unintended drops in performance after changing some code (regression testing).

Roughly `run_testcase.py` implements the following steps: 
1. Create DLC project folder from `dataset` template
2. Train a network with specified `net_type` and `detector` and `training` hyperparameters. 
3. Runs evaluation, and video_analysis on an example video in the dataset.
4. Compares the results with the expected `ref_output` for this testcase configuration. 

All that is needed is a testcase configuration consisting of `dataset`, `net_type`, `detector`, `training` hyperparameters and the expected `ref_output`. 

### Basic usage
To implement a new test, set up a new test configuration in `config/testcase/`, and put the expected output in `ref_output/`.
Run your test configuration, or a previously implemented one with the following command:
``` bash
python functional_integration/run_testcase.py testcase=my-testcase-name
```

The integration tests make use of [hydra](https://hydra.cc/), providing a highly a modular configuration structure.
To test any existing dataset with any net_type agains some expected output, you can just pass these as arguments

``` bash
python functional_integration/run_testcase.py \
testcase.net_type.name=resnet_50 \
testcase.dataset.name=trimice \
testcase.ref_output_path=path/to/my.pckl
```

A matrix of common test cases will be implemented soon.

### Configuration structure

To configure a new test case, you can create a new yaml file inside `config/testcase`, with the following structure:

```yaml
name: name-of-testcase
ref_output: path/to/expected/output.pckl

defaults: 
  - dataset: fly
  - net_type: resnet_50
  - detector: ssdlite
  - training: short
  - _self_
```
This hierarcical config structure imports all fields from predefined lower-level configs (e.g. dataset/fly.yaml, net_type/resnet_50.yaml).

You can also directly add a complete configuration yourself, instead of importing the fields:

```yaml
name: name-of-testcase
ref_output: path/to/expected/output.pckl

dataset:
  name: fly
  data_path: functional_testing/data/fly/
  eval_video: functional_testing/data/fly/videos/fly.mp4
net_type: 
  name: resnet_50
detector: 
  name: sddlite
training:
  seed: 42
  batch_size: 8
  epochs: 5
  detector_epochs: 5
```
