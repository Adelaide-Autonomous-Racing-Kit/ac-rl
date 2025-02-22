# Assetto Corsa Renforcement Learning
Renforcement learning based control solutions using ACI.

## Installation
Make sure ACI is installed and working correctly by following the steps outlined [here](https://github.com/Adelaide-Autonomous-Racing-Kit/ac-interface?tab=readme-ov-file#installation)

### Custom Shaders Patch
For the specific method used to reset the agent back onto the track CSP is required.
Installation of CSP without using Content Manager is undocumented so it is recommend to follow the steps outlined [here](https://github.com/Adelaide-Autonomous-Racing-Kit/ac-interface?tab=readme-ov-file#opttional-setup-1)
Once installed you can then used the hambugrer menu in the top right to select `CSP settings` and install CSP.
The default version selected will not work and you need to manually select a newer version, we have tested v0.2.6.
Using Content Manage launch the game to make sure everything works and CSP is enabled.

### AARK Plugin
In order to leverage Custom Shaders Patch `resetCar` script you need to install and activate the Asseto Corsa AARK plugin found [here](https://github.com/Adelaide-Autonomous-Racing-Kit/ac-plugin).

## Usage
For more detailed documentation refer to our [website](https://adelaideautonomous.racing/docs-acrl/)

## Citation
If you use AC-RL in your research or quote our baselines please cite us in your work:
```BibTeX
@misc{bockman2024aarkopentoolkitautonomous,
      title={AARK: An Open Toolkit for Autonomous Racing Research}, 
      author={James Bockman and Matthew Howe and Adrian Orenstein and Feras Dayoub},
      year={2024},
      eprint={2410.00358},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2410.00358}, 
}
```
