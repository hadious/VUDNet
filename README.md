# VUDNet: From Single Shot to Structure: End-to-End Network based Deflectometry for Specular Free-Form Surface Reconstruction

This repository contains the implementation of VUDNet, a novel deep neural network designed for the end-to-end 3D reconstruction of specular free-form surfaces using single-shot deflectometry. VUDNet leverages both discriminative and generative components, specifically combining a Variational Autoencoder (VAE) with a modified U-Net, to perform depth estimation and fine-detail refinement. The network excels in challenging environments, producing highly accurate reconstructions from single-shot 2D images of reflective surfaces.

## Key Features
- **Hybrid Architecture**: VUDNet integrates a Variational Autoencoder (VAE) for coarse depth estimation and a modified U-Net for detail refinement.
- **Single-Shot Deflectometry**: The network accurately interprets fringe patterns reflected from specular surfaces, reconstructing complex geometries in a single shot.
- **Extensive Simulation Dataset**: The project includes a dataset generated using Blender, capturing diverse deformed surfaces and their depth maps.

## Paper
You can find more details in the accompanying [pre-print paper](https://www.preprints.org/manuscript/202409.1851/v1).

## Architecture
Below is the architecture diagram of VUDNet, showing the integration of the VAE and U-Net for 3D surface reconstruction of specular surfaces:

![VUDNet Architecture](VUDNet_Arch.jpg)

## Installation
To install the required dependencies for this project, use the `requirements.txt` file. You can install the dependencies by running the following command:

  pip install -r requirements.txt

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.



