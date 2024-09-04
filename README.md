# VUDNet: From Single Shot to Structure: End-to-End Network based Deflectometry for Specular Free-Form Surface Reconstruction

This project implements VUDNet, a novel deep neural network designed for the end-to-end 3D reconstruction of specular free-form surfaces using single-shot deflectometry. VUDNet innovatively combines discriminative and generative components—specifically a Variational Autoencoder (VAE) and a modified U-Net—to accurately interpret orthogonal fringe patterns and generate high-fidelity 3D surface reconstructions. The network excels in both depth estimation and detail refinement, achieving superior performance in challenging environments.

Key Features:

- Hybrid Architecture: Integrates a VAE for coarse depth estimation and a modified U-Net for fine detail refinement, ensuring high accuracy and generalization.
- Single-Shot Deflectometry: Captures complex surface geometries from single-shot 2D images, optimizing the reconstruction process for dynamic and real-time applications.
- Extensive Data Simulation: Uses Blender's Cycles engine to simulate a diverse dataset of specular surfaces, enhancing the network's robustness and applicability to real-world scenarios.

This project sets a new standard for single-shot deflectometry, demonstrating the potential of deep learning in advancing optical metrology for specular surfaces.
