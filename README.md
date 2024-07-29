# Elastic Neural Network Full-Waveform Inversion (ENFWI)

## Introduction
Elastic Full-Waveform Inversion (EFWI) is a cutting-edge technique utilized for high-resolution inversion of velocity structures, such as P-wave and S-wave velocities. This method plays a crucial role in indirectly estimating subsurface material properties like porosity and saturation through established petrophysical relationships. However, traditional EFWI approaches often encounter challenges such as low signal-to-noise ratios (SNR) in near-surface seismic data and heterogeneous Poisson's ratios in complex environments. These issues can hinder convergence and introduce crosstalk noise during the inversion process.

## Our Solution
In this project, we introduce Elastic Neural Network Full-Waveform Inversion (ENFWI), an integrated approach that combines neural network-based seismic inversion with a petrophysical inversion framework. Our approach leverages:

- **Convolutional Neural Networks (CNNs)** and **Automatic Differentiation (AD)** to invert the $V_p$ (primary wave velocity) and $V_s$ (secondary wave velocity) models, efficiently suppressing crosstalk noise and enhancing robustness against low SNR conditions.
- **Hertz-Mindlin (HM) Contact Theory** in our petrophysical iterative inversion framework to update porosity and saturation properties based on the inverted velocity models, incorporating a resistivity constraint term to refine the estimations.

## Key Features
- **Dual CNN Architectures**: Utilizes two independent CNNs to generate $V_p$ and $V_s$ gradients, tailored to address the specific needs of each velocity type.
- **Advanced Petrophysical Modeling**: Integrates Hertz-Mindlin theory to iteratively update subsurface properties, ensuring that each inversion cycle refines these estimates based on the latest velocity models.
- **Field Data Validation**: Demonstrated capability to accurately identify critical subsurface features such as water-bearing interlayers and Wadi basement interfaces using field data.

## Framework Architecture
The ENFWI framework is constructed using Pytorch, optimizing computational efficiency and flexibility:
- **Forward and Backward Pass**: Defines specific behaviors for the forward and backward propagation, ensuring high computational efficiency and accuracy.
- **Automatic Differentiation (AD)**: Employs AD to construct objective functions and calculate gradients for backpropagation, simplifying the use of diverse objective functions.
- **ADHM Integration**: Uses AD to establish a conversion relationship between porosity, saturation, and velocity models, enhancing the inversion's accuracy and reliability.

## Applications
This framework is particularly suitable for imaging structural distributions and estimating properties in near-surface critical zones (CZ), where traditional methods often fall short. It offers significant improvements in the reliability and detail of subsurface imaging, crucial for environmental studies, resource exploration, and geotechnical assessments.

## Conclusion
The ENFWI project encompasses both the codebase and practical data used in the accompanying research paper. It represents a significant advancement in the field of geophysical inversion, providing a robust tool for researchers and professionals in geophysics to extract more accurate information from challenging datasets.

## How to Use
(Provide a brief guide or link to documentation on how to set up and run the inversion framework, including any dependencies, required configurations, and example usage.)

## Contributing
(Invite contributions from other developers. Provide guidelines on how they can contribute, report issues, and suggest enhancements.)

## License
(Specify the licensing information for your project.)

## Citation
(If applicable, provide a citation format for users who use this project for academic purposes.)

For more details, please refer to the accompanying research paper or contact the development team.
