# RESNET-18-Robustness-in-MATLAB# Robustness Testing of ResNet-18 Image Classifier on CIFAR-10 Against Adversarial White-Box Attacks

This repository contains a series of experiments designed to evaluate the robustness of a ResNet-18 architecture trained on the CIFAR-10 dataset against adversarial white-box attacks. Specifically, the experiments focus on testing the model against Fast Gradient Sign Method (FGSM) and Projected Gradient Descent (PGD) attacks. The results highlight the trade-offs between standard accuracy on clean data and robustness to adversarial examples.

## Project Structure

- `resnet_18_v1/`: Contains the first model trained on clean CIFAR-10 data.
- `resnet_18_v2/`: Contains the model trained on adversarial CIFAR-10 data generated using FGSM.
- `resnet_18_v3/`: Contains the model trained on a mix of clean and adversarial CIFAR-10 data.
- `resnet_18_v4/`: Contains the model trained on adversarial CIFAR-10 data generated using a progressively stronger PGD attack.
- `runmatlab_v1.sh`: Job script for running the experiment `resnet_18_v1.m` on the ARC4 HPC cluster.
- `resnet_18_v1.m`: MATLAB script used for the first experiment.
- `resnet_18_v2.m`: MATLAB script used for the second experiment, including adversarial data generation using FGSM.
- `resnet_18_v3.m`: MATLAB script used for the third experiment.
- `resnet_18_v4.m`: MATLAB script used for the fourth experiment.

## Experiments

### Experiment 1: Training on Clean CIFAR-10 Data

**Objective:** To establish a baseline accuracy of ResNet-18 on clean CIFAR-10 data.

- **Model Name:** `resnet_18_v1.mat`
- **Training Data:** Clean CIFAR-10 data.
- **Results:**
  - **Validation Accuracy (Clean Data):** 91.58%
  - **Validation Accuracy (Adversarial Data):** 61.31%

This experiment serves as a control to evaluate the impact of adversarial training on model performance.

### Experiment 2: Training on 100% Adversarial CIFAR-10 Data (FGSM)

**Objective:** To test the robustness of ResNet-18 when trained entirely on adversarial examples.

- **Model Name:** `resnet_18_v2.mat`
- **Training Data:** 100% adversarial CIFAR-10 data generated using FGSM (`epsilon = 2`, `alpha = 2`, iteration step = 1).
- **Results:**
  - **Validation Accuracy (Clean Data):** 85.81%
  - **Validation Accuracy (Adversarial Data):** 76%

The training was performed using a white-box FGSM attack, aiming to create a model robust to this specific attack.

### Experiment 3: Training on Mixed Data (50% Clean, 50% Adversarial)

**Objective:** To balance normal validation accuracy with adversarial robustness by training on both clean and adversarial data.

- **Model Name:** `resnet_18_v3.mat`
- **Training Data:** 
  - First 50 epochs on clean CIFAR-10 data.
  - Next 50 epochs on adversarial CIFAR-10 data generated using FGSM (`epsilon = 2`, `alpha = 2`, iteration step = 1).
- **Results:**
  - **Validation Accuracy (Clean Data):** [To be updated]
  - **Validation Accuracy (Adversarial Data):** [To be updated]

This experiment aims to find a middle ground where the model retains decent accuracy on both clean and adversarial data.

### Experiment 4: Training on Progressively Stronger Adversarial Data (PGD)

**Objective:** To develop a model resilient to a strong PGD attack by progressively increasing the strength of the adversarial examples during training.

- **Model Name:** `resnet_18_v4.mat`
- **Training Data:** Adversarial CIFAR-10 data generated using a PGD attack with:
  - `epsilon` values increasing from 0 to 8.
  - Iteration steps increasing from 1 to 20 every 10 epochs.
- **Results:**
  - **Validation Accuracy (Clean Data):** [To be updated]
  - **Validation Accuracy (Adversarial Data):** [To be updated]

The objective was to train a model robust against a strong PGD attack, even at the cost of reduced accuracy on clean data.

## Usage on ARC4 High-Performance Computing Cluster

To replicate the experiment results on the ARC4 HPC cluster, follow these steps:

1. **Organize Files:**
   - Place the job script (`runmatlab_v1.sh`) and the corresponding MATLAB script (e.g., `resnet_18_v1.m`) in the same folder. 
   - Ensure that the folder contains all necessary files for running the specific experiment.

2. **Change Directory:**
   - Navigate to the directory containing the job script and MATLAB script using the `cd` command in your terminal:
     ```bash
     cd path/to/your/folder
     ```

3. **Submit Job:**
   - Submit the job to the ARC4 cluster using the `qsub` command:
     ```bash
     qsub runmatlab_v1.sh
     ```
   - This will queue the job on the HPC cluster, where it will execute the corresponding MATLAB script.

4. **Monitor Job:**
   - You can monitor the status of your job using the `qstat` command.

5. **Results:**
   - After the job completes, results and logs will be available in the same directory.

## Results and Discussion

- The results of these experiments demonstrate the trade-offs between standard accuracy and adversarial robustness.
- Training purely on adversarial data (Experiment 2) improves robustness but reduces accuracy on clean data.
- Mixed training (Experiment 3) attempts to strike a balance but requires careful tuning of the training process.
- Training with progressively stronger adversarial examples (Experiment 4) aims to create a robust model against a wide range of attacks, though it may lead to lower accuracy on non-adversarial data.

## References

- **Explaining and Harnessing Adversarial Examples** by Ian Goodfellow et al. (2015)
- **Towards Deep Learning Models Resistant to Adversarial Attacks** by Aleksander Madry et al. (2018)

## Contact

For any questions or feedback, please contact [Rajarshi Nandi](https://www.linkedin.com/in/rajarshi-nandi-a77aa5214/) at [mm23rn.leeds.ac.uk](mm23rn.leeds.ac.uk).
