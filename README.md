# Robustness Testing and Defenses of ResNet-18 Image Classifier on CIFAR-10 against Adversarial White-Box Attacks

This repository contains a series of experiments designed to train and evaluate the robustness of a ResNet-18 architecture trained on the CIFAR-10 dataset against adversarial white-box attacks. Specifically, the experiments focus on training the model and testing it against Fast Gradient Sign Method (FGSM) attacks. The results highlight the trade-offs between standard accuracy on clean data and robustness to adversarial examples. All the necessary files are in this [MATLAB Drive](https://drive.mathworks.com/sharing/cb389a73-b6a4-40ca-90fb-6c3c5b7aaa21) or [OneDrive](https://leeds365-my.sharepoint.com/:f:/g/personal/mm23rn_leeds_ac_uk/ElwFONIwIXZJrJ4z74KYqJQBHWhH2CdpmkmRxQvjS9LsuQ?e=IcZvGk) under folder 'MATLAB Project'.

NOTE: The experiments have been performed using NVIDIA Tesla V100 32GB GPU Accelerator and all MATLAB Scripts are GPU compatible if available.

## Prerequisite

- MATLAB R2022a
- HPC ARC Cluster access

## Project Structure

- `resnet_18_v1/Model`: Contains the first model trained on clean CIFAR-10 data.
- `resnet_18_v2/Model`: Contains the model trained on adversarial CIFAR-10 data generated using FGSM.
- `resnet_18_v3/Model`: Contains the model trained on a mix of clean and adversarial CIFAR-10 data.
- `resnet_18_v4/Model`: Contains the model trained on adversarial CIFAR-10 data generated using a progressively stronger PGD attack.
- `runmatlab_v1.sh`: Job script for running the experiment 1.
- `runmatlab_v2.sh`: Job script for running the experiment 2.
- `runmatlab_v3.sh`: Job script for running the experiment 3.
- `runmatlab_v4.sh`: Job script for running the experiment 4.

## Experiments

### Common Parameters & Preprocessing

Throughout all the four experiments, `LearnRate = 0.01`, `miniBatchSize = 128` and `MaxEpoch = 100` has been maintaied for consistency.
In all the experiments, training data set has been scaled up from 32x32x3 to 224x224x3, randomly shifted vertically and horizontally by upto 4 pixels and flipped using `ImageDataAugmenter` before being passed on to `miniBatchQueue`.

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
  - **Validation Accuracy (Clean Data):** 88.75%
  - **Validation Accuracy (Adversarial Data):** 77.3%

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
   - To get normal and adversarial validation accuracy just run `nrml_validation_test.m` or `adv_validation_test.m` with job script `runtest_nrml.sh` or `runtest_adv.sh` respectively by keeping both in the same directory.
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
- Mixed training (Experiment 3) attempts to strike a balance and achieves higher score than both the previous models v2 on both normal and adversarial validation but fails to achive normal validation accuracy of v1.
- Training with progressively stronger adversarial examples (Experiment 4) aims to create a robust model against a wide range of attacks, though it may lead to lower accuracy on non-adversarial data.

## References

- **Explaining and Harnessing Adversarial Examples** by Ian Goodfellow et al. (2015)
- **Towards Deep Learning Models Resistant to Adversarial Attacks** by Aleksander Madry et al. (2018)

## Contact

For any questions or feedback, please contact [Rajarshi Nandi](https://www.linkedin.com/in/rajarshi-nandi-a77aa5214/) at [mm23rn.leeds.ac.uk](mm23rn.leeds.ac.uk).
