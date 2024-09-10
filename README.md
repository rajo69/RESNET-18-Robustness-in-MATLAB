# Enhancing-Neural-Network-Robustness-using-Hybrid-Adversarial-Training

This project addresses the challenge of improving neural network robustness against adversarial attacks, which can cause models to misclassify inputs with minimal perturbations. The primary problem is the trade-off between model accuracy on clean data and robustness to adversarial examples [(Tsipras D. et al., 2019)](https://arxiv.org/pdf/1805.12152). The proposed solution is a hybrid training approach that combines both clean and adversarial (FGSM) data, aiming to maintain strong performance on unperturbed inputs while enhancing the model's resilience to progressively stronger adversarial (FGSM & PGD) attacks.

## Fast Gradient Sign Method (FGSM) Attack

The Fast Gradient Sign Method (FGSM) attack is a type of adversarial attack on machine learning models, particularly neural networks. It works by adding small, carefully calculated perturbations to the input data in the direction of the gradient of the model's loss function, with respect to the input. This causes the model to misclassify the altered input, even though the perturbations are often imperceptible to humans. FGSM is widely used to evaluate the robustness of models against adversarial examples.

## Project Gradient Descent (PGD) Attack

The Projected Gradient Descent (PGD) attack is an iterative adversarial attack used to evaluate the robustness of machine learning models. It builds on the FGSM attack by applying multiple small perturbations to the input data over several iterations. After each step, the perturbed input is projected back into a constrained space to ensure the perturbation remains within a specified limit. This makes PGD a stronger and more powerful attack compared to FGSM, as it refines the adversarial example over multiple steps.

## Project Overview

The project involve training and testing a ResNet-18 mode using three different methodologies: 

1. **Experiment 1 (v1)** - Model trained solely on normal, unperturbed data.
2. **Experiment 2 (v2)** - Model trained exclusively on adversarial (FGSM) data.
3. **Experiment 3 (v3)** - Model trained using a hybrid approach of both normal and adversarial (FSGM) data.

## Experiments

### Data

The project was carried out in MATLAB R2022a and the dataset used for training and validation of is [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html).

### Requirements

To run these scripts, you need MATLAB R2022a or later installed, along with the following MATLAB toolboxes:
- Deep Learning Toolbox
- Parallel Computing Toolbox (for utilizing GPUs)

### Repository Structure

```bash
├── experiment_v1
│   ├── train_resnet_18_v1.m       # Training script for Experiment 1
│   ├── test_normal_v1.m           # Testing script on normal data (v1)
│   ├── test_fgsm_v1.m             # Testing script against FGSM adversarial data (v1)
│   ├── test_pgd_v1.m              # Testing script against PGD adversarial data (v1)
├── experiment_v2
│   ├── train_resnet_18_v2.m       # Training script for Experiment 2
│   ├── test_normal_v2.m           # Testing script on normal data (v2)
│   ├── test_fgsm_v2.m             # Testing script against FGSM adversarial data (v2)
│   ├── test_pgd_v2.m              # Testing script against PGD adversarial data (v2)
├── experiment_v3
│   ├── train_resnet_18_v3.m       # Training script for Experiment 3
│   ├── test_normal_v3.m           # Testing script on normal data (v3)
│   ├── test_fgsm_v3.m             # Testing script against FGSM adversarial data (v3)
│   ├── test_pgd_v3.m              # Testing script against PGD adversarial data (v3)
└── README.md                      # This README file
```

### Common Parameters & Preprocessing

Throughout all the four experiments, `LearnRate = 0.01`, `miniBatchSize = 128` and `MaxEpoch = 100` have been maintained for consistency.
In all the experiments, training data set has been scaled up from 32x32x3 to 224x224x3, randomly shifted vertically and horizontally by up to 4 pixels and flipped using `imageDataAugmenter` before being passed on to `miniBatchQueue`.

### Experiment 1: Training on Clean CIFAR-10 Data

**Objective:** To establish a baseline accuracy of ResNet-18 on clean CIFAR-10 data.

- **Model Name:** `resnet_18_v1.mat`
- **Results:**
    - Validation Accuracy on Normal Data: **91.58%**
    - Validation Accuracy under FGSM Attack: Varies between **61.31%** and **16.84%** depending on the strength of `epsilon`.
    - Validation Accuracy under PGD Attack: **0.57%** (`epsilon` = `alpha` = 8, iteration = 30)

This experiment serves as a control to evaluate the impact of adversarial training on model performance.

### Experiment 2: Training on 100% Adversarial Data (FGSM)

**Objective:** To test the robustness of ResNet-18 when trained entirely on adversarial examples.

- **Model Name:** `resnet_18_v2.mat`
- **Training Data:** 100% adversarial data generated using FGSM (`epsilon = 2`, `alpha = 2`, iteration step = 1).
- **Results:**
    - Validation Accuracy on Normal Data: **85.81%**
    - Validation Accuracy under FGSM Attack: Varies between **76.00%** and **46.01%**.
    - Validation Accuracy under PGD Attack: **38.75%** (`epsilon` = `alpha` = 8, iteration = 30)

The training was performed using exclusively adversarial data (FGSM), aiming to create a model robust to this specific attack.

### Experiment 3: Training on Mixed Data (50% Clean, 50% Adversarial)

**Objective:** To balance normal validation accuracy with adversarial robustness by training on both clean and adversarial data.

- **Model Name:** `resnet_18_v3.mat`
- **Training Data:** 
  - First 50 epochs on clean data.
  - Next 50 epochs on adversarial data generated using FGSM (`epsilon = 2`, `alpha = epsilon`, iteration step = 1).
- **Results:**
    - Validation Accuracy on Normal Data: **88.75%**
    - Validation Accuracy under FGSM Attack: Varies between **77.30%** and **43.78%**.
    - Validation Accuracy under PGD Attack: **35.66%** (`epsilon` = `alpha` = 8, iteration = 30)

The objective was to train a model robust against a strong PGD attack, even at the cost of reduced accuracy on clean data.

### Result Visaulization

![norm_acc_comp](https://github.com/user-attachments/assets/29f3f85f-dc50-42e9-b2db-59c7db64fbdc)
Validation accuracy of all models on normal data.

![comparative_adversarial_accuracy](https://github.com/user-attachments/assets/6a21f674-d9b8-4e83-b897-a8bfa9faa863)
Validation accuracy of all models on progressively stronger adversarial (FGSM) data.

![pgd_attack_comparison_final](https://github.com/user-attachments/assets/852ef8c0-6e50-4df8-8502-f6be57bb73f1)
Validation accuracy of all models on adversarial (PGD) data.

![grad_cam_op_all](https://github.com/user-attachments/assets/07294c07-2cde-493b-9399-040891b48204)
Grad-CAM output for all models

### Training the Models

To train the models for each experiment, navigate to the corresponding folder and run the training script. For example, to train the model for Experiment 1 (v1), use:

```matlab
cd experiment_v1
train_resnet_18_v1.m
```

This script will train the ResNet-18 model on the CIFAR-10 dataset. The trained model will be saved as `resnet_18_v1.mat`.

### Testing the Models on Normal Data

To test the model on normal, unperturbed CIFAR-10 data, use the following script after training:

```matlab
test_normal_v1.m
```

This will output the validation accuracy on clean data for the model.

### Testing the Models under Adversarial Attacks

1. **FGSM Testing**: To test the model against adversarial data generated using FGSM, run:

    ```matlab
    test_fgsm_v1.m
    ```

2. **PGD Testing**: To test the model against adversarial data generated using PGD, run:

    ```matlab
    test_pgd_v1.m
    ```

Replace `v1` with `v2` or `v3` for the other experiments. These scripts will calculate and display the model's accuracy under different adversarial perturbation strengths.

### Customizing the Adversarial Attacks

You can modify the strength of the adversarial perturbations `epsilon` in the testing scripts. For FGSM, the value of `epsilon` can be adjusted directly in the code:

```matlab
epsilon = 8; % Set the perturbation strength for FGSM
```

Similarly, for PGD, the number of iterations and the step size can be adjusted:

```matlab
epsilon = 8;  % Maximum allowed perturbation
alpha = 0.01; % Step size for each iteration
num_iterations = 40; % Number of iterations for PGD
```

## Key Findings from the Experiments

1. **Trade-off Between Accuracy and Robustness**: 
   - Models trained exclusively on normal data (Experiment 1) achieve high accuracy on clean inputs but are highly vulnerable to adversarial attacks, while models trained on adversarial data (Experiment 2) exhibit greater robustness but lower accuracy on clean data.

2. **Hybrid Training Provides a Balanced Solution**: 
   - The hybrid training approach (Experiment 3), which uses both clean and adversarial data, strikes a balance between accuracy and robustness, improving performance on adversarial examples while maintaining respectable accuracy on clean data.

3. **Stronger Adversarial Training Reduces Clean Data Performance**: 
   - As adversarial perturbations become stronger (higher epsilon values), models trained purely on adversarial data show a steady decline in performance on clean data, reinforcing the need for a balanced training strategy.

4. **Model Resilience Against PGD Attacks**: 
   - While models trained on normal data performed poorly under the PGD attack, the adversarially-trained and hybrid-trained models exhibited significantly better resilience, highlighting the effectiveness of adversarial training against stronger attacks.

5. **Visualization Insights Through Grad-CAM**: 
   - The Grad-CAM visualizations revealed that models trained with adversarial data (Experiments 2 and 3) focused on broader and more relevant regions of the input, suggesting that adversarial training enables models to learn more robust and meaningful feature representations.
  
## References

- **Explaining and Harnessing Adversarial Examples** by Ian Goodfellow et al. (2015)
- **Towards Deep Learning Models Resistant to Adversarial Attacks** by Aleksander Madry et al. (2018)
- **Robustness may be at odds with accuracy** by Dimitris Tsipras et al. (2019)

## Contact

For any questions or feedback, please contact [Rajarshi Nandi](https://www.linkedin.com/in/rajarshi-nandi-a77aa5214/) at [mm23rn.leeds.ac.uk](mm23rn.leeds.ac.uk).
