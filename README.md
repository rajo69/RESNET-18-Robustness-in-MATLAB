# Enhancing-Neural-Network-Robustness-using-Hybrid-Adversarial-Training

This project addresses the challenge of improving neural network robustness against adversarial attacks, which can cause models to misclassify inputs with minimal perturbations. The primary problem is the trade-off between model accuracy on clean data and robustness to adversarial examples [(Tsipras D. et al., 2019)](https://arxiv.org/pdf/1805.12152). The proposed solution is a hybrid training approach that combines both clean and adversarial (FGSM) data, aiming to maintain strong performance on unperturbed inputs while enhancing the model's resilience to progressively stronger adversarial (FGSM & PGD) attacks.

## Fast Gradient Sign Method (FGSM) Attack

The Fast Gradient Sign Method (FGSM) attack is a type of adversarial attack on machine learning models, particularly neural networks. It works by adding small, carefully calculated perturbations to the input data in the direction of the gradient of the model's loss function, with respect to the input. This causes the model to misclassify the altered input, even though the perturbations are often imperceptible to humans. FGSM is widely used to evaluate the robustness of models against adversarial examples. This attack has been used in this project for both training and testing.

## Project Gradient Descent (PGD) Attack

The Projected Gradient Descent (PGD) attack is an iterative adversarial attack used to evaluate the robustness of machine learning models. It builds on the FGSM attack by applying multiple small perturbations to the input data over several iterations. After each step, the perturbed input is projected back into a constrained space to ensure the perturbation remains within a specified limit. This makes PGD a stronger and more powerful attack compared to FGSM, as it refines the adversarial example over multiple steps. This attack has been used in this project only for both testing.

## Project Overview

The project involve training and testing a ResNet-18 mode using three different methodologies:

![proj_dia](https://github.com/user-attachments/assets/d946c414-5c40-4fac-b1ff-c3f4063cece5)

1. **Experiment 1 (v1)** - Model trained solely on normal, unperturbed data.
2. **Experiment 2 (v2)** - Model trained exclusively on adversarial (FGSM) data.
3. **Experiment 3 (v3)** - Model trained using a hybrid approach of both normal and adversarial (FSGM) data.

## Experiments

### Model Architecture

The model architecture is based on ResNet-18, which uses residual blocks with shortcut connections to enable efficient learning in deeper networks. Part (a) in the figure illustrates the stacked convolutional layers, with shortcut connections (red arrows) bypassing each block to prevent vanishing gradients. Part (b) shows the residual block structure, where the input \( X \) is added back to the transformed output \( F(X) \), resulting in \( F(X) + X \). This design allows for faster convergence and improved performance.

![resnet_18_architecture](https://github.com/user-attachments/assets/51cc382b-ef0b-4d43-9ddd-ccb3c3144512)

### Data

The project was carried out in MATLAB R2022a and the dataset used for training and validation is [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html).

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

- **Model Name:** [`resnet_18_v1.mat`](https://leeds365-my.sharepoint.com/:u:/g/personal/mm23rn_leeds_ac_uk/Ec1muL4D9otIpPROkZbwTLkB56tOvUrWYt-S3V5u-4LrXg?e=wVUsdf)
- **Results:**
    - Validation Accuracy on Normal Data: **91.58%**
    - Validation Accuracy under FGSM Attack: Varies between **61.31%** and **16.84%** depending on the strength of `epsilon`.
    - Validation Accuracy under PGD Attack: **0.57%** (`epsilon` = `alpha` = 8, iteration = 30)

This experiment serves as a control to evaluate the impact of adversarial training on model performance.

### Experiment 2: Training on 100% Adversarial Data (FGSM)

**Objective:** To test the robustness of ResNet-18 when trained entirely on adversarial examples.

- **Model Name:** [`resnet_18_v2.mat`](https://leeds365-my.sharepoint.com/:u:/g/personal/mm23rn_leeds_ac_uk/ERFbx5yI4mFKmK_mStWvI28BMqSotgc_k6w4KIPgjkAB-g?e=zwPHfw)
- **Training Data:** 100% adversarial data generated using FGSM (`epsilon = 2`, `alpha = 2`, iteration step = 1).
- **Results:**
    - Validation Accuracy on Normal Data: **85.81%**
    - Validation Accuracy under FGSM Attack: Varies between **76.00%** and **46.01%**.
    - Validation Accuracy under PGD Attack: **38.75%** (`epsilon` = `alpha` = 8, iteration = 30)

The training was performed using exclusively adversarial data (FGSM), aiming to create a model robust to this specific attack.

### Experiment 3: Training on Mixed Data (50% Clean, 50% Adversarial)

**Objective:** To balance normal validation accuracy with adversarial robustness by training on both clean and adversarial data.

- **Model Name:** [`resnet_18_v3.mat`](https://leeds365-my.sharepoint.com/:u:/g/personal/mm23rn_leeds_ac_uk/EcxcOlYXlEJAu4GSPOdTr8MBlF_a8bxboEHNEhHbL47O5g?e=3zpAdc)
- **Training Data:** 
  - First 50 epochs on clean data.
  - Next 50 epochs on adversarial data generated using FGSM (`epsilon = 2`, `alpha = epsilon`, iteration step = 1).
- **Results:**
    - Validation Accuracy on Normal Data: **88.75%**
    - Validation Accuracy under FGSM Attack: Varies between **77.30%** and **43.78%**.
    - Validation Accuracy under PGD Attack: **35.66%** (`epsilon` = `alpha` = 8, iteration = 30)

The objective was to train a model robust against a strong PGD attack, even at the cost of reduced accuracy on clean data.

### Training the Models

To train the models for each experiment update `datadir` to the desired folder destination where CIFAR-10 dataset will be downloaded, navigate to the corresponding folder and run the training script. For example, to train the model for Experiment 1 (v1), use:

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

### Result Visaulization

![norm_acc_comp](https://github.com/user-attachments/assets/bfcd9bb5-1bed-441f-b531-1c7df744829d)
Validation accuracy of all models on normal data.

![comparative_adversarial_accuracy](https://github.com/user-attachments/assets/72c09d2b-7750-4f82-82a4-06310fa8941e)
Validation accuracy of all models on progressively stronger adversarial (FGSM) data.

![pgd_attack_comparison](https://github.com/user-attachments/assets/61cc394b-6fef-41a8-b3ee-8020a9de6f71)
Validation accuracy of all models on adversarial (PGD) data.

![grad_cam_op_all](https://github.com/user-attachments/assets/333ac9d0-ab0f-4568-9561-018ea0f2ef02)
Grad-CAM output for all models

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
  
**NOTE:** The detailed analysis can be found in the `Project_Report.pdf` file.
  
## References

- **Explaining and Harnessing Adversarial Examples** by Ian Goodfellow et al. (2015)
- **Towards Deep Learning Models Resistant to Adversarial Attacks** by Aleksander Madry et al. (2018)
- **Robustness may be at odds with accuracy** by Dimitris Tsipras et al. (2019)
- [**Compress Image Classification Network for Deployment to Resource-Constrained Embedded Devices - MATLAB & Simulink - MathWorks United Kingdom**](https://uk.mathworks.com/help/coder/ug/deploy-compressed-network-to-resource-constrained-devices.html?searchHighlight=downloadCIFARData&s_tid=srchtitle_support_results_7_downloadCIFARData)
- [**Train Image Classification Network Robust to Adversarial Examples - MATLAB & Simulink - MathWorks United Kingdom**](https://uk.mathworks.com/help/deeplearning/ug/train-network-robust-to-adversarial-examples.html)
- [**Grad-CAM Reveals the Why Behind Deep Learning Decisions - MATLAB & Simulink - MathWorks United Kingdom**](https://uk.mathworks.com/help/deeplearning/ug/gradcam-explains-why.html)

## Acknowledgments

Special thanks to [@luisacutillo78](https://github.com/luisacutillo78) & [@mikecroucher](https://github.com/mikecroucher) for their valuable feedback, guidance and support.

This work was undertaken on ARC4, part of the High Performance Computing facilities at the University of Leeds, UK.

## Contact

For any questions or feedback, please contact [Rajarshi Nandi](https://www.linkedin.com/in/rajarshi-nandi-a77aa5214/) at [mm23rn.leeds.ac.uk](mm23rn.leeds.ac.uk).
