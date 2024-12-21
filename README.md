## About
This is a fork of the repository used in "AdvReverb: Rethinking the Stealthiness of Audio Adversarial Examples to Human Perception" by Chen, et al.
This repo is for a class project in the course CS:4980:0005 Fall24 Topics in Computer Science II - Adversarial Machine Learning at The University of Iowa.

### Changes to the AdvReverb Algorithm
We have made several changes to the AdvReverb algorithm implementation to improve its effectiveness and stealthiness. Those changes are summarized below:

1. **Incorporation of Perceptual Loss Functions**
    - Issue
      - Auditory Perception Alignment: The original implementation used Mean Squared Error (MSE) loss, which does not adequately model human auditory perception.
      - Perceptual Metric Usage: Integrated the Short-Time Objective Intelligibility (STOI) metric as a perceptual loss function by adding a perceptual_loss method.
      - Improved Stealthiness: Aligns the optimization process with human auditory perception, making adversarial perturbations less perceptible.
      - Quality Assurance: Enhances the naturalness of the adversarial examples by focusing on perceptual audio quality.

2. **Adaptive Penalty Weighting**
   - Fixed Penalty Limitations: Using fixed weights for penalty terms may not effectively balance attack success and audio quality throughout the optimization.
   - Dynamic Weight Adjustment: Introduced adaptive weighting for the loss terms based on the value of the perceptual loss. Adjusts the rir_loss_weight during optimization:
   - Increases weight when perceptual loss is high (audio difference is noticeable).
   - Decreases weight when perceptual loss is low (audio difference is minimal).
   - Balanced Optimization: Dynamically balances the importance of attacking effectiveness and audio imperceptibility.
   - Customization: Allows for fine-tuning based on specific thresholds and weight factors defined in the configuration.

3. **Adaptive Learning Rate Scheduling**
   - Optimization Efficiency: A fixed learning rate may lead to slow convergence or overshooting minima during optimization.
   - Learning Rate Scheduler: Implemented ReduceLROnPlateau from PyTorch to adjust the learning rate based on the total loss:
   - Reduces the learning rate when the loss plateaus.
   - Helps in fine-tuning the perturbations during later stages of optimization.
   - Improved Convergence: Enhances the optimizer's ability to find a minimum efficiently.
   - Stability: Prevents oscillations and promotes smoother convergence.

4. **Phoneme Importance Weighting**
   - Uniform Perturbation Application: Treating all phonemes equally may not be optimal since some are more perceptible than others.
   - Weight Assignment: Assigned weights to phonemes based on their perceptibility:
   - Less Perceptible Phonemes: Given higher weights (e.g., unvoiced consonants like 'k', 't', 'p', 's').
   - More Perceptible Phonemes: Given lower weights (e.g., vowels and voiced consonants).
   - Targeted Perturbations: Focuses adversarial perturbations on less perceptible phonemes, reducing the likelihood of detection by human listeners.
   - Enhanced Stealthiness: Improves the naturalness of the adversarial audio by minimizing alterations to perceptually significant phonemes.

## Sources
  - https://zju-muslab.github.io/projects/advreverb/
  - https://github.com/zju-muslab/AdvReverb/
