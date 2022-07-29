In this project, I implemented an MLP model with pytorch framework for a multi-class classification task. Then I experimented with various settings and analysed the effects of various settings in order to come up with the best setupt, my experimentations included:
1. Effects of various weight initialization methods. Inaccuraty weight inirialization can lead to vanishing gradients, eploding gradients or symmetry updating. I experimnetd with the effects of zero-constant and uniform initializations.
2. Effetcs of various loss functions. I examined the effects of Sigmoid and ReLU activation functions.
3. Effets of various number of layers and neurons.
4. Implementing data augmentation techniques and drop out layers to prevent overfitting.
5. Fine-Tuning hyper-parameters.

**In the end, I achieved test accuray of 95.83% and train accuracy of 97.59%.**
