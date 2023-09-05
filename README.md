# DARL reproduction
This codebase is supposed to reproduce the results achieved in https://arxiv.org/pdf/2102.07097.pdf. 

tl;dr:
We want to train an SAC model based on pixel input. In order to achieve that we're gonna need an encoder that will map the image into latent space. The encoder is going to be shared between the actor and Q-network, although during training of the actor the encoder is frozen. 

The encoder will be jointly trained with Q-network and additional classifier. The classifier will receive image embedding and try to classify it to the domain it belongs. Between classifier module and encoder module there'll be a Gradient Reversal layer - it's supposed to make the encoder domain-invariant my maximizing domain classification loss. 

The rest is a standard SAC training (with some data augmentation: https://arxiv.org/pdf/2004.14990.pdf) for continuous domain.

Currently the code is adapted for discrete domain - namely Minigrid environment.
## Spin it up
You can train the model with the command `python train.py`. The config is supported by Hydra library and you can edit it in cfg/default.yaml. In addition, the config also includes configuration for Optima plugin, which allows for automatic fine-tuning. In order to run a series experiments to find the best hyperparameters you can run the command `python train.py --multirun`.

# TODO

- [x] Train SAC to overfit in basic environment.
- [ ] Add a shared pixel encoder.
- [ ] Add a code for evaluation on test environment in the training loop.
- [ ] Setup saving of the model and inference.
- [ ] Clean the code.