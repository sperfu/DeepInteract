# DeepInteract-for-CircRNA-Disease-Inference-
CircRNA-disease inference using deep ensemble model based on triple association
- DeepInteract_new.py is the main code using both supervised and unsupervised version of deep ensemble models
- Supervised version utilized autoencoder and deep neuron network to classify one circRNA to disease-related one or non-disease-related 
one.
- Unsupervised version of DeepInteract untilized two parts of autoencoders to cluster the samples into two groups representing disease-related circRNAs or non-disease-related circRNAs.
## Usage:
------
-THEANO_FLAGS=mode=FAST_RUN,device=gpu0,floatX=float32 python DeepInteract_new.py
using THEANO_FLAGS=mode=FAST_RUN,device=gpu0,floatX=float32 is for using gpu.

