# Predicting labels for microscope images

## Cytoplasm, label 0

## Model 5
### architecture
input layer: dimension is 100x100x3
first layer: convolution - 2 5x5 filters and a relu activation
second layer: max pool - 7x7 pool size 
third layer: flatten layer
fourth layer: Dense layer, with 10 or 100 neurons and relu activation
fifth layer: Dense output layer, with softmax activation

### search
I searched over learning rate=[.1, 1, 10], momentum=[0, .9], and fifth layer neurons=[10, 100] with 10% of the training data.
The results were not very sensitive to fifth layer neuron count. All combinations of learning rate
and momentum lead to no improvements in training or testing loss, except for lr=.1 and momentum=0.
Figures 1 through 4 below show results.

<img src="/readmePics/model5100neurons.png" alt="Best (100 neurons)" width="350"/> <img src="/readmePics/model510neurons.png" alt="Best (10 neurons)" width="350"/>
 <img src="/readmePics/model5noImprov.png" alt="typical (10 neurons)" width="350"/> <img src="/readmePics/model5noImprov1.png" alt="typical (100 neurons)" width="350"/>

Conclusion: model5, lr=.1, m=0, and fifth layer neurons=10 might be able to pick up a general pattern if I train it on more of the data

### train

Training this model with the above set of optimal training paramters does not give successful results. The training and validation losses during the training session are given below, as well as its precision and recall. 


## Model 6
### search 

input layer: dimension is 512x512X3
The following layers are: convolution(f), pooling, convolution(2f), pooling, flatten, dense(N), dense(2) where f is the number of filters in the given layer and N is the number of neurons. A search was conducted over the hyperparameters learning rate=[.1, 1], momentum=[0, .5], N=[2, 10], and f=[4,10]. The best results were for learning rate=.1, momentum=0, N=4, and f=10. Below the training and validation loss as well as the frequency of predicted 1's and actual 1's (presence of cyctoplasm=1) are given for the best hyperparameter sets.

<img src="/readmePics/model6_best.png" alt=".." width="350"/> <img src="/readmePics/model6_best1.png" alt=".." width="350"/>

Below are what the typical results looked like for most of the search space.

<img src="/readmePics/model6_typical.png" alt=".." width="350"/>

### train

With the above results in mind, I train a model to 5 epochs with hyperparameters learning rate=.1, momentum=0, N=10, f=10. The precision for this model was .92, while the recall was .68. The losses as a function of training epoch are depicted below. 
 
<img src="/readmePics/model6epochs5.png" alt=".." width="350"/> <img src="/readmePics/5-20-35-39model6weights.png" alt=".." width="350"/>

It is arguable that this model is doing nothing more than drawing from a bernoulli distribution with p = .46

The same model was also trained to 12 epochs, producing precision=.85 and recall=.80. Its training and validation loss as a function of epoch are given below. We do not see the signature shape of generalized learning - our validation curve is rather flat with no minimum. The training loss continues to improve, fitting the data in a way that will not generalize. 

<img src="/readmePics/model6epochs12.png" alt=".." width="350"/> <img src="/readmePics/6-19-39-43model6weights.png" alt=".." width="350"/>

## Model 7
### search

We next switch optimizers, from keras's 'sgd' to 'adam', as well as add regularization. This is known as model7. The architecture is the same as model6 and we conduct a search of N = [2, 10] and filters = [4, 10]. For rigor, we train four networks for each combination of hyperparamters. We find that a some set of hyperparamters give variable performance, and the the exact same shape curves are never observed. None of these results are particularly enticing. Below are a couple results. We see that 10 neurons in the final layer with four filters performs the most reliably. Two neurons in the final layer leads to an untrainable model. I conjecture that the number of neurons in the final layer must be larger for larger numbers of filters, and that if I increased the final layer neurons, I would get a better performance for 10 filters. Just a conjecture though.

10 neurons, 4 filters:

<img src="/readmePics/model7Search/10-4_1.png" alt=".." width="350"/> <img src="/readmePics/model7Search/10-4_2.png" alt=".." width="350"/>
<img src="/readmePics/model7Search/10-4_3.png" alt=".." width="350"/> <img src="/readmePics/model7Search/10-4_4.png" alt=".." width="350"/>

10 neurons, 10 filters:

<img src="/readmePics/model7Search/10-10_1.png" alt=".." width="350"/> <img src="/readmePics/model7Search/10-10_2.png" alt=".." width="350"/>
<img src="/readmePics/model7Search/10-10_3.png" alt=".." width="350"/> <img src="/readmePics/model7Search/10-10_4.png" alt=".." width="350"/>

2 neurons, 4 filters:

<img src="/readmePics/model7Search/2-10_1.png" alt=".." width="350"/> <img src="/readmePics/model7Search/2-10_2.png" alt=".." width="350"/>
<img src="/readmePics/model7Search/2-10_3.png" alt=".." width="350"/> <img src="/readmePics/model7Search/2-10_4.png" alt=".." width="350"/>

2 neurons, 10 filters:

<img src="/readmePics/model7Search/2-4_1.png" alt=".." width="350"/> <img src="/readmePics/model7Search/2-4_2.png" alt=".." width="350"/>
<img src="/readmePics/model7Search/2-4_3.png" alt=".." width="350"/> <img src="/readmePics/model7Search/2-4_4.png" alt=".." width="350"/>

### train

Sadly, our model fails to achieve any predictive power.

<img src="/readmePics/model7epochs12.png" alt=".." width="350"/> <img src="/readmePics/7-11-46-35model7weights.png" alt=".." width="350"/>

## Model 8
### search
We set the architecture of the network with N=10 and f=10, and search over the paramters associated with the adam optimizer: alpha (learning rate) = [.01, .1, 1], beta1=[.8, .9], beta2=[.999], epsilon=[.1, 1].

The results for learning rate = .01 are:

<img src="/readmePics/model8Search/01-1.png" alt=".." width="350"/> <img src="/readmePics/model8Search/01-2.png" alt=".." width="350"/>
<img src="/readmePics/model8Search/01-3.png" alt=".." width="350"/> <img src="/readmePics/model8Search/01-4.png" alt=".." width="350"/>

The results for learning rate = .1 are:

<img src="/readmePics/model8Search/1-1.png" alt=".." width="350"/> <img src="/readmePics/model8Search/1-2.png" alt=".." width="350"/>
<img src="/readmePics/model8Search/1-3.png" alt=".." width="350"/> <img src="/readmePics/model8Search/1-4.png" alt=".." width="350"/>

The results for learning rate = 1 are:

<img src="/readmePics/model8Search/10-1.png" alt=".." width="350"/> <img src="/readmePics/model8Search/10-2.png" alt=".." width="350"/>
<img src="/readmePics/model8Search/10-3.png" alt=".." width="350"/> <img src="/readmePics/model8Search/10-4.png" alt=".." width="350"/>

### train 

[lr, beta1, beta2, epsilon]=[.1, .8, .999, 1] looks like a good model. Lets train it on 28000 images over 12 epochs.

<img src="/readmePics/9-12model8epochs12.png" alt=".." width="350"/> <img src="/readmePics/9-12model8weights.png" alt=".." width="350"/>

Sadly, this is a poor predictor. It achieves a recall of .72 and a precision of .81 (measured on the validation data, images 28000-31000) and it appears that its score on the validation loss never improves.

But the search results were promising for these parameters. When we trained to four epochs during the search with these parameters, I saw the validation loss drop to 60% of its original value, but when I train just this model, I see a drop to only 90%. That doesn't make sense. The only difference is the numbr of images we are training on. So lets train to 12 epochs but just use 2800 images to train on, as in the training session. We should at least see the same drop in the validation loss. 

<img src="/readmePics/9-16model8light12epochs.png" alt=".." width="350"/> <img src="/readmePics/9-16model8lightweights.png" alt=".." width="350"/>

We see an improved validation loss, but the performance is poor. Recall is .62 and Precision is .87.

## Model 9 - pretrained model

### Search

### Train