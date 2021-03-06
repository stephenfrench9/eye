# Predicting labels for microscope images


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

Woo I can type lots of stuff really fast using macdown.

<img src="./readmePics/model5100neurons.png" alt="Best (100 neurons)" width="350"/> <img src="./readmePics/model510neurons.png" alt="Best (10 neurons)" width="350"/>
 <img src="./readmePics/model5noImprov.png" alt="typical (10 neurons)" width="350"/> <img src="./readmePics/model5noImprov1.png" alt="typical (100 neurons)" width="350"/>

Conclusion: model5, lr=.1, m=0, and fifth layer neurons=10 might be able to pick up a general pattern if I train it on more of the data

### train

Training this model with the above set of optimal training paramters does not give successful results. The training and validation losses during the training session are given below, as well as its precision and recall. 


## Model 6
### search 

input layer: dimension is 512x512X3
The following layers are: convolution(f), pooling, convolution(2f), pooling, flatten, dense(N), dense(2) where f is the number of filters in the given layer and N is the number of neurons. A search was conducted over the hyperparameters learning rate=[.1, 1], momentum=[0, .5], N=[2, 10], and f=[4,10]. The best results were for learning rate=.1, momentum=0, N=4, and f=10. Below the training and validation loss as well as the frequency of predicted 1's and actual 1's (presence of cyctoplasm=1) are given for the best hyperparameter sets.

<img src="./readmePics/model6_best.png" alt=".." width="350"/> <img src="./readmePics/model6_best1.png" alt=".." width="350"/>

Below are what the typical results looked like for most of the search space.

<img src="./readmePics/model6_typical.png" alt=".." width="350"/>

### train

With the above results in mind, I train a model to 5 epochs with hyperparameters learning rate=.1, momentum=0, N=10, f=10. The precision for this model was .92, while the recall was .68. The losses as a function of training epoch are depicted below. 
 
<img src="./readmePics/model6epochs5.png" alt=".." width="350"/> <img src="./readmePics/5-20-35-39model6weights.png" alt=".." width="350"/>

It is arguable that this model is doing nothing more than drawing from a bernoulli distribution with p = .46

The same model was also trained to 12 epochs, producing precision=.85 and recall=.80. Its training and validation loss as a function of epoch are given below. We do not see the signature shape of generalized learning - our validation curve is rather flat with no minimum. The training loss continues to improve, fitting the data in a way that will not generalize. 

<img src="./readmePics/model6epochs12.png" alt=".." width="350"/> <img src="./readmePics/6-19-39-43model6weights.png" alt=".." width="350"/>

## Model 7
### search

We next switch optimizers, from keras's 'sgd' to 'adam', as well as add regularization. This is known as model7. The architecture is the same as model6 and we conduct a search of N = [2, 10] and filters = [4, 10]. For rigor, we train four networks for each combination of hyperparamters. We find that a some set of hyperparamters give variable performance, and the the exact same shape curves are never observed. None of these results are particularly enticing. Below are a couple results. We see that 10 neurons in the final layer with four filters performs the most reliably. Two neurons in the final layer leads to an untrainable model. I conjecture that the number of neurons in the final layer must be larger for larger numbers of filters, and that if I increased the final layer neurons, I would get a better performance for 10 filters. Just a conjecture though.

10 neurons, 4 filters:

<img src="./readmePics/model7Search/10-4_1.png" alt=".." width="350"/> <img src="./readmePics/model7Search/10-4_2.png" alt=".." width="350"/>
<img src="./readmePics/model7Search/10-4_3.png" alt=".." width="350"/> <img src="./readmePics/model7Search/10-4_4.png" alt=".." width="350"/>

10 neurons, 10 filters:

<img src="./readmePics/model7Search/10-10_1.png" alt=".." width="350"/> <img src="./readmePics/model7Search/10-10_2.png" alt=".." width="350"/>
<img src="./readmePics/model7Search/10-10_3.png" alt=".." width="350"/> <img src="./readmePics/model7Search/10-10_4.png" alt=".." width="350"/>

2 neurons, 4 filters:

<img src="./readmePics/model7Search/2-10_1.png" alt=".." width="350"/> <img src="./readmePics/model7Search/2-10_2.png" alt=".." width="350"/>
<img src="./readmePics/model7Search/2-10_3.png" alt=".." width="350"/> <img src="./readmePics/model7Search/2-10_4.png" alt=".." width="350"/>

2 neurons, 10 filters:

<img src="./readmePics/model7Search/2-4_1.png" alt=".." width="350"/> <img src="./readmePics/model7Search/2-4_2.png" alt=".." width="350"/>
<img src="./readmePics/model7Search/2-4_3.png" alt=".." width="350"/> <img src="./readmePics/model7Search/2-4_4.png" alt=".." width="350"/>

### train

Sadly, our model fails to achieve any predictive power.

<img src="./readmePics/model7epochs12.png" alt=".." width="350"/> <img src="./readmePics/7-11-46-35model7weights.png" alt=".." width="350"/>

## Model 8
### search
We set the architecture of the network with N=10 and f=10, and search over the paramters associated with the adam optimizer: alpha (learning rate) = [.01, .1, 1], beta1=[.8, .9], beta2=[.999], epsilon=[.1, 1].

The results for learning rate = .01 are:

<img src="./readmePics/model8Search/01-1.png" alt=".." width="350"/> <img src="./readmePics/model8Search/01-2.png" alt=".." width="350"/>
<img src="./readmePics/model8Search/01-3.png" alt=".." width="350"/> <img src="./readmePics/model8Search/01-4.png" alt=".." width="350"/>

The results for learning rate = .1 are:

<img src="./readmePics/model8Search/1-1.png" alt=".." width="350"/> <img src="./readmePics/model8Search/1-2.png" alt=".." width="350"/>
<img src="./readmePics/model8Search/1-3.png" alt=".." width="350"/> <img src="./readmePics/model8Search/1-4.png" alt=".." width="350"/>

The results for learning rate = 1 are:

<img src="./readmePics/model8Search/10-1.png" alt=".." width="350"/> <img src="./readmePics/model8Search/10-2.png" alt=".." width="350"/>
<img src="./readmePics/model8Search/10-3.png" alt=".." width="350"/> <img src="./readmePics/model8Search/10-4.png" alt=".." width="350"/>

### train 

[lr, beta1, beta2, epsilon]=[.1, .8, .999, 1] looks like a good model. Lets train it on 28000 images over 12 epochs.

<img src="./readmePics/9-12model8epochs12.png" alt=".." width="350"/> <img src="./readmePics/9-12model8weights.png" alt=".." width="350"/>

Sadly, this is a poor predictor. It achieves a recall of .72 and a precision of .81 (measured on the validation data, images 28000-31000) and it appears that its score on the validation loss never improves.

But the search results were promising for these parameters. When we trained to four epochs during the search with these parameters, I saw the validation loss drop to 60% of its original value, but when I train just this model, I see a drop to only 90%. That doesn't make sense. The only difference is the numbr of images we are training on. So lets train to 12 epochs but just use 2800 images to train on, as in the training session. We should at least see the same drop in the validation loss. 

<img src="./readmePics/9-16model8light12epochs.png" alt=".." width="350"/> <img src="./readmePics/9-16model8lightweights.png" alt=".." width="350"/>

We see an improved validation loss, but the performance is poor. Recall is .62 and Precision is .87.

## Model 12 - Resnet18, with pretrained weights (training session 15-12-12/)

### Train

Resnet18 is a deep residual network used for image classification. A deeper version of it won the ILSVRC 2015 classification task. This implementation is adapted from this repository: https://github.com/qubvel/classification_models. The image size for the 2015 ImageNet challenge was 224 pixels in 3-channels, so I select the top left 224 x 224 x 3 section of the cell images. I use the 'Adam' optimizer. I set [lr, beta1, beta2, epsilon] = [0.1, 0.8, 0.999, 1]. I attempt to predict 3 classes, and train to 8 epochs. The results of the training session are graphed below.

<img src="./readmePics/15-12model12epochs8.png" alt=".." width="350"/> <img src="./readmePics/15-12model12weights.png" alt=".." width="350"/>

Note that the green line indicates the portion of class 1 which is truely a 1, and the yellow line gives the fraction of time the net predicted a 1 for class 1. Also, note that there are a handful of large weights. This is undesirable, though it is likely the case that the original model had these weights before training happened

This model predicts all 0's after training. Before training, it predicts the same value for every picture, though it predicts different values for the 0, 1, 2 classes.

## Model 13 - Resnet18, without pretrained weights (training session 17-7-25/)

### Train

This is the same model as 12, except with random initialized weights.

The results for the training session are shown below.

<img src="./readmePics/17-7-25/training_session.png" alt=".." width="350"/> <img src="./readmePics/17-7-25/weight_distribution.png" alt=".." width="350"/>

Note that the weight distribution has improved, but it still has some run-away weights. The vast majority of the 11 million are small. Compare this to the weight distribution of the randomly initialized network:

<figure>
	<img src="./readmePics/17-7-25/weight_distribution_init.png" alt=".." width="" 	description="this is a thing"/>
	<figcaption>Figure 13-1: Weight distribution for randomly initialized 	network<figcaption/>
</figure>

The validation loss curve is erratic for this model, and the training loss fails to improve substantially. Additionally, the weights are not uniformly distributed before training. 

## Model 14 - Augmented InceptionResNetV2, with pretrained weights (training session 18-16-41/)

I load most of this model from the Keras.applications library. It is the InceptionResnetModel, with weights loaded from the 'imagenet' competition. The rest is an augmentation written by Vitoly Byranchonok (https://www.kaggle.com/byrachonok/pretrained-inceptionresnetv2-base-classifier).

### Architecture

	Layer (type)                 Output Shape              Param    
	input_2 (InputLayer)         (None, 299, 299, 3)       0         
	_________________________________________________________________
	batch_normalization_204 (Bat (None, 299, 299, 3)       12        
	_________________________________________________________________
	inception_resnet_v2 (Model)  (None, 8, 8, 1536)        54336736  
	_________________________________________________________________
	conv2d_204 (Conv2D)          (None, 8, 8, 128)         196736    
	_________________________________________________________________
	flatten_1 (Flatten)          (None, 8192)              0         
	_________________________________________________________________
	dropout_1 (Dropout)          (None, 8192)              0         
	_________________________________________________________________
	dense_1 (Dense)              (None, 512)               4194816   
	_________________________________________________________________
	dropout_2 (Dropout)          (None, 512)               0         
	_________________________________________________________________
	dense_2 (Dense)              (None, 28)                14364     
	================================================================= 
	Total params: 58,742,664
	Trainable params: 58,682,108
	Non-trainable params: 60,556

### Training

The entire model was trained on 1000 images for 8 epochs with the Adam optimizer, using default training parameters and epsilpn = .001, giving the following loss curve and weight distribution:

<img src="./readmePics/17-17-22/training_session.png" alt=".." width="350"/> <img src="./readmePics/17-17-22/weight_distribution.png" alt=".." width="350"/>

The same model was trained again, this time on all the images, over 15 epochs, and with the InceptionResNet weights frozen. The model looks like this: 

	Layer (type)                 Output Shape              Param    
	input_2 (InputLayer)         (None, 299, 299, 3)       0         
	_________________________________________________________________
	batch_normalization_204 (Bat (None, 299, 299, 3)       12        
	_________________________________________________________________
	inception_resnet_v2 (Model)  (None, 8, 8, 1536)        54336736  
	_________________________________________________________________
	conv2d_204 (Conv2D)          (None, 8, 8, 128)         196736    
	_________________________________________________________________
	flatten_1 (Flatten)          (None, 8192)              0         
	_________________________________________________________________
	dropout_1 (Dropout)          (None, 8192)              0         
	_________________________________________________________________
	dense_1 (Dense)              (None, 512)               4194816   
	_________________________________________________________________
	dropout_2 (Dropout)          (None, 512)               0         
	_________________________________________________________________
	dense_2 (Dense)              (None, 28)                14364     
	=================================================================
	Total params: 58,742,664
	Trainable params: 4,405,916
	Non-trainable params: 54,336,748
	_________________________________________________________________


Note that there are far fewer trainable parameters, which allowed me to train this model on the entire dataset. This generated the following loss curve:

<img src="./readmePics/18-16-41/training_session.png" alt=".." width="350"/>

Unfortunately, this model was corrupted as it was saved. It remains to be seen if a model with frozen weights can be saved. Also, I should check and make sure that I can load the model before I turn the aws machine off. Also, this may be because I severed the connection to the aws machine.

### Conclusion
This model has successful for other people in the past. It must be that my data generation scheme is preventing a successfull fit. 

# Model 14 again (31-9-1/)
This time I use a different preprocessing stragegy with the image. Previously, I was just grabbing a 300X300X3 subset of the image, and using that. I justified it by saying that the patterns that I am looking for are generally image-wide (though not always). This assumption is looking to be incorrect, so I use the entire image, first compressing using the cv2 library, and I also flip the images over both horizontal and vertical axes, and rotate the image as well, following Vitoly Byranchonok (https://www.kaggle.com/byrachonok/pretrained-inceptionresnetv2-base-classifier)

### train
The training scheme is successful this time, producing the following loss curves and weight distribution for the model. 

<img src="./readmePics/31-9-1/training_session.png" alt=".." width="325"/> <img src="./readmePics/31-9-1/weight_distribution.png" alt=".." width="325"/>

Note that the validation loss is still improving at epoch 60, and that the weights are starting to become larger. Around 100 of the 54 million weights are starting to become very large. These facts suggest more training, as well as more agressive regularization of the weights in the suffix network. 

Each epoch of 100 batches of size 10 took about 660 seconds.

### Threshold

I used .2 as the threshold for all categories.

### Performance

This model achieves a raw performance score of .190 and 1752/2021 placement on the kaggle leaderboard. The f1 score on the validation data is ~.10. 

### Conclusion
I need to train more, but this already took 12 hours to do 60 epochs. I need a speedup. There are two options: 1) just enable multiprocessing in the fit_generator method, and possibly make the batch size bigger. 2) I could cache the images as I load them, increasing my usage of RAM. General wisdom is to only do one of these strategies at a time. Lets try both and see if I get any sort of speedup and the accompanying improvement in performance. 

# Model 14, caching images  (3-6-57/)

### train
I use a generator that caches images as they are loaded from the hard drive. (modified haltuf generator) I train over roughly 180,000 images, seeing some images more than others (this was an error). The training took 29 hours, for a training velocity of 1.72 images/sec. I don't actually see a speedup as a result of the cacheing strategy. Note: I also did not augment the data set here, as I did before.

<figure>
<img src="./readmePics/3-6-57/training_session.png" alt="train_ses" width="350"/><img src="./readmePics/3-6-57/weight_distribution.png" alt="weights" width="350"/>	<figcaption>Figure 99: InceptionResNet trained for 39 hours over 180,000 images<figcaption/>
</figure>

### Notes
I ran train_3.py on commit 0f5a56928677ef7f9, my optimizer was Adam(.0001), which means lr I think. All the others were default.

### Threshold
I used this as the threshold:     T = [0.407, 0.441, 0.161, 0.145, 0.299, 0.129, 0.25, 0.414, 0.01, 0.028, 0.021, 0.125,
         0.113, 0.387, 0.602, 0.001, 0.137, 0.199, 0.176, 0.25, 0.095, 0.29, 0.159, 0.255,
         0.231, 0.363, 0.117, 0]

### Performance

This model achieves a raw performance score of .258

### Conclusion
It is import to compare training session 31-9-1/ with training session 3-6-57/. They are the same model but with different generators. 3-6-57/ uses a modified Haltuf generator. 

* caching was not helpful in speeding training. 
* My scheme for recording loss means I can't directly compare loss values between the two sessions
* 31-9-1/ was trained on 60,000 images (training speed 1.6 im/sec), while 3-6-57 was trained on 180,000 images (training speed 1.7 im/sec).
* Was the superior performance due to more training, or better thresholding? 
* 31-9-1/ was able to do just as well with less training time when thresholded the same. This is evidence that augmentation was very helpful.
* Further training with 31-9-1/ (augmented generator) can improve the score. 

The conclusion: augmentation improves score considerably. Caching does not cause a speedup. Thresholding has a significant impact on score. 

# Model16 , predict first 14 labels (training session 7-3-59/), predict the last 14 labels (training session 7-4-58)
 
### train
During training session 7-3-59/, I load all images into ram and then train. The total training time is about 12 hours. For training session 7-4-58/, I read and write images from disk as they are required. The total training time is about 12 hours. The methods are equivalent.

<figure>
<img src="./readmePics/7-3-59/training_session.png" alt="train_ses" width="300"/><img src="./readmePics/7-4-58/training_session.png" alt="weights" width="300"/>	<figcaption>Figure 99: loss and validation curves for training models to fit only 14 classes at a time<figcaption/>
</figure>


### Threshold
I used this as the threshold:T = [0.407, 0.441, 0.161, 0.145, 0.299, 0.129, 0.25, 0.414, 0.01, 0.028, 0.021, 0.125, 0.113, 0.387, 0.602, 0.001, 0.137, 0.199, 0.176, 0.25, 0.095, 0.29, 0.159, 0.255, 0.231, 0.363, 0.117, 0]

### Performance

This model achieves a raw performance score of .2 (the first half gets .13 points and then the second half gets .07 points.)


# Model18, smaller network (9-9-23/)
I use a smaller model this time. I use the InceptionV3, it is roughly half the size of the V2 network. 

### train

<figure>
<img src="9-9-23-training_session.png" alt="train_ses" width="300"/> <img src="9-9-23-weights.png" alt="weights" width="300"/>	<figcaption>Figure 99: loss and validation curves, weights for model<figcaption/>
</figure>

<figure>
<img src="9-9-23-suffix_weights.png" alt="train_ses" width="300"/>	<figcaption>Figure 99: Weight distribution of the suffix portion of the model<figcaption/>
</figure>

### Notes
train.py run on commit

### Threshold
I used this as the threshold:     T = [0.407, 0.441, 0.161, 0.145, 0.299, 0.129, 0.25, 0.414, 0.01, 0.028, 0.021, 0.125,
         0.113, 0.387, 0.602, 0.001, 0.137, 0.199, 0.176, 0.25, 0.095, 0.29, 0.159, 0.255,
         0.231, 0.363, 0.117, 0]

### Performance

This model achieves a raw performance score of .12, substantially worse than the larger model


