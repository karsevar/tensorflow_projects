###Hands on machine learning with Scikit learning and tensorflow
###chapter 14 Recurrent neural networks exercises: 

##1.) Sequence to sequence: I believe that the best illustration of 
#a sequence to sequence recurrent neural network is the auto-encoder to 
#decoder problem (the english to french translation exercise) since both 
#the input and output values can both be conceptuallized as sequences.
#Another very good illustration of this is stock market prediction RNN
#models and weather prediction RNN models because they all take in 
#a sequence as a input and creates a sequence for the output. 

#Sequence to vector: Sentiment tests for movies and customer products 
#because these two use cases accept as inputs a sequence of written 
#comments of these two customer services and outputs, in place of 
#a sequence in response, a single vector illustrating the reviewers 
#perception of the service. 

#vector to sequence: In retrospect the best vector the sequence problem 
#that I can think of is again the decoder problem (with the english to french 
#translator exercise) because the model takes in an autoencoded value 
#of a particular word or groups of words (ranging from 1 to the total 
#number of words in the learned dictionary) and uses a softmax probability 
#model to assign a new word that matches the encoding the best. In this 
#case, an english word is encoded into a numeric vector (this is called 
#sequence to vector) then it is ultimately decoded from a numeric vector 
#to a french phrase (this is called vector to sequence). 

##2.) Through the encoder to decoder methodology translation models have 
#increased in accuracy to efficiency because simply the creation of a 
#truthful (or rather accurate) translation is affected by grammar as well 
#as word choice. And so through a sequence to sequence model translation 
#word choice would probably be accurate the grammar between the words 
#will most likely be inaccurate at best and pigeoned at worst. Thus through 
#the encoder and decoder method the algorithm is forced to look through the 
#entire sentence (sequence) before translating it into the target language.
#While the sequence to sequence method will effectively translate the 
#sentence from a word to word prespective. 

##3.) How to combine a convolutional neural network with am RNN to classify 
#videos? 

#The best implementation that I can think of with these two different methods 
#is to separate the recurrent neurons and the convolutional neurons into 
#two separate parts of the graph and aggregate the results through a 
#regular softmax logist layer. The convolutional neurons will be the 
#first layers in the model, the recurrent neurons will be the middle layer, 
#and the softmax logistic layer will (or course) by the last. The problem 
#with this method is that most likely the author wants this hypothetical 
#model to pick up on artifacts that illustrate the video belongs in a 
#specific category and save these artifacts within a system of 
#recurrent nuerons. For this hypothetical model to be created one will 
#have to position each recurrent neuron inbetween each convolutional layer 
#(much like the early implementation of pooling layers). Such a architecture 
#might affect the back propagation through time method (I will really need to 
#look into this). 

##author's solution: to classify videos based on the visual content, one possible
#architecture could be to take say one frame per second, then run each frame through a 
#convolutional neural network, feed the output of the CNN to a sequence-to-vector RNN, 
#and finally run its output through a softmax layer, giving you all the class 
#probabilities. For training you would just use cross entropy as the cost function. 

##4.) I believe that the author said that dynamic_rnn() is better than static_rnn() 
#because it accepts a single tensor for all inputs at every time step and it 
#outputs a single tensor for all outputs at every time step as well as the function 
#contains the argument swap_memory which if set to true allows the practitioner to 
#swap the GPU's memory to the CPU's memory during backpropagation to 
#avoid out of memory errors. 

#As for the static_rnn() function a person will need to create individual 
#placeholder tensors for each time step (which in the case of over fifty 
#time steps will equate into 50 different X placeholders to name, code and assign to 
#the contrib.rnn.static_rnn() function). In addition, static_rnn() is memory 
#inefficient during backpropagation, meaning that OOM errors can occur during the 
#BPTT process. 

##5.) How can you deal with variable-length input sequences? 
#For variable length input sequences the tf.nn.dynamic_rnn() function 
#has a sequence_length argument that accepts 1D vectors as inputs. Through 
#this argument you can specify the number of characters, lists, or vectors that 
#the target input contains. 

##What about variable-length output sequences? 
#If you know in advance what length each sequence will have (for example if you 
#know that it will be the same length as the input sequence), then you can set the 
#sequence length parameter and if the output length is unknown you will have to define 
#a special output called an end of sequence token. Look at page 411 
#for a real world implementation of the latter method. 

##6.) Author's solution: To distribute training and execution of a deep 
#RNN across multiple GPUs, a common technique is simply to place each layer 
#on a different GPU. 

