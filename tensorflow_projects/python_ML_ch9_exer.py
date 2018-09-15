###Hands on machine learing with scikit learn and tensorflow:
###Chapter 9 exercises:
##1.) The main benefits that I can think of are that through the computational map 
#api one can create a problem without having to worry about syntax within the 
#python programming environment. Thus lending to one to be a little more creative 
#in the creation of machine learning algorithms. The draw backs to this method is 
#that the construction phase doesn't tell the programmer what computations will 
#work within python and because of that one can become lost in the process of 
#trouble shooting their tensorflow model code. 

#Putting coding comfort to the side, the computational benefits to using tensorflow's
#computational graphs are that specific parts of the computations can be allocated to 
#different cpus or gpus (unlike directly executing the computations through the terminal) 
#and that the syntax conventions lends to programmer to translate mathematical equations 
#more comfortably within the python environment. 

##2.) 
import tensorflow as tf
import numpy as np 

b = tf.Variable(3, name = "b") 
z = tf.Variable(5, name = "z") 
a = b + z 

init = tf.global_variables_initializer() 

with tf.Session() as sess:
	init.run() 
	result = a.eval() 
print(result)
tf.reset_default_graph() 

b = tf.constant(3)
z = b + 4 
a = z - 4 

init = tf.global_variables_initializer() 

with tf.Session() as sess: 
	a_val = a.eval(session=sess) 

print(a_val) 

with tf.Session() as sess:
	a_val = sess.run(a) 

print(a_val) 
#Interesting they bost have the same results. According to the author these two computations 
#are indeed the same. Will need to look into this.

##3.) These computations use the same graph as question 2. 
init = tf.global_variables_initializer() 

with tf.Session() as sess: 
	init.run() 
	a_val = a.eval(session = sess)

print(a_val)
tf.reset_default_graph() 
#Now I understand, the operation a_val, b_val = a.eval(session = sess) only con evaluate 
#the value within a and, since b is a constant, tensorflow could implant the value within 
#b into b_val. 

z = tf.constant(3) 
b = z + 10 
a = b + 20 

init = tf.global_variables_initializer() 

with tf.Session() as sess:
	init.run() 
	a_val = a.eval(session = sess) 
	b_val = b.eval(session = sess) 

print(a_val) 
print(b_val)#Interestingly enough the tensorflow session can't process b_val without 
#calling the object b within an individual eval() function call. In the end though, 
#this modified command illustrates that a_val, b_val = a.eval(session = sess) is not 
#the same as:

init = tf.global_variables_initializer() 

with tf.Session() as sess:
	init.run() 
	a_val, b_val = sess.run([a, b])
print(a_val, b_val)
#Where a_val and b_val are all assessed in one computation (or rather eval function call).

##4.) You can run two graphs in the same python window session through the command 
#graph = tf.Graph() and with graph.as_default(): x2 = tf.Variable(2), but then this 
#extra graph will not show up in the same tensorflow session call when using the default_graph() 
#assembly. 

#So in other words, you can't run a tensorflow session with two different graphs but 
#you can have more than one computational graph populating the python window session. 

##According to the author: No, you cannot run two graphs in the same session. You would 
#have to merge the graphs into a single graph first. 

##5.) Author's answer: In local tensorflow, sessions manage variable values, so if you create
#a graph g containing a variable w, then start two threads and open a local session 
#in each thread, both using the same graph g, then each session will have its own 
#copy of the variable w. However, in distributed tensorflow, variable values are stored 
#in containers managed by the cluster, so if both sessions connect to the same cluster
#and use the same container, then they will share the same variable value for w. 

##6.) A variable is created through calling either the tf.global_variables_initializer() 
#function or individually calling sess = tf.Session() sess.run(object.initializer) for 
#each tensor you want to initialize. The variables are destroyed through calling the command 
#sess.close() or automatically if one uses the function tf.global_variables_initializer() with 
#the code block with tf.Session() as sess:

##7.) The main different I can see between tf.Variable and tf.placeholder is that 
#tf.placeholder() is (exactly what the name specifies) just a placeholder meaning that 
#it requires no initial input values during the functions construction and that values 
#are only inputted into the function during the evaluation phase through the command 
#sess.run(construction_equation, feed_dict={X: train_var, y: train_labels}). this functionality 
#allows you to carryout stochastic gradient descent and mini batch gradient descent computations.
#While the tf.Variable() function requires an initial value during the function's 
#construction and can't utilize the feed_dict argument within the sess.run() function 
#call. 

##author's answer (an important definition of tf.Variable) A variable is an operation 
#that holds a value. If you run the variable, it returns that value. Before you can run it,
#you need to initialize it. You can change the variable's value (for example, by using
#an assignment operation). It is stateful: the variable keeps the same value upon successive 
#runs of the graph. It is typically used to hold model parameters  but also for other 
#purposes. 

##8.) If you run a computation that has a tf.placeholder() function without specifying 
#the values through the feed_dict argument during the tf.Session() the console will 
#output an error and if the computational map does not require tf.placeholder() values 
#I believe an except will be raised and the computation will be ran.

##Author's answer: If you run the graph to evaluate an operation that depends on a placeholder
#but you don't feed its value, you get an exception. If the operation does not depend 
#on the placeholder, then no exception is raised.

##9.) Author's answer: When you run a graph, you can feed the output value of any operation, not 
#just the value of placeholders. In practice, however, this is rather rare. 

##10.) Author's answer: According to the author you can place different values into the 
#evaluation phase of a tensor graph through writing an tf.assign() function call and passing 
#it a new tf.placeholder() value within the evaluation phase.

#illustration of this: 
x = tf.Variable(tf.random_uniform(shape=(), minval = 0.0, maxval = 1.0))
x_new_val = tf.placeholder(shape=(), dtype = tf.float32)
x_assign = tf.assign(x, x_new_val) 

with tf.Session():
	x.initializer.run() 
	print(x.eval())
	x_assign.eval(feed_dict={x_new_val: 5.0})
	print(x.eval())

tf.reset_default_graph() 

##11.) Author's answer: Reverse-mode autodiff needs to traverse the graph only twice in 
#order to compute the gradients of the cost function with regards to any number of variables 
#On the other hand, forward-mode autodiff would need to run once for each variable (so 10 
#times if we want the gradients with regards to 10 different variables). As for symbolic 
#differentiation, it would build a different graph to compute the gradients, so it 
#would not traverse the original graph at all (except when building the new gradients 
#graph). 

##12.) 
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler 

moons = make_moons(n_samples = 1000, noise = 0.15, random_state = 42) 
moons_data = moons[0]

moons_target = moons[1] 
m, n = moons_data.shape
moons_data_bias = np.c_[np.ones((m, 1)), moons_data]

test_data = moons_data_bias[800:]
test_target = moons_target[800:]
train_data = moons_data_bias[:800]
train_target = moons_target[:800] 

n_epoches = 1000
learning_rate = 0.01

#X = tf.placeholder(tf.float32, shape = (None, n + 1), name = "X") 
#y = tf.placeholder(tf.float32, shape = (None, 1), name = "y")

batch_size = 50
n_batches = int(np.ceil(len(train_data) / batch_size)) 

def fetch_batch(epoch, batch_index, batch_size): 
	indices = np.random.randint(0, len(train_data), size = batch_size) 
	X_batch = train_data[indices] 
	y_batch = train_target.reshape(-1, 1)[indices] 
	return X_batch, y_batch

#Construction phase: Using Andrew Ng's logistical regression implementation:
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0), name = "theta")
logits = tf.matmul(X, theta, name = "logits")  
y_prob = 1 / (1 + tf.exp(-logits))
epsilon = 1e-7
loss = tf.losses.log_loss(y, y_prob) 
optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate) 
training_op = optimizer.minimize(loss)  

init = tf.global_variables_initializer() 
with tf.Session() as sess:
	sess.run(init)

	for epoch in range(n_epoches):
		for batch_index in range(n_batches):
			X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size) 
			sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
			loss_val = loss.eval() 
		if epoch % 100 == 0:
			print("Epoch:", epoch, "loss: ", loss_val)
	best_theta = theta.eval()  
print(best_theta) 
 

##blog implementation using numpy:
def sigmoid(z):
	return 1.0 / (1 + np.exp(-z))

def predict(features, weights):
	z = np.dot(features, weights) 
	return sigmoid(z) 

#this author uses cross entropy for the cost function of the equation.
#Again will need to relisten to Andrew Ng. 

def cost_function(features, labels, weights):
	observations = len(labels) 
	predictions = predict(features, weights) 

	#Take the error when label = 1 
	class1_cost = -labels*np.log(predictions) 

	#Take the error when label = 0
	class2_cost = (1-labels) * np.log(1-predictions) 

	#Take the sum of both costs 
	cost = class1_cost - class2_cost 

	#Take the average cost 
	cost = cost.sum()/observations

	return cost 

def update_weights(features, labels, weights, lr): 
	N = len(features) 

	predictions = predict(features, weights) 
	gradient = np.dot(features.T, predictions - labels) 
	gradient /= N  
	gradient *= lr 
	weights -= gradient 
	return weights 

def decision_boundary(prob):
	return 1 if prob >= 0.5 else 0 

def classify(preds):
	decision_bounrdary = np.vectorize(decision_boundary)
	return decision_boundary(predictions).flatten() 

def train(features, labels, weights, lr, iters):
	cost_history = [] 

	for i in range(iters):
		weights = update_weights(features, labels, weights, lr) 

		#Calculate error for auditing purposes:
		cost = cost_function(features, labels, weights)
		cost_history.append(cost) 

		#log process:
		if i % 100 == 0:
			print ("iter :" + str(i) + " cost: " + str(cost)) 

	return weights, cost_history

train(moons_data_bias, moons_target, np.random.uniform(-1.0, 1.0, size = [n + 1, 1]) , 0.01, 10000)



