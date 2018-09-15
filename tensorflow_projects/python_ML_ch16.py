###Hands on machine learning with Scikit learning and tensorflow
###chapter 16 Reinforcement learning
##Learning to optimize rewards:
#In reinforcement learning, a software agent makes observations and takes 
#actions within an environment, and in return it recieves rewards. It's objective 
#is to learn to act in a way that will maximize its expected long term rewards. 

##Policy search:
#The algorithm used by the software agent to determine its actions is called 
#its policy. The policy can be any algorithm you can think of, and it does not even 
#have to be deterministic. 

#Methods to find which policy works with the agent are:
#policy search look at page 444 for a full discription of this.

#genetic algorithms which is randomly creating a first generation of 
#x number of policies and tring them out, then killing n number of policies 
#that aren't producing good enough results for the task at hand. the surviving 
#group will be kept and forced to create offsprings. An offspring is just a
#copy of its parent plus some random variation. The surviving policies 
#plus their offspring together constitute the second generation.

#optimization techniques that uses gradient ascent to find the policy 
#that creates the best rewards.

##Introduction to openAI gym:
#OpenAI gym is a toolkit that provides a wide variety of simulated environments 
#so you can train agents, compare them, or develop new RL algorithms.

import PIL
import gym
import matplotlib.pyplot as plt 
import matplotlib.animation as animation 
import matplotlib 
plt.rcParams["axes.labelsize"] = 14
plt.rcParams["xtick.labelsize"] = 12 
plt.rcParams["ytick.labelsize"] = 12 

env = gym.make("CartPole-v0")  
obs = env.reset()
for _ in range(1000):
	img = env.render(mode="rgb_array")
	env.step(env.action_space.sample())
	env.close() 
	#So this is the way you're supposed to run openai gym on your system 
	#interesting. 

plt.figure(figsize=(5,4))
plt.imshow(img)
plt.axis("off") 
plt.show()#Interesting so the pyplot module is needed for this exercise 
#to work correctly.    
 

#the make() function creates an environment, in this case a Cartpole environment 
#This is a 2d simulation in which a cart can be accelerated left or right in order to balance 
#the pole placed on top of it. After the environment is created, we must initialize
#it using the reset() method. This returns the first observation. Observations 
#depend on the type of environment. Finally the render() method displays the 
#environment. 

#If you want render() to return the rendered image as a Numpy array, you can set the 
#mode parameter to rgb_array 

print(env.action_space)#This command tells you the number of actions that 
#are possible in the environment. In this case there are 2 possible actions 
#0 for left and 1 for right. 

action = 1#right 
obs, reward, done, info = env.step(action) 
print(obs) 
print(reward) 
print(done) 

#The step() method executes the given action and returns four values:

#obs: this is the new observation, The cart is now moving toward the right obs[1]>0 
#The pole is still tilted toward the right (obs[2] > 0), but its angular velocity is now negative 
#(obs[3]<0), so it will likely be tilted toward the left after the next step. 

#reward: In this environment you get a reward 1 at every step, no matter what you do, so the 
#goal is to keep running as long as possible. 

#done: this value will be True when the episode is over. This will happen 
#when the pole tilts too much. After that, the environment must be reset before it can be 
#used again. 

#Let's hardcode a simple policy that accelerates left when the pole is learning 
#toward the left and accelerates right when the pole is learning toward the right.
#this policy will be ran over 500 episodes.

def basic_policy(obs):
	angle = obs[2]
	return 0 if angle < 0 else 1 

totals = [] 
for episode in range(500):
	episode_rewards = 0 
	obs = env.reset() 
	for step in range(1000):
		action = basic_policy(obs) 
		obs, reward, done, info = env.step(action) 
		episode_rewards +=reward
		if done:
			break
	totals.append(episode_rewards) 

import numpy as np 
print(np.mean(totals), np.std(totals), np.min(totals), np.max(totals))
#No way this actually worked!!! 
#This code states that the pole was only kept upright for 68 consecutive 
#steps. 

##Neural network policies:
#This neural network will take an observation as input, and it will output the action to 
#be executed. More precisely, it will estimate a probability for each action, and then 
#we will select an action randomly according to the estimated probabilities. 
#For more information on this implementation look at page 449. 

import tensorflow as tf
from PIL import Image, ImageDraw 

try:
	from plglet.gl import gl_info 
	openai_cart_pole_rendering = True 
except Exception:
	openai_cart_pole_rendering = False 

#openai customized render function from the author's github:
def update_scene(num, frames, patch):
	patch.set_data(frames[num]) 
	return patch, 

def plot_animation(frames, repeat=False, interval=40):
	plt.close() 
	fig = plt.figure() 
	patch = plt.imshow(frames[0]) 
	plt.axis("off") 
	return animation.FuncAnimation(fig, update_scene, fargs=(frames, patch), frames=len(frames),
		repeat=repeat, interval=interval) 

def render_cart_pole(env, obs):
	if openai_cart_pole_rendering:
		return env.render(model="rgb_array") 
	else:
		img_w = 600 
		img_h = 400 
		cart_w = img_w // 12 
		cart_h = img_h //15 
		pole_len = img_h //3.5 
		pole_w = img_w // 80 + 1 
		x_width = 2
		max_ang = 0.2 
		bg_col = (255, 255, 255) 
		cart_col = 0x000000#Blue green red 
		pole_col = 0x669acc #blue green red 

		pos, val, ang, ang_el = obs 
		img = Image.new("RGB", (img_w, img_h), bg_col) 
		draw = ImageDraw.Draw(img) 
		cart_x = pos * img_w // x_width + img_w // x_width 
		cart_y = img_h * 95 //100 
		top_pole_x = cart_x + pole_len * np.sin(ang)  
		top_pole_y = cart_y - cart_h // 2 - pole_len * np.cos(ang)
		draw.line((0, cart_y, img_w, cart_y), fill=0) 
		draw.rectangle((cart_x - cart_w // 2, cart_y - cart_h // 2, cart_x + cart_w // 2, 
			cart_y + cart_h // 2), fill=cart_col)
		draw.line((cart_x, cart_y - cart_h // 2, top_pole_x, top_pole_y ), fill = pole_col, 
			width = pole_w) 
		return np.array(img) 

def plot_cart_pole(env, obs):
	plt.close() 
	img = render_cart_pole(env, obs) 
	plt.imshow(img) 
	plt.axis("off") 
	plt.show() 

#specify the neural network architecture:
n_inputs = 4# == env.observation_space.shape[0]
n_hidden = 4
n_outputs = 1 #Since this is only outputing the probability of 
#picking left as a means to keep the pole stable. 
initializer = tf.contrib.layers.variance_scaling_initializer() #he_initializer 

#build the neural network:
X = tf.placeholder(tf.float32, shape=[None, n_inputs])
hidden = tf.layers.dense(X, n_hidden, activation = tf.nn.elu,
						kernel_initializer=initializer) 
logits = tf.layers.dense(hidden, n_outputs, kernel_initializer=initializer) 
outputs = tf.nn.sigmoid(logits) 

#Select a random action based on the estimated probabilities:
p_left_and_right = tf.concat(axis = 1, values=[outputs, 1-outputs]) 
action = tf.multinomial(tf.log(p_left_and_right), num_samples=1) 

init = tf.global_variables_initializer() 

frames = []
totals = [] 

with tf.Session() as sess:
	init.run() 
	obs = env.reset()
	for step in range(1000):
		img = render_cart_pole(env, obs) 
		frames.append(img)
		action_val = action.eval(feed_dict={X:obs.reshape(1, n_inputs)})   
		obs, reward, done, info = env.step(action_val[0][0]) 
		episode_rewards +=reward
		if done:
			break
	totals.append(episode_rewards)

env.close() 
video = plot_animation(frames)
plt.show() 
 
print(np.mean(totals), np.std(totals), np.min(totals), np.max(totals))
#this model only inscreased the number of consecutive pole stable events 
#to 70 (which is only a 4 episode inscrease)

##Evaluating Actions: The credit Assignment Problem: 
#the credit assignment problem. How does the policy know immediately 
#that an action was good or bad? 

#to tackle this problem, a common strategy is to evaluate an action based on the sum rewards 
#that come after it, usually applying a discount rate r at each step. 
#To see the full definition of this methodology look at page 451. 

##Policy Gradients: 
#As discussed earlier, PG algorithms optimize the parameters of a policy 
#by following the gradients toward higher rewards. 

#The steps in creating this algorithm:

#1.) First, let the neural network policy play the game several times and at 
#each step compute the gradients that would make the chosen action even more likely 
#but don't apply these gradients yet. 

#2.) Once you have run several episodes , compute each action's score (using
#the credit assignment method).

#3.) If an action's score is positive, it means that the action was good and you 
#want to apply the gradients computed earlier to make the action even more 
#likely to be chosen in the future. However, it the score is negative, it means the 
#action was bad and you want to apply the opposite gradients to make this action 
#less likely in the future. The solution is simply to multiply each gradient 
#vector by the corresponding action's score. 

#4.) Finally, compute the mean of all the resulting gradient vectors, and use it to perform 
#a gradient descent step. 

#This model will use the same construction components as the last neural 
#network except the inclusion of a chosen probability for an action being 
#1 and 0 for left and right respectively, a sigmoid_cross_entropy_with_logits() 
#function step, an optimizer step, and a compute_gradients() method step (since 
#it returns a list of gradient vector/variable pairs (one pair per trainable variable)). 

tf.reset_default_graph()

n_inputs = 4
n_hidden = 4
n_outputs = 1 
initializer = tf.contrib.layers.variance_scaling_initializer() 

learning_rate = 0.01 

X = tf.placeholder(tf.float32, shape=[None, n_inputs]) 
hidden = tf.layers.dense(X, n_hidden, activation=tf.nn.elu,
						kernel_initializer=initializer)
logits = tf.layers.dense(hidden, n_outputs, kernel_initializer=initializer) 
outputs = tf.nn.sigmoid(logits) 
p_left_and_right = tf.concat(axis=1, values=[outputs, 1 - outputs]) 
action = tf.multinomial(tf.log(p_left_and_right), num_samples = 1) 

y = 1. - tf.to_float(action) 
cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
					labels=y, logits=logits) 
optimizer = tf.train.AdamOptimizer(learning_rate) 
grads_and_vars = optimizer.compute_gradients(cross_entropy)
gradients = [grad for grad, variable in grads_and_vars] 
gradient_placeholders = [] 
grads_and_vars_feed = [] 
for grad, variable in grads_and_vars:
	gradient_placeholder = tf.placeholder(tf.float32, shape = grad.get_shape())
	gradient_placeholders.append(gradient_placeholder) 
	grads_and_vars_feed.append((gradient_placeholder, variable))
training_op = optimizer.apply_gradients(grads_and_vars_feed) 

init = tf.global_variables_initializer() 
saver = tf.train.Saver() 

#functions to compute the discounted rewards for the execution phase:
def discount_rewards(rewards, discount_rate):
	discounted_rewards = np.empty(len(rewards)) 
	cumulative_rewards = 0 
	for step in reversed(range(len(rewards))):
		cumulative_rewards = rewards[step] + cumulative_rewards * discount_rate 
		discounted_rewards[step] = cumulative_rewards 
	return discounted_rewards 

def discount_and_normalize_rewards(all_rewards, discount_rate):
	all_discounted_rewards = [discount_rewards(rewards, discount_rate)
												for rewards in all_rewards]
	flat_rewards = np.concatenate(all_discounted_rewards)
	reward_mean = flat_rewards.mean() 
	reward_std = flat_rewards.std() 
	return [(discounted_rewards - reward_mean)/reward_std
			for discounted_rewards in all_discounted_rewards]

#Training the policy:
n_iterations = 250 
n_max_steps = 1000 
n_games_per_update = 10 #train the policy every 10 episodes 
save_iterations = 10 
discount_rate = 0.95

with tf.Session() as sess:
	init.run() 
	for iteration in range(n_iterations):
		all_rewards = []
		all_gradients = [] 
		for game in range(n_games_per_update):
			current_rewards = [] 
			current_gradients =[] 
			obs = env.reset() 
			for step in range(n_max_steps):
				action_val, gradients_val = sess.run(
					[action, gradients], 
					feed_dict={X: obs.reshape(1, n_inputs)})
				obs, reward, done, info = env.step(action_val[0][0])
				current_rewards.append(reward) 
				current_gradients.append(gradients_val) 
				if done:
					break
			all_rewards.append(current_rewards) 
			all_gradients.append(current_gradients) 

		#At this point we have run the policy for 10 episodes, and 
		#we are ready for a policy update using the algorithm described 
		#earlier.
		all_rewards = discount_and_normalize_rewards(all_rewards, discount_rate=discount_rate)
		feed_dict = {}
		for var_index, gradient_placeholder in enumerate(gradient_placeholders):
			mean_gradients = np.mean([reward * all_gradients[game_index][step][var_index]
									for game_index, rewards in enumerate(all_rewards)
										for step, reward in enumerate(rewards)], axis=0)
			feed_dict[gradient_placeholder] = mean_gradients 
		sess.run(training_op, feed_dict=feed_dict) 
		if iteration % save_iterations == 0:
			saver.save(sess, "./my_policy_net_pg.ckpt") 
env.close() 

def render_policy_net(model_path, action, X, n_max_steps = 1000):
	frames = [] 
	env = gym.make("CartPole-v0") 
	obs = env.reset() 
	with tf.Session() as sess:
		saver.restore(sess, model_path) 
		for step in range(n_max_steps):
			img = render_cart_pole(env,obs) 
			frames.append(img) 
			action_val = action.eval(feed_dict={X: obs.reshape(1, n_inputs)})
			obs, reward, done, info = env.step(action_val[0][0]) 
			if done:
				break 
	env.close() 
	return frames 

frames = render_policy_net("./my_policy_net_pg.ckpt", action, X, n_max_steps = 1000)
video = plot_animation(frames) 
plt.show() 

##Markov Decision Processes:
#these concepts are a little too advanced for me right now but from what 
#my meager understanding of advanced mathematics can tell me is that markov
#decision processes were first a thought experiment used to find a optimal 
#state within a random environment. 
#And over time other mathematicians found that the optimal set decisions can be 
#solved mathematically through the Bellman optimality equation. this equation leads directly 
#to an algorithm that can precisely estimate the optimal state value of every 
#possible state: you first initialize all the state value estimates to zero, 
#then you iteratively update them using the Value iteration algorithm. A remarkable 
#result is that, given enough time, these estimates are guaranteed to converge to the 
#optimal state values, corresponding to the optimal policy. 

#A counter part to this algorithm is the Q-values algorithm which estimates the 
#the action values. 

#Q-value algorithm applied to the MDP representation.
nan = np.nan

T = np.array([ #shape= [s, a, s']
	[[0.7, 0.3, 0.0], [1.0, 0.0, 0.0], [0.8, 0.2, 0.0]],
	[[0.0, 1.0, 0.0], [nan, nan, nan], [0.0, 0.0, 1.0]],
	[[nan, nan, nan], [0.8, 0.1, 0.1], [nan, nan, nan]],
])

R = np.array([ #shape=[s, a, s']
	[[10., 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
	[[10., 0.0, 0.0], [nan, nan, nan], [0.0, 0.0, -50.]],
	[[nan, nan, nan], [40., 0.0, 0.0], [nan, nan, nan]],
]) 
possible_actions = [[0, 1, 2], [0,2], [1]]

Q = np.full((3,3), -np.inf) 
for state, actions in enumerate(possible_actions):
	Q[state, actions] = 0.0 #Initial value = 0.0 for all possible actions 

learning_rate = 0.01 
discount_rate = 0.95
n_iterations = 100 

for iteration in range(n_iterations):
	Q_prez = Q.copy() 
	for s in range(3):
		for a in possible_actions[s]:
			Q[s,a] = np.sum([
				T[s, a, sp] * (R[s, a, sp] + discount_rate * np.max(Q_prez[sp]))
				for sp in range(3) 
			])

print(Q) 
print(np.argmax(Q, axis = 1)) #the optimal action for each state 

##Temporal Difference Learning and Q-learning:
#Reinforcement learning problems with discrete actions can often be modeled as 
#Markov decision processes, but the agent initially has no idea what the transition 
#probabilities are, and it does not know what the rewards are going to be either.
#It must experience each state and each transition at least once to know the 
#rewards, and it must experience them multiple times if it is to have a reasonable 
#estimate of the transition probabilities.

#The Temporal difference learning algorithm is very similar to the value 
#iteration algorithm, but tweaked to take into account the fact that the agent 
#has only partial knowledge of the MDP. In general we assume that the agent
#initially knows only the possible states and actions, and nothing more. the agent 
#uses an exploration policy -- for example, a purely random policy --to explore the MDP
#and as it progresses the TD learning algorithm updates the estimates of the state values 
#based on the transitions and rewards that are actually observed. 

import numpy.random as rnd 

learning_rate0 = 0.05
learning_rate_decay = 0.1 
n_iterations = 20000 

s = 0 
Q = np.full((3,3), -np.inf) 
for state, actions in enumerate(possible_actions):
	Q[state, actions] = 0.0 

for iteration in range(n_iterations):
	a = rnd.choice(possible_actions[s])
	sp = rnd.choice(range(3), p=T[s, a]) 
	reward = R[s, a, sp]
	learning_rate = learning_rate0 / (1 + iteration * learning_rate_decay) 
	Q[s, a] =learning_rate * Q[s, a] + (1 - learning_rate) * (reward + discount_rate * np.max(Q[sp])
		)
	s = sp 

##A variation to the Q-learning algorithm is the Q-learning algorithm with 
#an exploration policy bonus. this speeds up the convergence time and 
#allows for the algorithm to only explore regions of the environment that 
#are relatively interesting to the algorithm (regions that give rise to 
#more rewards). 












