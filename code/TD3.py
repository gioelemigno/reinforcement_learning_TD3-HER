import numpy as np
import tensorflow as tf
import keras
import os

'''
############################################ ACTOR ###############################################	
'''
class Actor_network(tf.keras.Model):
    def __init__(self, state_dim, hidden_units, action_dim, max_value_action):
        super(Actor_network, self).__init__()

        self.max_value_action = max_value_action #save max value that will be used in forward step

        #input units
        self.input_layer = tf.keras.layers.InputLayer(input_shape=(state_dim, ))
        
        #hidden layers
        self.hidden_layers = []
        for units in hidden_units:
            self.hidden_layers.append(tf.keras.layers.Dense(units, activation='relu', kernel_initializer='RandomNormal'))

        #output unit
        self.output_layer = tf.keras.layers.Dense(action_dim, activation='tanh', kernel_initializer='RandomNormal')

    @tf.function
    def call(self, state):  #forward step
        z = self.input_layer(state)
        for layer in self.hidden_layers:
            z = layer(z)
        output = self.output_layer(z)
        return output * self.max_value_action   #output is within the action values range

'''
############################################ CRITIC ###############################################	
'''
class Critic_network(tf.keras.Model):
    def __init__(self, state_dim, action_dim, hidden_units):
        super(Critic_network, self).__init__()

        #input units
        self.input_layer = tf.keras.layers.InputLayer(input_shape=(state_dim+action_dim, ))
        
        #hidden layers
        self.hidden_layers = []
        for units in hidden_units:
            self.hidden_layers.append(tf.keras.layers.Dense(units, activation='relu', kernel_initializer='RandomNormal'))

        #output unit
        self.output_layer = tf.keras.layers.Dense(1, activation='linear', kernel_initializer='RandomNormal')

    @tf.function
    def call(self, state, action):  #forward step
        inputs = tf.concat([state, action], 1)
        z = self.input_layer(inputs)
        for layer in self.hidden_layers:
            z = layer(z)
        output = self.output_layer(z)
        return output 

'''
############################################ init weights ###############################################	
'''
#These function are used to force the program to allocate the weights of the network 
# They are used before the copy of the weights to the target network
def init_weights_actor(keras_model, input_dim):
  init_weight = np.array(range(1,input_dim+1))
  init = np.atleast_2d(init_weight.astype('float32'))
  keras_model(init) #force to allocate weights

def init_weights_critic(keras_model, input_dim_1, input_dim_2):
  init_weight_1 = np.array(range(1,input_dim_1+1))
  init_1 = np.atleast_2d(init_weight_1.astype('float32'))

  init_weight_2 = np.array(range(1,input_dim_2+1))
  init_2 = np.atleast_2d(init_weight_2.astype('float32'))

  keras_model(init_1, init_2) #force to allocate weights

'''
############################################ update weights target ###############################################	
'''

def update_target_network(tau, network, target):
    new_weights = network.get_weights()
    i=0
    for param, target_param in zip(network.get_weights(), target.get_weights()):
        new_weights[i] = tau*param + (1-tau)*target_param 
        i = i+1
    target.set_weights(new_weights)

'''
############################################ TD3 ###############################################	
'''

class TD3(object):
	def __init__(self,
		state_dim, 
		action_dim,
		max_value_action,
		discount=0.99,
		tau=0.005,
		policy_noise=0.2,
		noise_clip=0.5,
		policy_freq=2,
		hidden_layers_actor=[256,256],
		hidden_layers_critic=[256,256],
		lr_adam=3e-4):

		## ACTORS
		#init actor network
		self.actor = Actor_network(state_dim, hidden_layers_actor, action_dim, max_value_action)
		init_weights_actor(self.actor, state_dim)
		self.actor_optimizer = tf.optimizers.Adam(lr_adam)

		#build and init actor_target with the same weights of the actor network
		self.actor_target = Actor_network(state_dim, hidden_layers_actor, action_dim, max_value_action)
		init_weights_actor(self.actor_target, state_dim)
		self.actor_target.set_weights(self.actor.get_weights())


		## CRITICS
		#init critic networks
		self.critic_Q1 = Critic_network(state_dim, action_dim, hidden_layers_critic)
		init_weights_critic(self.critic_Q1, state_dim, action_dim)
		self.critic_Q1_optimizer = tf.optimizers.Adam(lr_adam)

		self.critic_Q2 = Critic_network(state_dim, action_dim, hidden_layers_critic)
		init_weights_critic(self.critic_Q2, state_dim, action_dim)
		self.critic_Q2_optimizer = tf.optimizers.Adam(lr_adam)


		#init target critics 
		self.critic_Q1_target = Critic_network(state_dim, action_dim, hidden_layers_actor)
		init_weights_critic(self.critic_Q1_target, state_dim, action_dim)
		self.critic_Q1_target.set_weights(self.critic_Q1.get_weights())

		self.critic_Q2_target = Critic_network(state_dim, action_dim, hidden_layers_actor)
		init_weights_critic(self.critic_Q2_target, state_dim, action_dim)
		self.critic_Q2_target.set_weights(self.critic_Q2.get_weights())

		self.max_value_action = max_value_action
		self.discount = discount
		self.tau = tau
		self.policy_noise = policy_noise
		self.noise_clip = noise_clip
		self.policy_freq = policy_freq
		self.action_dim = action_dim

		self.total_it = 0   

	def pi(self, state):
		state = np.atleast_2d(state.astype('float32'))
		return self.actor(state)


	def train(self, replay_buffer, batch_size=100, func_preprocess_state=None):
		self.total_it += 1

		# Sample replay buffer 
		state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
		
		# Used in HER to normalize the observation 
		if func_preprocess_state != None:
			state_prep = []
			next_state_prep = []
			for idx in range(len(state)):	
				state_prep.append(func_preprocess_state(state[idx]))
				next_state_prep.append(func_preprocess_state(next_state[idx]))
			state = np.array(state_prep)
			next_state = np.array(next_state_prep)

		# policy noise 
		noise =  np.random.normal(0, self.policy_noise, size=self.action_dim) 
		noise.clip(-self.noise_clip, self.noise_clip)

		# chose new action using actor target
		next_state = tf.constant(next_state, dtype=tf.float32) #cast from float64 to float32
		next_action = self.actor_target(next_state) + noise 
		next_action = tf.clip_by_value(next_action, clip_value_min=-self.max_value_action, clip_value_max=self.max_value_action)

		# compute Q function of the pair (next_state, next_action) using the target critics
		target_Q1 = self.critic_Q1_target(next_state, next_action)
		target_Q2 = self.critic_Q2_target(next_state, next_action)

		#chose the minimum (minimum element wise operation)
		min_target_Q = tf.math.minimum(target_Q1, target_Q2)

		# compute the target 
		target_Q = reward + not_done * self.discount * min_target_Q

		state = tf.constant(state, dtype=tf.float32)
		action = tf.constant(action, dtype=tf.float32)

		# Training critic Q1
		with tf.GradientTape() as tape:
			current_Q1 = self.critic_Q1(state, action)	# current estimation of the pair (state, action) 
			target = tf.stop_gradient(target_Q)	# target value of pair (state, action)
			loss = tf.keras.losses.mean_squared_error(target, current_Q1) # compute MSE #tf.mean_squared_error(y_true, y_pred)
		variables = self.critic_Q1.trainable_variables
		gradients = tape.gradient(loss, variables)
		self.critic_Q1_optimizer.apply_gradients(zip(gradients, variables))

		# Training critic Q2
		with tf.GradientTape() as tape:
			current_Q2 = self.critic_Q2(state, action)
			target = tf.stop_gradient(target_Q)
			loss = tf.keras.losses.mean_squared_error(target, current_Q2) #tf.mean_squared_error(y_true, y_pred)
		variables = self.critic_Q2.trainable_variables
		gradients = tape.gradient(loss, variables)
		self.critic_Q2_optimizer.apply_gradients(zip(gradients, variables))

		if self.total_it % self.policy_freq == 0:
			#training the actor
			with tf.GradientTape() as tape:
				actor_loss = -self.critic_Q1(state, self.actor(state))
				actor_loss = tf.math.reduce_mean(actor_loss)
			variables = self.actor.trainable_variables
			gradients = tape.gradient(actor_loss, variables)
			self.actor_optimizer.apply_gradients(zip(gradients, variables))

			#update target networks
			#Q1
			update_target_network(self.tau, self.critic_Q1, self.critic_Q1_target)

			#Q2
			update_target_network(self.tau, self.critic_Q2, self.critic_Q2_target)

			#actor
			update_target_network(self.tau, self.actor, self.actor_target)

	def save_weights_networks(self, root_folder_dst):
		root = root_folder_dst
		folder_Q1 = os.path.join(root, "critic_Q1")
		folder_Q2 = os.path.join(root, "critic_Q2")
		folder_Q1_target = os.path.join(root, "critic_Q1_target")
		folder_Q2_target = os.path.join(root, "critic_Q2_target")

		folder_actor = os.path.join(root, "actor")
		folder_actor_target = os.path.join(root, "actor_target")

		#os.makedirs(root)
		os.makedirs(folder_Q1)
		os.makedirs(folder_Q2)
		os.makedirs(folder_Q1_target)
		os.makedirs(folder_Q2_target)
		os.makedirs(folder_actor)
		os.makedirs(folder_actor_target)

		Q1 = os.path.join(folder_Q1, "critic_Q1")
		Q2 = os.path.join(folder_Q2, "critic_Q2")
		Q1_target = os.path.join(folder_Q1_target, "critic_Q1_target")
		Q2_target = os.path.join(folder_Q2_target, "critic_Q2_target")

		actor = os.path.join(folder_actor, "actor")
		actor_target = os.path.join(folder_actor_target, "actor_target")

		self.critic_Q1.save_weights(Q1)
		self.critic_Q2.save_weights(Q2)
		self.critic_Q1_target.save_weights(Q1_target)
		self.critic_Q2_target.save_weights(Q2_target)

		self.actor.save_weights(actor)
		self.actor_target.save_weights(actor_target)

	def load_weights_networks(self, root_folder_src):
		root = root_folder_src
		folder_Q1 = os.path.join(root, "critic_Q1")
		folder_Q2 = os.path.join(root, "critic_Q2")
		folder_Q1_target = os.path.join(root, "critic_Q1_target")
		folder_Q2_target = os.path.join(root, "critic_Q2_target")

		folder_actor = os.path.join(root, "actor")
		folder_actor_target = os.path.join(root, "actor_target")

		Q1 = os.path.join(folder_Q1, "critic_Q1")
		Q2 = os.path.join(folder_Q2, "critic_Q2")
		Q1_target = os.path.join(folder_Q1_target, "critic_Q1_target")
		Q2_target = os.path.join(folder_Q2_target, "critic_Q2_target")

		actor = os.path.join(folder_actor, "actor")
		actor_target = os.path.join(folder_actor_target, "actor_target")

		self.critic_Q1.load_weights(Q1)
		self.critic_Q2.load_weights(Q2)
		self.critic_Q1_target.load_weights(Q1_target)
		self.critic_Q2_target.load_weights(Q2_target)

		self.actor.load_weights(actor)
		self.actor_target.load_weights(actor_target)



