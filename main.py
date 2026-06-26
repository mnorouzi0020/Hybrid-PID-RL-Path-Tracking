#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import numpy as np
import random
import math
import time
import matplotlib.pyplot as plt
from IPython.display import display, clear_output, HTML
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
from time import sleep
from scipy.optimize import minimize
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, LSTM, LeakyReLU
from tensorflow.keras.optimizers import Adam, Nadam, RMSprop, SGD
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.regularizers import l2
import tensorflow_probability as tfp
from sklearn.preprocessing import StandardScaler
from tensorflow.summary import create_file_writer
from tensorboard import notebook
from tensorflow import keras


max_steer = np.radians(40.0)  # [rad] max steering angle
max_steer_tf = tf.constant(40 * (3.14159 / 180), dtype=tf.float32)
L = 2.9  # [m] Wheel base of the vehicle
dt = 0.1
Lr = L / 2.0  # [m]
Lf = L - Lr
Cf = 1600.0 * 2.0  # N/rad
Cr = 1700.0 * 2.0  # N/rad
Iz = 2250.0  # kg/m2
m = 1500.0  # kg
eps = 1e-6

Kp_min, Kp_max = 0, 100.0
Ki_min, Ki_max = 0, 1.0
Kd_min, Kd_max = 0, 100.0


class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.states_bar = []
        self.next_states = []
        self.actions = []
        self.actions_norm = []
        
        self.prev_actions = []
        self.prev_actions_norm = []
        
        self.actions_bar = []
        self.actions_bar_norm = []
        
        self.rewards = []
        self.dones = []
        
        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        # Convert data to float32 here
        return np.array(self.states, dtype=np.float32),            np.array(self.states_bar, dtype=np.float32),            np.array(self.next_states, dtype=np.float32),            np.array(self.actions, dtype=np.float32),            np.array(self.actions_norm, dtype=np.float32),            np.array(self.prev_actions, dtype=np.float32),            np.array(self.prev_actions_norm, dtype=np.float32),            np.array(self.actions_bar, dtype=np.float32),            np.array(self.actions_bar_norm, dtype=np.float32),            np.array(self.rewards, dtype=np.float32),            np.array(self.dones, dtype=np.float32),            batches

    def store_memory(self, state, state_bar, next_state, action, action_norm, prev_action, prev_action_norm, action_bar, action_bar_norm, reward, done):
        # Convert data to float32 before storing
        self.states.append(state)
        self.states_bar.append(state_bar)
        self.next_states.append(next_state)
        
        self.actions.append(action)
        self.actions_norm.append(action_norm)
        
        self.prev_actions.append(prev_action)
        self.prev_actions_norm.append(prev_action_norm)
        
        
        self.actions_bar.append(action_bar)
        self.actions_bar_norm.append(action_bar_norm)
        
        
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        del self.states[:]
        del self.states_bar[:]
        del self.next_states[:]
        
        del self.actions[:]
        del self.actions_norm[:]
        
        del self.prev_actions[:]
        del self.prev_actions_norm[:]
        
        del self.actions_bar[:]
        del self.actions_bar_norm[:]
        
        
        del self.rewards[:]
        del self.dones[:]

        
class KNet(tf.keras.Model):
    def __init__(self, global_lips, k_init):
        super(KNet, self).__init__()
        self.global_lips = global_lips
        if global_lips:
            self.k = tf.Variable(k_init, dtype=tf.float32, trainable=True)
        else:
            self.k = tf.keras.Sequential([
                tf.keras.layers.Dense(32, activation='tanh'),
                tf.keras.layers.Dense(1, activation='softplus')
            ])

    def call(self, inputs):
        if self.global_lips:
            return tf.math.softplus(self.k) * tf.ones((tf.shape(inputs)[0], 1)) # to make the size (batch_size, 1)
        else:
            return self.k(inputs)

        

class Actor_Model_Throttle(Model):
    def __init__(self, state_dim, action_dim):
        super(Actor_Model_Throttle, self).__init__()
        
        self.d1     = Dense(1024, activation='relu', kernel_regularizer=l2(1e-5), kernel_initializer=GlorotUniform())
        self.d2     = Dense(1024, activation='relu', kernel_regularizer=l2(1e-5), kernel_initializer=GlorotUniform())
        self.d3     = Dense(1024, activation='relu', kernel_regularizer=l2(1e-5), kernel_initializer=GlorotUniform())
        #self.d4     = Dense(512, activation='relu', kernel_regularizer=l2(1e-5), kernel_initializer=GlorotUniform())
        
        # Define separate output layers for each head
        self.out = Dense(4, activation=None, kernel_initializer=GlorotUniform()) #Ref. Speed, Kp, Ki, and Kd

    def call(self, x):
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        #x = self.d4(x)
        
        # Apply each head separately and scale to the desired range
        out = self.out(x) 
        
        
        return out

    
class LipsNet(tf.keras.Model):
    def __init__(self, state_dim, action_dim, mode, K_net, eps=1e-4, squash_action=True):
        super(LipsNet, self).__init__()
        self.f_net = Actor_Model_Throttle(state_dim, action_dim)
        self.k_net = K_net
        self.mode = mode
        self.eps = eps
        
        self.squash_action = squash_action
        #self.optimizer = optimizer

    def call(self, x):
        k_out = self.k_net(x)        
        f_out = self.f_net(x)
        
        # Calculate Jacobian of f_net with respect to x
        jacobi = self.compute_jacobian(x) #(batch_size, # of outputs, # of features/states)
        jac_norm = tf.norm(jacobi, axis=-1) #(norm over the last axis) -> (batch_size, # of outputs)
        
        #tf.print('k_out', k_out)
        #tf.print('f_out', f_out)
        #tf.print('jac_norm', jac_norm)
        
        action = k_out * f_out / (jac_norm + self.eps)
        
        #tf.print('action', action)
        
        # Apply custom activation functions to each output
        mu = self.custom_activation(action)
        
        return mu
    
    def compute_jacobian(self, x):
        batch_size = tf.shape(x)[0]
        num_outputs = 4 #Ref. Speed, Kp, Ki, and Kd
        
        jacobi = []
        for i in range(num_outputs):
            with tf.GradientTape() as tape:
                tape.watch(x)
                output_i = self.f_net(x)[:, i]
            jacobi.append(tape.gradient(output_i, x))
        
        return tf.stack(jacobi, axis=1)
    
    def custom_activation(self, action):
        # Define custom activation functions for each output
        # You can define different activation functions here
        act1 = tf.nn.sigmoid(action[:, 0]) * 4  # Output between 0 and 4
        act2 = tf.nn.sigmoid(action[:, 1]) * Kp_max # Kp
        act3 = tf.nn.sigmoid(action[:, 2]) * Ki_max # Ki
        act4 = tf.nn.sigmoid(action[:, 3]) * Kd_max # Kd
        
        mu = tf.stack([act1, act2, act3, act4], axis=1)
        return mu

    
class Critic_Model_Throttle(Model):
    def __init__(self, state_dim, action_dim):
        super(Critic_Model_Throttle, self).__init__()
        
        self.d1     = Dense(1024, activation='relu', kernel_regularizer=l2(1e-5), kernel_initializer=GlorotUniform())
        self.d2     = Dense(1024, activation='relu', kernel_regularizer=l2(1e-5), kernel_initializer=GlorotUniform())
        self.d3     = Dense(1024, activation='relu', kernel_regularizer=l2(1e-5), kernel_initializer=GlorotUniform())
        #self.d4     = Dense(512, activation='relu', kernel_regularizer=l2(1e-5), kernel_initializer=GlorotUniform())
        self.value  = Dense(1, activation='linear', kernel_initializer=GlorotUniform())

    def call(self, x):
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        #x = self.d4(x)
        return self.value(x)
        
        
        
        
# PPO Actor-Critic DRL Function
class DRLController():
    def __init__(self, state_dim_throttle, action_dim, learning_rate, gamma, lambda_,
                 batch_size, best_avg_reward_throttle, decay_rate, entropy_coefficient_throttle,
                 policy_kl_range, policy_params, value_clip, update_freq, num_epochs, lam_T, lam_S, lam_V,
                 lamda_k, K_lr, decay_rate_std, mode):
        ### Hyperparameters
        #self.state_dim_throttle = state_dim_throttle
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.lambda_ = lambda_
        # Decay rate for exponential learning rate decay
        self.decay_rate = decay_rate
        # Specify the batch size
        #self.batch_size = batch_size
        # Initialize variables to keep track of the best actor weights and average reward (for IML)
        self.best_avg_reward_throttle = best_avg_reward_throttle 
        self.entropy_coefficient_throttle = entropy_coefficient_throttle
        
        self.policy_kl_range = policy_kl_range #-> Fixed (15 km/h increase or decrease in speed)
        self.policy_params = policy_params     #High-> too much exploit/ low-> too much explore
                                               #High->High positive total loss/ Low-> low penalty-> more explore
        self.value_clip = value_clip
        self.update_freq = update_freq
        self.num_epochs = num_epochs
        
        self.lam_T = lam_T
        self.lam_S = lam_S
        self.lam_V = lam_V
        
        self.lamda_k = lamda_k
        self.K_lr = K_lr
        
        self.decay_rate_std = decay_rate_std
        
        self.mode = mode #Training = 1(True), Inference = 0(False)
        
        
        self.std = [2.0, Kp_max/3, Ki_max/3, Kd_max/3] * tf.ones([1, action_dim])
        # Minimum standard deviation values
        self.min_std = tf.constant([0.5, 3.0, 0.2, 2.0], dtype=tf.float32)
        # Decay rate for reducing standard deviation
        
        
        ### Actor/Critic + info
        ### Actor Network + Clone the actor model to create an "old" actor
        self.Knet = KNet(global_lips=False, k_init=30.0)
        
        self.actor_throttle = LipsNet(state_dim_throttle, action_dim, self.mode, self.Knet)
        self.old_actor_throttle = LipsNet(state_dim_throttle, action_dim, self.mode, self.Knet)
        
        
        ### Critic Network + Clone the critic model to create an "old" critic
        self.critic_throttle = Critic_Model_Throttle(state_dim_throttle, action_dim)
        self.old_critic_throttle = Critic_Model_Throttle(state_dim_throttle, action_dim)

        # Build the loss functions and optimizers
        self.optimizer = Adam(learning_rate = self.learning_rate)
        self.optimizer_Knet = Adam(learning_rate = self.K_lr)

        ### Classes
        self.Memory = PPOMemory(batch_size)
        self.PID_throttle = PID_Controller_Throttle(kp_speed=5.0, ki_speed=0.01, kd_speed=10.0)
        self.PID_steering = PID_Controller(kp_position=0.0, ki_position=0.0, kd_position=0.0)
        # Create an instance of the NonLinearBicycleModel class
        self.vehicle_model = NonLinearBicycleModel(x=0.0, y=0.0, yaw=0.0)

        
    def store_transition(self, state, state_bar, next_state, action, action_norm, prev_action, prev_action_norm, action_bar, action_bar_norm, reward, done):
        self.Memory.store_memory(state, state_bar, next_state, action, action_norm, prev_action, prev_action_norm, action_bar, action_bar_norm, reward, done)
        
        
    @tf.function
    def get_action_throttle(self, state_throttle):
        state_throttle = tf.expand_dims(tf.cast(state_throttle, dtype = tf.float32), 0)  # Add an extra dimension for batch_size of 1 (because it shows that (1,# of states) is the input and not (None, # of states))
        
        throttle_mu = self.actor_throttle(state_throttle)
        
        ### Sample actions only in the training mode otherwise, choose the mean
        if self.mode:
            distribution = tfp.distributions.Normal(throttle_mu, self.std)
            action_throttle = distribution.sample()
            #tf.print('Training')
            
        else:
            action_throttle = throttle_mu
            #tf.print('Inference')
                    
        ref_speed = tf.clip_by_value(action_throttle[:, 0], 0.0, 4.0)
        Kp = tf.clip_by_value(action_throttle[:, 1], 0.0, Kp_max)
        Ki = tf.clip_by_value(action_throttle[:, 2], 0.0, Ki_max)
        Kd = tf.clip_by_value(action_throttle[:, 3], 0.0, Kd_max)
        
        # Ensure the clipped actions have the same shape as action_throttle
        ref_speed = tf.expand_dims(ref_speed, axis=1)
        Kp = tf.expand_dims(Kp, axis=1)
        Ki = tf.expand_dims(Ki, axis=1)
        Kd = tf.expand_dims(Kd, axis=1)
        
        
        # Concatenate the clipped actions to form the final action tensor
        actions_tot = tf.concat([ref_speed, Kp, Ki, Kd], axis=1)
                        
        # During post-training inference, use the mean action without noise
        return actions_tot, throttle_mu

    def norm_action(self, action):
        # Define the range for each action
        ref_speed_min, ref_speed_max = 0.0, 4.0
        
        # Define the normalized range
        normalized_min, normalized_max = -1.0, 1.0

        # Normalize the reference speed
        normalized_ref_speed = tf.clip_by_value(
            (action[0] - ref_speed_min) / (ref_speed_max - ref_speed_min) * (normalized_max - normalized_min) + normalized_min,
            clip_value_min=normalized_min,
            clip_value_max=normalized_max
        )

        # Normalize Kp
        normalized_Kp = tf.clip_by_value(
            (action[1] - Kp_min) / (Kp_max - Kp_min) * (normalized_max - normalized_min) + normalized_min,
            clip_value_min=normalized_min,
            clip_value_max=normalized_max
        )
        
        # Normalize Ki
        normalized_Ki = tf.clip_by_value(
            (action[2] - Ki_min) / (Ki_max - Ki_min) * (normalized_max - normalized_min) + normalized_min,
            clip_value_min=normalized_min,
            clip_value_max=normalized_max
        )
        
        # Normalize Kd
        normalized_Kd = tf.clip_by_value(
            (action[3] - Kd_min) / (Kd_max - Kd_min) * (normalized_max - normalized_min) + normalized_min,
            clip_value_min=normalized_min,
            clip_value_max=normalized_max
        )

        # Stack the normalized values into a single tensor
        normalized_actions_tensor = tf.stack([normalized_ref_speed, normalized_Kp, normalized_Ki, normalized_Kd], axis=0)
        
        return normalized_actions_tensor

    def update_learning_rate(self):
        self.learning_rate *= self.decay_rate
        
    def update_std(self):
        # Check if the current std is greater than the minimum std
        mask = tf.greater(self.std, self.min_std)

        # Reduce the std towards the minimum value with the decay rate
        self.std = tf.where(mask, self.std * self.decay_rate_std, self.min_std)
                         #Condition,        if True,            , if False
        

    def save_weights(self):
        self.actor_throttle.save_weights('E:\Saved Models Friction new/actor_ppo', save_format='tf')
        self.old_actor_throttle.save_weights('E:\Saved Models Friction new/actor_old_ppo', save_format='tf')
        self.critic_throttle.save_weights('E:\Saved Models Friction new/critic_ppo', save_format='tf')
        self.old_critic_throttle.save_weights('E:\Saved Models Friction new/critic_old_ppo', save_format='tf')
        
    def load_weights(self):
        self.actor_throttle.load_weights('E:\Saved Models Friction new/actor_ppo')
        self.old_actor_throttle.load_weights('E:\Saved Models Friction new/actor_old_ppo')
        self.critic_throttle.load_weights('E:\Saved Models Friction new/critic_ppo')
        self.old_critic_throttle.load_weights('E:\Saved Models Friction new/critic_old_ppo')
        
    def jeffreys_divergence(self, action1, action2):
        # Ensure probabilities sum up to 1
        p = tf.nn.softmax(action1, axis=1)
        q = tf.nn.softmax(action2, axis=1)

        # Compute the KL divergences in both directions
        kl_pq = tf.reduce_sum(p * tf.math.log(p / q), axis=1)
        kl_qp = tf.reduce_sum(q * tf.math.log(q / p), axis=1)

        # Compute the average of the two KL divergences
        jeffreys_div = 0.5 * (tf.reduce_mean(kl_pq) + tf.reduce_mean(kl_qp))

        return jeffreys_div

    @tf.function
    def train_step(self, states_throttle, states_bar, next_states_throttle, actions_throttle, actions_throttle_norm, prev_actions, prev_actions_norm, action_bar, action_bar_norm, rewards_throttle, path_completeds):
        states_throttle = tf.convert_to_tensor(states_throttle, dtype=tf.float32)
        states_bar = tf.convert_to_tensor(states_bar, dtype=tf.float32)
        
        #print()
        #tf.print("states in the train_step function are: ", tf.shape(states_throttle))
        #print()
        
        next_states_throttle = tf.convert_to_tensor(next_states_throttle, dtype=tf.float32)
        
        #print()
        #tf.print("next states in the train_step function are: ", tf.shape(next_states_throttle))
        #print()
        
        actions_throttle = tf.convert_to_tensor(actions_throttle, dtype=tf.float32)
        prev_actions = tf.convert_to_tensor(prev_actions, dtype=tf.float32)
        action_bar = tf.convert_to_tensor(action_bar, dtype=tf.float32)
        
        actions_throttle_norm = tf.convert_to_tensor(actions_throttle_norm, dtype=tf.float32)
        prev_actions_norm = tf.convert_to_tensor(prev_actions_norm, dtype=tf.float32)
        action_bar_norm = tf.convert_to_tensor(action_bar_norm, dtype=tf.float32)
        
        #print()
        #tf.print("actions_throttle in the train_step function are: ", tf.shape(actions_throttle))
        #print()
        #print()
        #tf.print("actions_throttle in the train_step function are: ", tf.shape(prev_actions))
        #print()
        #print()
        #tf.print("actions_throttle in the train_step function are: ", tf.shape(action_bar))
        #print()
        
        # Remove singleton dimensions
        #actions_throttle = tf.squeeze(actions_throttle, axis=-1)
          
        #rewards_throttle = tf.convert_to_tensor(rewards_throttle, dtype=tf.float32)
        rewards_throttle = tf.convert_to_tensor(rewards_throttle, dtype=tf.float32)
        rewards_throttle = tf.expand_dims(rewards_throttle, axis=-1)
        # Remove singleton dimensions
        #rewards = tf.squeeze(rewards, axis=-1)
        #print()
        #tf.print("rewards in the train_step function are: ", tf.shape(rewards_throttle))
        #print()
        
        #path_completeds = tf.cast(path_completeds, tf.float32) #boolean to float (False=0, True=1)
        path_completeds = tf.convert_to_tensor(path_completeds, dtype=tf.float32)
        path_completeds = tf.expand_dims(path_completeds, axis=-1)
        #print()
        #tf.print("path_completeds in the train_step function are: ", tf.shape(path_completeds))
        #print()
        
        
        #critic_value_ad = tf.sueeze(critic_value_ad, 1)


        #Calculate critic loss
        with tf.GradientTape(persistent=True) as tape:
            
            k_out = self.Knet(states_throttle)
            tape.watch(k_out)
            lips_loss = self.lamda_k * tf.reduce_mean(tf.square(k_out))
            
            #### Actor (mu) + Values
            mu_throttle = self.actor_throttle(states_throttle)
            
            #print()
            #tf.print("mu_throttle in the train_step function are: ", tf.shape(mu_throttle))
            #print()
            
            values = self.critic_throttle(states_throttle)
            values_bar = self.critic_throttle(states_bar)
            #print()
            #tf.print("values in the train_step function are: ", tf.shape(values))
            #print()
            

            #### Old actor (old_mu) + Old values
            old_mu = self.old_actor_throttle(states_throttle)
            
            #print()
            #tf.print("old_sigma in the train_step function are: ", tf.shape(old_sigma))
            #print()
            
            old_values = self.old_critic_throttle(states_throttle)
            
            #print()
            #tf.print("old_values in the train_step function are: ", tf.shape(old_values))
            #print()

            
            #### Next_state value
            next_values = self.critic_throttle(next_states_throttle)
            #critic_value_next_state = tf.squeeze(critic_value_next_state, 1)
            #critic_throttle_value_next_state = tf.convert_to_tensor(critic_throttle_value_next_state, dtype=tf.float32)
            ####

            #print()
            #tf.print("next_values in the train_step function are: ", tf.shape(next_values))
            #print()
            
            #####
            old_values_stop = tf.stop_gradient(old_values)
            
            advantages_throttle = compute_gae(values, next_values, rewards_throttle, path_completeds, gamma=self.gamma, lambda_=self.lambda_)
            
            returns_throttle = tf.stop_gradient(advantages_throttle + values)
            #print()
            #tf.print("returns in the train_step function are: ", tf.shape(returns_throttle))
            #print()
            
            advantages_throttle = tf.stop_gradient((advantages_throttle - tf.math.reduce_mean(advantages_throttle)) / (tf.math.reduce_std(advantages_throttle) + 1e-6))
            
            #advantages_throttle = tf.stop_gradient(advantages_throttle)
            
            #print()
            #tf.print("tensor-converted advantages in the train_step function are: ", tf.shape(advantages_throttle))
            #print()
            
            
            distribution_new = tfp.distributions.Normal(mu_throttle, self.std)
            log_prob_new_throttle = distribution_new.log_prob(actions_throttle)
            
            #log_prob_new_throttle = tf.reduce_sum(log_prob_new_throttle, 1)
            
            #print()
            #tf.print("log_prob_new_throttle in the train_step function are: ", tf.shape(log_prob_new_throttle))
            #print()
            

            distribution_old = tfp.distributions.Normal(old_mu, self.std)
            log_prob_old_throttle = tf.stop_gradient(distribution_old.log_prob(actions_throttle))
            
            #print()
            #tf.print("log_prob_old_throttle in the train_step function are: ", tf.shape(log_prob_old_throttle))
            #print()


            #### Obtain Prob_ratio -> Clip_prob_ratio to avoid NaN
            #if Nan-> the policy has diverged too much from the old one
            prob_ratio_throttle = tf.math.exp(log_prob_new_throttle - log_prob_old_throttle) #(Equals to dividing these since its log)
            #prob_ratio_throttle = tf.clip_by_value(prob_ratio_throttle, 0.0, 50.0)
            #print()
            #tf.print("prob_ratio in the train_step function are: ", tf.shape(prob_ratio_throttle))
            #print()
            
            
            #### KL calculations
            Kl = tfp.distributions.kl_divergence(distribution_old, distribution_new)
            #Kl = tf.clip_by_value(Kl, 0.0, 100.0)
            #print()
            #tf.print("Kl in the train_step function are: ", tf.shape(Kl))
            #print()

            ################### Truly PPO ###################
            
            pg_loss = tf.where(
                tf.logical_and(Kl >= self.policy_kl_range, prob_ratio_throttle > 1),
                prob_ratio_throttle * advantages_throttle - self.policy_params * Kl,
                prob_ratio_throttle * advantages_throttle
            )
            pg_loss = tf.math.reduce_mean(pg_loss)
            
            ################### CAPS ###################
            
            #Distance= (1/batch_size) (normalize across the bathc) * (sum of all sqrt of difference ** 2)
            #LT = tf.math.reduce_mean(tf.norm(actions_throttle - prev_actions, axis=1))
            LT = self.jeffreys_divergence(actions_throttle_norm, prev_actions_norm)
            #LS = tf.math.reduce_mean(tf.norm(actions_throttle - action_bar, axis=1))
            LS = self.jeffreys_divergence(actions_throttle_norm, action_bar_norm)
            
            CAPS = (self.lam_T*LT) + (self.lam_S*LS)
            #print()
            #tf.print("CAPS in the train_step function are: ", CAPS)
            #print()
            #CAPS = tf.math.reduce_mean(CAPS)
            
            pg_loss = pg_loss - CAPS

            #print()
            #tf.print("pg_loss in the train_step function are: ", tf.shape(pg_loss))
            #print()
            
            ################### ################### ################### ################### ###################

            # Compute entropy
            # Sum over the dimensions to get the total entropy
            entropy_loss_throttle = tf.math.reduce_mean(distribution_new.entropy())
            #entropy_loss_throttle = tf.squeeze(entropy_loss_throttle, axis=-1)

            #print()
            #tf.print("entropy_loss_total in the train_step function are: ", tf.shape(entropy_loss_throttle))
            #print()

            # Getting critic loss by using Clipped critic value
            vpredclipped    = old_values + tf.clip_by_value(values - old_values_stop, -self.value_clip, self.value_clip) # Minimize the difference between old value and new value
            vf_losses1      = tf.math.square(returns_throttle - values) * 0.5 # Mean Squared Error
            vf_losses2      = tf.math.square(returns_throttle - vpredclipped) * 0.5 # Mean Squared Error
            
            Smooth_C_term = self.lam_V * tf.math.reduce_mean(tf.math.square(values - values_bar))
            
            #print()
            #tf.print("Smooth_C_term in the train_step function are: ", Smooth_C_term)
            #print()
            
            critic_loss = tf.math.reduce_mean(tf.math.maximum(vf_losses1, vf_losses2)) + Smooth_C_term  
            
            #print()
            #tf.print("critic_loss in the train_step function are: ", tf.shape(critic_loss))
            #print()
            
            Total_Loss = critic_loss - pg_loss - self.entropy_coefficient_throttle*entropy_loss_throttle
            
            Loss_prime = Total_Loss + lips_loss
            #print()
            #tf.print("Total_Loss is: ", tf.shape(Total_Loss))
            #print()
            
        #print(f"entropy_loss (throttle): {entropy_loss_throttle}")
        #print(f"actor loss (throttle): {actor_loss_throttle}")
        #print(f"critic loss (throttle): {critic_loss_throttle}")
        
        
        gradients = tape.gradient(Loss_prime, self.actor_throttle.trainable_variables + self.critic_throttle.trainable_variables)
        #clipped_gradients, _ = tf.clip_by_global_norm(gradients, 0.5)
        self.optimizer.apply_gradients(zip(gradients, self.actor_throttle.trainable_variables + self.critic_throttle.trainable_variables)) 
        
        
        # Perform assignment
        grads = tape.gradient(Loss_prime, self.Knet.trainable_variables)
        # Reduce gradients to scalar
        #grads = tf.reduce_mean(grads)
        #clipped_gradients, _ = tf.clip_by_global_norm(gradients, 0.5)
        self.optimizer_Knet.apply_gradients(zip(grads, self.Knet.trainable_variables)) 

        #return Total_Loss, critic_loss, pg_loss, entropy_loss_throttle
        
        del tape
        

    # Function to perform training using Proximal Policy Optimization (PPO)
    def train_ppo_agent(self, num_episodes, dt, num_ittr):
        
        show_anim = False
                        
        avg_reward_throttle = []
        
        episode_count_heading = 0  # Initialize episode count
        Learning_made_sure = False
        
        # Initialize some variables for collecting batch data
        
        update_time = 0

        for episode in range(1, num_episodes + 1):
            
            initial_sine_mag = 5 #random.uniform(-5, 5)
            
            tolerance_pos_err = np.abs(initial_sine_mag)/12.5
            tolerance_ang_err = np.radians(np.abs(initial_sine_mag))
            
            path = generate_path(initial_sine_mag)
            path_angles = calculate_path_angles(path)
            checkpoints = []

            for i in range(0, len(path), 100):
                checkpoints.append(path[i])

            tracker = CheckpointsTracker(path, checkpoints)
            ####################################################################
            #if episode>= 100:
            #    show_anim = True

            # Reset the vehicle position and heading for each episode
            #To avoid overfitting, we start the episode with a little error from the main path
            initial_x, initial_y = path[0][0], path[0][1] + random.choice([-1, 1])*random.uniform(0.0, 0.05)
            
            current_position = np.array([initial_x, initial_y])
            initial_heading = path_angles[0] + np.radians(random.choice([-1, 1])*random.uniform(0, 1))  # Initial heading

            prev_pos = np.array([initial_x, initial_y])
            prev_heading = np.copy(initial_heading)

            # Set the current heading based on the updated initial_heading
            current_heading = initial_heading
            
            predict_trajectory = predict_trajectory_linear(path, current_position, current_heading, num_steps=60)

            self.vehicle_model.x = current_position[0]
            self.vehicle_model.y = current_position[1]
            self.vehicle_model.yaw = current_heading
            self.vehicle_model.vx = 0.0
            self.vehicle_model.vy = 0.0
            
            
            # Calculate the lookahead point (recalculate it for each frame)
            lookahead_point, lookahead_heading = find_lookahead_point(path, path_angles, current_position)
            
            tracker.update(lookahead_point)
            
            # To get the points ahead (Like MPC)
            points_ahead, points_ahead_heading = NthPointAhead(path, current_position, n=60)
            
            # Calculate the Euclidean distance between corresponding points
            pred_vs_future_dist = np.linalg.norm(predict_trajectory[-1] - points_ahead[-1])
            
            ############################################################
            ### check if the future changes in the reference path is drastic
            #if np.abs(points_ahead_heading - lookahead_heading) >= np.radians(10):
            #    Drastic_path_change = True
            #else:
            #    Drastic_path_change = False
                
            ### check if the current and the future are different -> you shold lower your speed
            #if np.abs(points_ahead_heading - current_heading) >= np.radians(10):
            #    Drastic_speed_change_needed = True
            #else:
            #    Drastic_speed_change_needed = False
            ############################################################
            
            # Calculate position error and angle error
            pos_error = calculate_position_error(path, current_position, tolerance = tolerance_pos_err)
            #tf.print("pos_error", pos_error)
            ang_error = calculate_angle_error(current_position, current_heading, path_angles, path, tolerance = tolerance_ang_err)

            # Calculate the steering angle using the DRL controller
            pid = 0.0
            #tf.print("pid_steering_angle", pid_steering_angle)

            path_completed = False
            
            ### PID_Throttle instance
            spd_error = 0.0
            throttle = self.PID_throttle.calculate_throttle(spd_error)
            Speed_x = 0.0
            prev_ref_speed = 0.0
            ref_spd_diff = 0.0
            
            prev_steering = 0.0
            steering_diff = 0.0
            
            
            steering = 0.0
            prev_Kp = 0.0
            prev_Ki = 0.0
            prev_Kd = 0.0
            prev_Beta = 0.0
            Beta_diff = 0.0
            Vx = 0.2
            Vy = 0.0
            Beta = 0.0
            alpha_f = 0.0
            alpha_r = 0.0
            omega = 0.0
            Ffy = 0.0
            Fry = 0.0
            
            ####Make a variable episode length based on the total movement of the vehicle ####
            #If the vehicle has not moved a lot, we would penalize it, and end the episode
            #The longer the episode takes, the better the agent is performing
            total_movement = 0.0
            Not_Moved_Enough = False
            
            
            ####################################################################
            ########################
            
            # Convert boolean values to float
            checkpoint_reached_float = [float(flag) for flag in tracker.checkpoint_reached]
            
            prev_checkpoint_reached_float = [float(flag) for flag in [False]*10]
            
            # Flatten the checkpoints list to include each coordinate separately
            checkpoints_flat = [coord for checkpoint in checkpoints for coord in checkpoint]
            
            points_ahead_flat = points_ahead[-1].flatten()
            points_ahead_heading_flat = points_ahead_heading[-1].flatten()
            predict_trajectory_flat = predict_trajectory[-1].flatten()
            
            
            state_throttle = np.array([
                initial_sine_mag,                                          #Path related Info
                current_position[0], current_position[1], current_heading, #Positional Info
                Speed_x, Vy, throttle,                                         #Speed 
                steering, Beta, alpha_f, alpha_r, omega,
                pos_error, ang_error,                                        #Error + Future Info
                *checkpoints_flat,                                          #Checkpoints location + Reached? Info
                predict_trajectory_flat[0], predict_trajectory_flat[1],    #Predicted Trajectory Info
                points_ahead_flat[0], points_ahead_flat[1]                 #Future Info
            ]).flatten()
            
            # Generate Gaussian noise for the elements before checkpoint_reached_float
            #noise_before = np.random.normal(loc=0, scale=0.1, size=11)

            # Generate Gaussian noise for the elements after checkpoint_reached_float
            #noise_after = np.random.normal(loc=0, scale=0.1, size=24)

            # Concatenate the noise vectors accordingly
            noise = np.random.normal(loc=0, scale=0.1, size=38)
            # Add the noise to the state vector
            state_bar = state_throttle + noise
            
            ########################
            episode_len = 0
            
            rewards_throttle = [] #For avg reward calculation
            spd_all = []
            ref_spd_all = []
            steering_all = []
            throttle_all = []
            current_pos_all = []
            
            spd_all.append(0.0)
            ref_spd_all.append(0.0)
            steering_all.append(0.0)
            current_pos_all.append(current_position)
            
            kp_all = []
            kp_all.append(0.0)
            
            ki_all = []
            ki_all.append(0.0)
            
            kd_all = []
            kd_all.append(0.0)
            
            Beta_tot = []
            Beta_tot.append(0.0)
            
            omega_tot = []
            omega_tot.append(0.0)
            
            Max_Episode_len = 1000
            
            #How long does each episode take?
            start_time = time.time()
            for frame in range(Max_Episode_len):
                
                actions_tot, mu_throttle = self.get_action_throttle(state_throttle)
                
                #Get action bar
                actions_tot_bar, _ = self.get_action_throttle(state_bar)
                
                
                ref_speed = actions_tot[0][0]
                Kp_str = actions_tot[0][1]
                Ki_str = actions_tot[0][2]
                Kd_str = actions_tot[0][3]
                
                self.PID_steering.kp_position = Kp_str.numpy()
                self.PID_steering.ki_position = Ki_str.numpy()
                self.PID_steering.kd_position = Kd_str.numpy()
                
                kp_all.append(Kp_str.numpy())
                ki_all.append(Ki_str.numpy())
                kd_all.append(Kd_str.numpy())
                
                ### Obtain Steering
                steering = self.PID_steering.calculate_steering(pos_error, ang_error)
                ####################################################################
                #update prev_position:
                prev_pos = np.copy(current_position)
                
                ### PID_Throttle
                spd_error = ref_speed.numpy() - Speed_x
                
                ### Obtain Throttle
                throttle = self.PID_throttle.calculate_throttle(spd_error)
                throttle_all.append(throttle)
                                
                
                # Calculate the steering angle using the DRL controller
                #pid_steering_angle = self.PID_steering.calculate_steering(pos_error, ang_error)
                steering_all.append(steering)
                ref_spd_all.append(ref_speed.numpy())
                
                # Call the function to update the location and heading
                # Update the vehicle model using the provided inputs
                Speed_x, Vy, alpha_f, alpha_r, Beta, omega, Ffy, Fry = self.vehicle_model.update(throttle, steering)
                
                spd_all.append(Speed_x)
                Beta_tot.append(np.degrees(Beta))
                #omega_tot.append(np.degrees(omega))
                #### 
                
                # Update the current location and heading
                current_position[0] = self.vehicle_model.x
                current_position[1] = self.vehicle_model.y
                
                current_position = np.array([current_position[0], current_position[1]])
                
                current_pos_all.append(current_position)

                current_heading = self.vehicle_model.yaw
                
                predict_trajectory = predict_trajectory_linear(path, current_position, current_heading, num_steps=60)
                
                total_movement += np.linalg.norm(current_position[0] - prev_pos[0])  # Euclidean distance as a measure of movement
                
                # Calculate the lookahead point (recalculate it for each frame)
                lookahead_point, lookahead_heading = find_lookahead_point(path, path_angles, current_position)
                
                tracker.update(lookahead_point)
                #print("Checkpoint reached:", tracker.checkpoint_reached)

                # To get the points ahead (Like MPC)
                points_ahead, points_ahead_heading = NthPointAhead(path, current_position, n=60)
                
                # Calculate the Euclidean distance between corresponding points
                pred_vs_future_dist = np.linalg.norm(predict_trajectory[-1] - points_ahead[-1])
                
                ############################################################
                ### check if the future changes in the reference path is drastic
                #if np.abs(points_ahead_heading - lookahead_heading) >= np.radians(10):
                #    Drastic_path_change = True
                #else:
                #    Drastic_path_change = False
                    
                ### check if the current and the future are different -> you shold lower your speed
                #if np.abs(points_ahead_heading - current_heading) >= np.radians(10):
                #    Drastic_speed_change_needed = True
                #else:
                #    Drastic_speed_change_needed = False
                ############################################################

                # Calculate position error and angle error
                pos_error = calculate_position_error(path, current_position, tolerance = tolerance_pos_err)
                
                ang_error = calculate_angle_error(current_position, current_heading, path_angles, path, tolerance = tolerance_ang_err)                

                
                #########
                #########
                # Convert boolean values to float
                checkpoint_reached_float = [float(flag) for flag in tracker.checkpoint_reached]
                

                points_ahead_flat = points_ahead[-1].flatten()
                points_ahead_heading_flat = points_ahead_heading[-1].flatten()
                predict_trajectory_flat = predict_trajectory[-1].flatten()
                
                ref_spd_diff = np.abs(ref_speed.numpy() - prev_ref_speed)
                steering_diff = np.abs(steering - prev_steering)
                
                Beta_diff = np.abs(Beta - prev_Beta)
                
                next_state_throttle = np.array([
                    initial_sine_mag,                                          #Path related Info
                    current_position[0], current_position[1], current_heading, #Positional Info
                    Speed_x, Vy, throttle,                                         #Speed 
                    steering, Beta, alpha_f, alpha_r, omega,
                    pos_error, ang_error,                                        #Error + Future Info
                    *checkpoints_flat,                                          #Checkpoints location + Reached? Info
                    predict_trajectory_flat[0], predict_trajectory_flat[1],    #Predicted Trajectory Info
                    points_ahead_flat[0], points_ahead_flat[1]                 #Future Info
                ]).flatten()

                ### Generate S bar
                # Add the noise to the state vector
                next_state_bar = next_state_throttle + noise
                
                #########
                #########
                episode_len = episode_len + 1
                update_time = update_time + 1
                
                #Display values
                pos_error_scalar = float(pos_error) if isinstance(pos_error, (int, float, np.float32, np.ndarray)) else pos_error.item()
                ang_error_scalar = float(np.degrees(ang_error)) if isinstance(np.degrees(ang_error), (int, float, np.float32, np.ndarray)) else np.degrees(ang_error).item()
                #reference_spd_scalar = float(reference_spd.numpy()) if isinstance(reference_spd, (int, float, np.float32, np.ndarray)) else reference_spd.numpy().item()
                Speed_x_scalar = float(Speed_x) if isinstance(Speed_x, (int, float, np.float32, np.ndarray)) else Speed_x
                
                #Check if we have reached the end of the path
                path_completed = PathIsComplete(path, current_position, lookahead_point, tracker.checkpoint_reached)
                
                
                ####### Reward calculation #######
                    
                

                #Low error reward
                low_error_rew = 1000 * (0.7 ** (np.abs(pos_error))) #+ 300 * (0.8 ** np.abs(np.degrees(ang_error)))
                
                # Checkpoint reward (given only if the agent visits a new checkpoint)
                if (prev_checkpoint_reached_float == checkpoint_reached_float):
                    reach_check_rew = 0
                else:
                    reach_check_rew = 100 * sum(checkpoint_reached_float)
                    
                #reach_check_rew = 100 * sum(checkpoint_reached_float)
                prev_checkpoint_reached_float = checkpoint_reached_float
                
                #Employ the future info with the predicted movement
                future_aware_rew = 1000 * (0.7 ** (pred_vs_future_dist))
                
                #if (np.abs(pos_error) <= tolerance_pos_err) and (np.abs(ang_error) <= tolerance_ang_err) and (pred_vs_future_dist <= 0.02):
                #    ASAP = 1000 * (Speed_x/4.0) #Try to be high (close to 1)
                    
                #elif (np.abs(pos_error) <= tolerance_pos_err) and (pred_vs_future_dist <= 0.02):
                #    ASAP = (1/2) * 1000 * (Speed_x/4.0)
                    
                #elif (np.abs(pos_error) <= tolerance_pos_err) and (pred_vs_future_dist <= 0.1):
                #    ASAP = (1/5) * 1000 * (Speed_x/4.0)
                    
                #else:
                #    ASAP = 1000 * (0.8 ** Speed_x)
                

                if (pred_vs_future_dist <= 0.1):
                    if (Speed_x >= 1.0) and (Speed_x <= 4.0):
                        ASAP = 1000.0
                    else:
                        ASAP = 0.0
                else:
                    if (Speed_x >= 0.3) and (Speed_x <= 0.9):
                        ASAP = 700.0
                    else:
                        ASAP = 0.0
                        
                #ASAP = 1000 * (Speed_x/4.0) #Try to be high (close to 1)
                #ASAP = -25*Speed_x
                
                ### Drift:
                #alpha_f, alpha_r, Beta, omega
                # Reward for side slip angle (Beta) (Beta must be high to make the drifting happen)
                if (np.degrees(np.abs(Beta)) >= 10) and (np.degrees(np.abs(Beta)) <= 40):
                    slip_reward = 1000.0
                else:
                    slip_reward = 10.0
                    
                #Beta_diff_rew = 1000 * (0.8 ** (np.degrees(Beta_diff)))
                #slip_reward = -10*(np.abs(Beta))

                # Reward for balanced slip angles (the difference should be lowered)
                #slip_balance = 1000 * (0.9 ** (np.degrees(np.abs(alpha_f - alpha_r))))
                #slip_balance = -10*(np.abs(alpha_f - alpha_r))

                # Reward for maintaining control (yaw rate) (yaw rate should be low for stable drift)
                #if (np.degrees(np.abs(omega)) >= 0) and (np.degrees(np.abs(omega)) <= 80):
                #    yaw_rate_reward = 1000
                #else:
                #    yaw_rate_reward = 0
                    
                #yaw_rate_reward = 1000 * (0.9 ** (np.degrees(np.abs(omega))))
                #yaw_rate_reward = -10*(np.abs(omega))
                ###
                
                #reward_throttle = 0.25*low_error_rew + 0.15*reach_check_rew + 0.2*future_aware_rew + 0.4*ASAP
                reward_throttle = 0.4*low_error_rew + 0.35*future_aware_rew + 0.25*ASAP
                
                #reward_throttle = 0.6 * throttle_term1 + 0.4 * throttle_term2 #+ w3 * ref_penalty
                #if ((episode_len%30 == 0) and episode_len < Max_Episode_len) and (total_movement <= 0.1): #If the vehicle has not moved enough at the beginning
                #    reward_throttle = -20.0 #0*throttle is used for the correct tensor calculation
                #    Not_Moved_Enough = True
                #    total_movement = 0.0
                
                if (episode_len == Max_Episode_len) or (np.abs(pos_error) >= 5.0):# or ((np.degrees(np.abs(steering - prev_steering))>=75) and (episode_len>=10)): #If you have not reached the end of the path and the maximum number of steps has reached
                    #Penalize the agent for not completing the path
                    reward_throttle = -100.0
                
                #if (path_completed):
                #    reward_throttle = +200.0
                ####################################################################
                ##If Not_Moved_Enough or end of episode -> terminate -> value = 0 -> augment with path_completed
                
                if (episode_len == Max_Episode_len) or (np.abs(pos_error) >= 5.0):# or ((np.degrees(np.abs(steering - prev_steering))>=75) and (episode_len>=10)):
                    path_completed = True
                #### batch_path_completed.append(path_completed)
                
                    
                # Print the reward and its components for debugging
                # The absolute values of pos_error and ang_error are used in the reward, but the real values are displayed
                reward_throttle_scalar = float(reward_throttle) if isinstance(reward_throttle, (int, float, np.float32)) else reward_throttle
                
                if show_anim:
                    update(current_position, lookahead_point, current_heading, predict_trajectory, pred_vs_future_dist, pos_error_scalar, ang_error_scalar, steering, frame, path, path_angles, checkpoints, tracker.checkpoint_reached, ref_speed.numpy(), Speed_x_scalar, None, reward_throttle_scalar, throttle, points_ahead, None, None)
                
                # Convert mu to degrees
                #mu_degrees = np.degrees(mu)

                if episode_len%10 == 0 or episode_len == 1:
                    print(episode_len)
                    print(f"Reward_Throttle= {reward_throttle_scalar:.2f} pos_error= {pos_error_scalar:.2f}, ang_error= {ang_error_scalar:.2f}, mu_throttle= {mu_throttle} , sigma_throttle= {self.std}, Speed_x= {Speed_x_scalar:.2f}")

                min_reward = -100.0
                new_min = -1.0
                
                max_reward = 1000.0
                new_max = 1.0
                
                normalized_rew = ((reward_throttle - min_reward) / (max_reward - min_reward)) * (new_max - new_min) + new_min
                
                ##Previous actions
                prev_action = tf.constant([prev_ref_speed, prev_Kp, prev_Ki, prev_Kd])
                
                
                ### Action Normalization
                norm_action_tot = self.norm_action(actions_tot[0])
                norm_prev_action = self.norm_action(prev_action)
                norm_action_tot_bar = self.norm_action(actions_tot_bar[0])
                
                ### Store transitions only in the training mode
                if self.mode:
                    # Store the state, action, and reward for training
                    self.store_transition(state_throttle, state_bar, next_state_throttle, actions_tot[0], norm_action_tot, prev_action, norm_prev_action, actions_tot_bar[0], norm_action_tot_bar, normalized_rew, float(path_completed))

                
                prev_ref_speed = ref_speed.numpy()
                prev_Kp = Kp_str.numpy()
                prev_Ki = Ki_str.numpy()
                prev_Kd = Kd_str.numpy()
                prev_Beta = Beta

                prev_steering = steering
                
                
                state_throttle = next_state_throttle
                
                state_bar = next_state_bar
                                
                rewards_throttle.append(reward_throttle)
                
                ### Train the model only if the mode is on training
                if self.mode:
                    # Check if it's time to update the neural network
                    if (update_time % self.update_freq == 0):

                        for _ in range(1, self.num_epochs+1):

                            batch_states_throttle, batch_states_bar, batch_next_state_throttle, batch_actions_throttle, batch_actions_throttle_norm, batch_prev_actions, batch_prev_actions_norm, batch_actions_bar, batch_actions_bar_norm, batch_rewards_throttle, batch_path_completed, batches = self.Memory.generate_batches()
                            for batch in batches:
                                # Perform training for the collected trajectory data using PPO
                                self.train_step(batch_states_throttle[batch], batch_states_bar[batch], batch_next_state_throttle[batch], batch_actions_throttle[batch], batch_actions_throttle_norm[batch], batch_prev_actions[batch], batch_prev_actions_norm[batch], batch_actions_bar[batch], batch_actions_bar_norm[batch], batch_rewards_throttle[batch], batch_path_completed[batch])
                                #print(f"------------ Epoch {i}/{num_epochs} ------------")
                                #print()

                        # Clear the collected batch data for the next update
                        self.Memory.clear_memory()
                        update_time = 0

                        self.old_actor_throttle.set_weights(self.actor_throttle.get_weights())
                        self.old_critic_throttle.set_weights(self.critic_throttle.get_weights())
                
                if (path_completed):
                    break
                    
            # Logging progress
            clear_output()
            # Create a figure with three subplots in a single row
            fig, axs = plt.subplots(2, 3, figsize=(10, 5))

            # Plot current position and path
            axs[0][0].plot(*zip(*current_pos_all))
            axs[0][0].plot(*zip(*path), color='g')
            axs[0][0].set_title('Current Position and Path')

            # Plot reference speed
            axs[0][1].plot(np.degrees(steering_all))
            axs[0][1].set_title('Steering')

            # Plot current speed
            axs[0][2].plot(spd_all)
            axs[0][2].set_title('Current Speed')
            
            # Plot Kp, Ki, and Kd
            axs[1][0].plot(kp_all)
            axs[1][0].set_title('Kp')
            
            axs[1][1].plot(Beta_tot)
            axs[1][1].set_title('Beta')
            
            axs[1][2].plot(kd_all)
            axs[1][2].set_title('Kd')
            

            # Adjust layout to prevent overlapping
            plt.tight_layout()

            # Show the plot
            plt.show()

            
            print(f"Iteration number {num_ittr + 1}")
            print(f"--- {(time.time() - start_time):0.2f} seconds for this episode ---" )
            print(f"Episode {episode}/{num_episodes} ({np.divide(episode, num_episodes) * 100:0.2f}% of the Training Process)")
            print(f"The total number of steps for this episode: {episode_len}")
            print(f"Successful episode for {initial_sine_mag} count is: {episode_count_heading}")
            print(f"The current initial heading is: {np.degrees(initial_heading)}")
            print(f"The last position of the vehicle: {current_position}")

            avg_reward_throttle.append(np.mean(rewards_throttle))
            
            if np.mean(rewards_throttle) > self.best_avg_reward_throttle:
                self.best_avg_reward_throttle = np.mean(rewards_throttle)
                #self.best_actor_throttle_weights = self.actor_throttle.get_weights()
            
            print(f"Average Reward (Throttle) of this episode:{avg_reward_throttle[episode-1]}")
            print(f"Current Best Average Reward (Throttle) is {self.best_avg_reward_throttle}")
            print(f"Updated learning rate: {self.learning_rate:.6f}")
            

            # After each episode, update the learning rate
            #if (episode >= 1000) and (self.learning_rate >= 5e-5):  # Only update after the first 100 episodes or if lr is high enough
            #    self.update_learning_rate()
            #    self.update_std()

            #if (avg_reward_throttle[episode-1] >= 9) : #Only when we have made sure, we can change the parameters
                                              #Why 80? Because it must have the right position error at all times
            #    Learning_made_sure = True
            #    episode_count_heading += 1  # Increment episode count for current_heading

            #with summary_writer.as_default():
            #    tf.summary.scalar("Average Reward", float(avg_reward[episode-1]), step=episode)

        return avg_reward_throttle

# PID Controller Class (Two PID controllers for both position and angle error)
class PID_Controller():
    def __init__(self, kp_position, ki_position, kd_position, max_integral=1.0):


        self.kp_position = kp_position
        self.ki_position = ki_position
        self.kd_position = kd_position

        self.max_integral = max_integral  # Maximum allowable integral term
        
        self.prev_angle_error = 0
        self.prev_position_error = 0
        self.angle_integral = 0
        self.position_integral = 0


    def calculate_steering(self, position_error, angle_error):
        # Calculate PID control outputs for position and angle errors

        # Position control
        self.position_integral += position_error
        self.position_integral = np.clip(self.position_integral, -self.max_integral, self.max_integral)  # Anti-windup
        position_derivative = position_error - self.prev_position_error
        position_output = self.kp_position * position_error + self.ki_position * self.position_integral + self.kd_position * position_derivative
        self.prev_position_error = position_error

        # Angle control
        self.angle_integral += angle_error
        self.angle_integral = np.clip(self.angle_integral, -self.max_integral, self.max_integral)  # Anti-windup
        angle_derivative = angle_error - self.prev_angle_error
        angle_output = self.kp_position * angle_error + self.ki_position * self.angle_integral + self.kd_position * angle_derivative
        self.prev_angle_error = angle_error

        # Combine the PID outputs with dynamic weights
        combined_output = 0.2 * position_output + 0.8 * angle_output

        # Limit the combined output to the range [-40, 40] degrees (convert to radians)
        max_steering_angle = max_steer
        combined_output = np.clip(combined_output, -max_steering_angle, max_steering_angle)

        return combined_output

    
class PID_Controller_Throttle():
    def __init__(self, kp_speed, ki_speed, kd_speed, max_integral=1.0):
        self.kp_speed = kp_speed
        self.ki_speed = ki_speed
        self.kd_speed = kd_speed
        self.max_integral = max_integral  # Maximum allowable integral term

        self.prev_position_error = 0
        self.position_integral = 0

    def calculate_throttle(self, position_error):
        # Calculate PID control outputs for position and angle errors
        self.position_integral += position_error
        self.position_integral = np.clip(self.position_integral, -self.max_integral, self.max_integral)  # Anti-windup
        position_derivative = position_error - self.prev_position_error
        position_output = self.kp_speed * position_error + self.ki_speed * self.position_integral + self.kd_speed * position_derivative
        self.prev_position_error = position_error

        # Combine the PID outputs with dynamic weights
        combined_output = position_output 

        # Limit the combined output to the range [-1, 1]
        max_throttle = 1.0
        combined_output = np.clip(combined_output, -max_throttle, max_throttle)

        return combined_output
    
def find_lookahead_point(path, path_angles, current_position):

    lookahead_point_index = find_closest_point_index(current_position, path)
    lookahead_point = path[lookahead_point_index]
    lookahead_heading = path_angles[lookahead_point_index]

    return np.array(lookahead_point), lookahead_heading


def NthPointAhead(path, current_position, n):
    lookahead_index = find_closest_point_index(current_position, path)
    path_angles = calculate_path_angles(path)
    path_length = len(path)

    if lookahead_index + n < path_length:
        nth_point = path[lookahead_index:lookahead_index + n]
        nth_point_heading = path_angles[lookahead_index:lookahead_index + n]
    else:
        # Calculate the number of points needed to fill up to n (in order to have a constant number of states)
        remaining_points = n - (path_length - lookahead_index)  # Adjusted for the lookahead point
        # Repeat the last point and its heading to fill up to n
        last_point = path[-1] 
        last_heading = path_angles[-1]
        
        last_point = np.repeat([last_point], remaining_points, axis=0)  # Repeat the entire point along axis 0
        last_heading = np.repeat(last_heading, remaining_points)
        
        # Concatenate the remaining points with the repeated last point
        nth_point = np.concatenate((path[lookahead_index:], last_point))
        nth_point_heading = np.concatenate((path_angles[lookahead_index:], last_heading))

    return np.array(nth_point), np.array(nth_point_heading)




# Function to generate different types of paths based on user input
#def generate_path(rc):

#    num_points = 100
#    x_vals = np.arange(0, num_points, 0.1)
#    y_vals = [-25+math.cos(ix_vals / rc) * 25 for ix_vals in x_vals]
#    path = list(zip(x_vals, y_vals))

#    return path

def generate_path(sine_mag):

    num_points = 1000
    x_vals = np.linspace(0, 20, num_points)
    y_vals = sine_mag * np.sin(x_vals*0.3) + sine_mag
    path = list(zip(x_vals, y_vals))

    return path


# Function to update the plot in each animation frame
def update(current_position, lookahead_point, current_heading, predict_trajectory, pred_vs_future_dist, pos_error, ang_error, pid_steering_angle, frame, path, path_angles, checkpoints, checkpoint_reached, Ref_Speed=None, Speed_x=None, DRL_steering_angle=None, reward=None, Throttle=None, points_ahead=None, Drastic_path_change=None, Drastic_speed_change_needed=None):

    # Calculate the arrow coordinates
    arrow_dx = 0.5 * np.cos(current_heading)
    arrow_dy = 0.5 * np.sin(current_heading)

    # Calculate the steering angle in degrees for display
    pid_steering_angle_degrees = np.degrees(pid_steering_angle)

    # Clear the previous plot and draw the updated plot
    plt.figure(figsize=(10, 5))
    plt.cla()
    plt.plot(*zip(*path), color='g', label='Path')

    plt.plot(current_position[0], current_position[1], marker='o', color='r', label='Vehicle')

    # Plot the vehicle direction arrow
    plt.quiver(current_position[0], current_position[1], arrow_dx, arrow_dy, angles='xy', scale_units='xy',
              scale=1, width=0.005, headwidth=0.2, headlength=0.2, color='r')

    # Plot the start and end points of the reference path
    plt.plot(path[0][0], path[0][1], marker='o', color='g', label='Start')
    plt.plot(path[-1][0], path[-1][1], marker='x', color='g', label='End')

    plt.axis('equal')
    plt.xlabel('X')
    plt.ylabel('Y')

    # Plot the line from the current position to the closest point on the path with a label for the legend
    plt.plot([current_position[0], lookahead_point[0]], [current_position[1], lookahead_point[1]], 'b--', label='Position Error')

    # Plot the closest point on the path as a red dot
    plt.plot(lookahead_point[0], lookahead_point[1], marker='o', color='cyan', label='Closest Point')

    
    # Plot checkpoints
    for checkpoint in checkpoints:
        plt.plot(checkpoint[0], checkpoint[1], marker='o', markersize=5, color='b')

        
    # Plot the next n points ahead
    if points_ahead is not None:
        points_ahead_x, points_ahead_y = points_ahead[:, 0], points_ahead[:, 1]
        plt.scatter(points_ahead_x, points_ahead_y, color='orange', label='Points Ahead')

    predict_trajectory_x, predict_trajectory_y = predict_trajectory[:, 0], predict_trajectory[:, 1]
    plt.scatter(predict_trajectory_x, predict_trajectory_y, color='yellow', label='predict trajectory')
        
    

    plt.legend()

    
    # Display the steering angle and throttle
    if DRL_steering_angle is not None:
        plt.text(current_position[0] + 0.4, current_position[1] + 0.2, f'PID+DRL: {pid_steering_angle_degrees.item():.2f} + {DRL_steering_angle:.2f} [deg]', fontsize=10)
    else:
        plt.text(current_position[0] + 0.4, current_position[1] + 0.2, f'PID: {pid_steering_angle_degrees.item():.2f} [deg]', fontsize=10)

    if Ref_Speed is not None:
        plt.text(current_position[0] + 0.4, current_position[1] - 0.5, f'Ref. Speed: {Ref_Speed:.2f}', fontsize=10)

    if Speed_x is not None:
        plt.text(current_position[0] + 0.4, current_position[1] - 1 , f'Speed: {Speed_x:.2f} [unit/s]', fontsize=10)

    plt.text(current_position[0] + 0.4, current_position[1] - 1.5 , f'position error: {pos_error:.2f} [units]', fontsize=10)

    plt.text(current_position[0] + 0.4, current_position[1] - 2 , f'angle error: {ang_error:.2f} [deg]', fontsize=10)

    if reward is not None:
        plt.text(current_position[0] + 0.4, current_position[1] - 2.5, f'Reward: {reward:.2f}', fontsize=10)

    if Throttle is not None:
        plt.text(current_position[0] + 0.4, current_position[1] - 3, f'Throttle: {Throttle:.2f}', fontsize=10)
    
    if (Drastic_path_change and Drastic_speed_change_needed) is not None:
        plt.text(current_position[0] + 0.4, current_position[1] - 3.5, f'Drastic_path_change: {Drastic_path_change}', fontsize=10)
        plt.text(current_position[0] + 0.4, current_position[1] - 4.0, f'Drastic_speed_change_needed: {Drastic_speed_change_needed}', fontsize=10)
        
    
    plt.text(current_position[0] + 0.4, current_position[1] - 4.0, f'pred_vs_future_dist: {pred_vs_future_dist}', fontsize=10)

    
    plt.text(current_position[0] + 0.4, current_position[1] - 4.5, f'checkpoint_reached: {checkpoint_reached}', fontsize=10)
    

    # Display the updated plot in Jupyter Notebook using display() and clear_output()
    display(plt.gcf())
    clear_output(wait=True)


def PathIsComplete(path, current_position, lookahead_point, checkpoints):
    
    # Check if the path is completed (reached the end)
    #If we just check for the checkpoints to end the path, the vehicle might finish the path sooner than when it actually ends
    distance_to_ending_point = np.linalg.norm(lookahead_point - path[-1])
    distance_to_current_position = np.linalg.norm(current_position - path[-1])
    path_completed = (all(checkpoints)) and ((distance_to_ending_point <= 0.1) or distance_to_current_position <= 0.1)
    
    return path_completed



def calculate_position_error(path, current_position, tolerance):
    # Find the nearest point on the path
    path_points = np.array(path)
    nearest_point_index = np.argmin(np.linalg.norm(path_points - current_position, axis=1))
    nearest_point = path_points[nearest_point_index]

    # Calculate the vector from the current position to the nearest point on the path
    to_nearest_point = nearest_point - current_position

    # Calculate the vector representing the direction of the path
    path_direction = path_points[-1] - path_points[0]

    # Calculate the cross product between the vectors
    cross_product = np.cross(to_nearest_point, path_direction)

    # Calculate the signed distance between the current position and the nearest point on the path
    min_distance = np.linalg.norm(to_nearest_point)

    if np.greater_equal(cross_product, 0).all():
        intended_error = -min_distance  # Vehicle is on the same side as the direction of the path
    else:
        intended_error = min_distance   # Vehicle is on the opposite side of the direction of the path

    # Adjust the intended error based on the tolerance
    if intended_error >= tolerance:
        return intended_error - tolerance
    elif intended_error <= -tolerance:
        return intended_error + tolerance
    else:
        return 0.0  # Within the desired tube
    
        

# Function to calculate path angles
def calculate_path_angles(path):
    path_angles = []
    for i in range(len(path)):
        if i == 0:
            # For the first point, use the angle between the first and second points
            dx = path[1][0] - path[0][0]
            dy = path[1][1] - path[0][1]
        else:
            dx = path[i][0] - path[i-1][0]
            dy = path[i][1] - path[i-1][1]

        angle = np.arctan2(dy, dx)
        path_angles.append(angle)
        
    return path_angles

# Function to find the index of the closest point on the path to the lookahead point
def find_closest_point_index(current_position, path):
    distances = np.linalg.norm(np.array(path) - np.array(current_position), axis=1)
    return np.argmin(distances)


# Modified angle error calculation function
def calculate_angle_error(current_position, current_heading, path_angles, path, tolerance):
    # Find the index of the closest point on the path to the lookahead point
    lookahead_index = find_closest_point_index(current_position, path)

    # Calculate the angle between the direction vector and the current heading
    angle_error = path_angles[lookahead_index] - current_heading  # Use path angle

    # Wrap the angle error to be between -pi and pi
    angle_error = (angle_error + np.pi) % (2 * np.pi) - np.pi

    # Wrap the angle error to be between -2pi and 2pi
    #angle_error = (angle_error + 2*np.pi) % (4*np.pi) - 2*np.pi

    # Ensure that the angle error is within the specified tolerance
    if np.abs(angle_error) <= tolerance:
        return 0.0
    else:
        return angle_error - np.sign(angle_error) * tolerance
    

def predict_trajectory_linear(path, current_position, current_heading, num_steps):
    # Calculate the average distance between consecutive points in the path
    path_distances = np.linalg.norm(np.diff(path, axis=0), axis=1)
    avg_distance = np.mean(path_distances)

    # Calculate velocity magnitude based on the average distance
    velocity_magnitude = avg_distance

    # Define velocity vector based on current heading and adjusted magnitude
    velocity = velocity_magnitude * np.array([np.cos(current_heading), np.sin(current_heading)])

    # Initialize trajectory with current position
    trajectory = [current_position]

    # Extrapolate future positions based on velocity
    for _ in range(num_steps - 1):  # Adjusted for including the current position (and lookahead)
        next_position = trajectory[-1] + velocity
        trajectory.append(next_position)

    return np.array(trajectory)


def compute_gae(values, next_values, rewards, completes, gamma, lambda_):
    gae = 0
    adv = []
    
    delta = rewards + gamma * next_values * (1-completes) - values
        
    for step in reversed(range(len(rewards))):
        gae = delta[step] + gamma * lambda_ * (1-completes[step]) * gae
        adv.insert(0, gae)
        
    return tf.stack(adv)


# non-linear lateral bicycle model
class NonLinearBicycleModel():
    def __init__(self, x=0.0, y=0.0, yaw=0.0, vx=0.0, vy=0.0, Beta=0.0, omega=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.vx = vx
        self.vy = vy
        self.Beta = Beta
        self.omega = omega
        # Aerodynamic and friction coefficients
        self.c_a = 1.36
        self.c_r1 = 0.01

    def update(self, throttle, delta):
        
        self.x = self.x + self.vx * math.cos(self.yaw) * dt - self.vy * math.sin(self.yaw) * dt
        self.y = self.y + self.vx * math.sin(self.yaw) * dt + self.vy * math.cos(self.yaw) * dt
        self.yaw = self.yaw + self.omega * dt
        self.yaw = normalize_angle(self.yaw)
        
        alpha_f = math.atan2((self.vy + Lf * self.omega), self.vx + 1e-6) - delta
        Ffy = -Cf * alpha_f
        
        alpha_r = math.atan2((self.vy - Lr * self.omega), self.vx + 1e-6)
        Fry = -0.05*Cr * alpha_r
        
        R_x = self.c_r1 * self.vx
        F_aero = self.c_a * self.vx ** 2
        F_load = F_aero + R_x

        self.vx = self.vx + (throttle - Ffy * math.sin(delta) / m - F_load/m + (self.vy+1e-6) * self.omega) * dt
        self.vx = max(1e-3, self.vx) #ensure tha the speed is always positive
        self.vx = min(self.vx, 4.0) #less than 4
        self.vy = self.vy + (Fry / m + Ffy * math.cos(delta) / m - (self.vx+1e-6) * self.omega) * dt
        #self.vy = max(0, self.vy) #ensure tha the speed is always positive
        #self.vy = min(self.vy, 4.0) #less than 4
        
        self.Beta = math.atan2((self.vy+1e-6), (self.vx+1e-6))
        
        # Calculate total speed (Euclidean norm)
        #self.speed = np.sqrt(self.vx**2 + self.vy**2)
        #self.speed = min(self.speed, 4.0) #ensure that the maximum speed is under 4.0
        
        self.omega = self.omega + (Ffy * Lf * math.cos(delta) - Fry * Lr) / Iz * dt
        
        return self.vx, self.vy, alpha_f, alpha_r, self.Beta, self.omega, Ffy, Fry
    

def normalize_angle(angle):
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle


#The function is responsible for finding the closest checkpoints
class CheckpointsTracker:
    def __init__(self, path, checkpoints, threshold = 0.2):
        self.path = path
        self.checkpoints = checkpoints
        self.threshold = threshold
        self.num_checkpoints = len(checkpoints)
        self.checkpoint_reached = [False] * self.num_checkpoints

    def update(self, current_position):
        # Find the index of the closest checkpoint to the current position
        closest_checkpoint_index = self.find_closest_checkpoint_index(current_position)

        # Check if the current position is within the threshold distance of the closest checkpoint
        distance_to_closest_checkpoint = np.linalg.norm(np.array(current_position) - np.array(self.checkpoints[closest_checkpoint_index]))
        if distance_to_closest_checkpoint <= self.threshold:
            # Set the flag of the closest checkpoint that has been reached to True
            self.checkpoint_reached[closest_checkpoint_index] = True

    def find_closest_checkpoint_index(self, current_position):
        distances = np.linalg.norm(np.array(self.checkpoints) - np.array(current_position), axis=1)
        return np.argmin(distances)

    
    

# Main loop to follow the path and visualize the results
# Start following the path
if __name__ == "__main__":
    
    global avg_rew
    
    ############################ Hyperparameters ############################
    state_dim_throttle = 14 + 2*10 + 2 + 2
    action_dim = 4 #Ref. Speed, Kp, Ki, and Kd
    learning_rate = 1e-4
    gamma = 0.90
    lambda_ = 0.95
    batch_size = 64
    best_avg_reward_throttle = -float('inf')
    decay_rate = 0.999
    entropy_coefficient_throttle = 5e-2
    policy_kl_range = 0.03
    policy_params = 5
    value_clip = 1.0
    update_freq = 1024
    num_epochs = 30
    
    lam_T = 1e-4
    lam_S = 5e-4
    lam_V = 1e-10
    
    lamda_k = 1e-5
    K_lr = 5e-5
    
    decay_rate_std = 0.99
    
    ########################### ############################
    
    #(Training:) In the beginning, mode=True (Train the model), save_model=True(Saved the trained model), and load_model=False 
                #mode = True
                #save_model = True
                #load_model = False
    #(Inference:) Then, mode=False (No training), load_model=True and save_model=False (Use the loaded model for inference)
                #mode = False
                #save_model = False
                #load_model = True
            
    mode = False #(True = Training, False = Inference)
    save_model = False
    load_model = True
    
    num_episodes = 5000
    
    
    num_ittr = 5
    avg_rew = np.zeros((num_ittr, num_episodes))

    #for itt in range(num_ittr):
    # Create an instance of the DRLController class
    #for itt in range(num_ittr):
    drl_controller = DRLController(state_dim_throttle, action_dim, learning_rate, gamma, lambda_, batch_size,
                                   best_avg_reward_throttle, decay_rate, entropy_coefficient_throttle, policy_kl_range,
                                   policy_params, value_clip, update_freq, num_epochs, lam_T, lam_S, lam_V,
                                   lamda_k, K_lr, decay_rate_std, mode)

    if load_model:
        drl_controller.load_weights()
        print('Model Loaded')

    # Train the DRL controller using the instance's method
    #avg_rew[itt, :] = drl_controller.train_ppo_agent(num_episodes, dt, itt)
    avg_rew = drl_controller.train_ppo_agent(num_episodes, dt, 0)

    if save_model:
        drl_controller.save_weights()
        print('Model Saved')

