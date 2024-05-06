"""modules for building dqn"""

from globals_vars import *
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization
from collections import deque
import random

class ReplayBuffer:
    """Stores samples of training experiences"""

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state):
        """Adds sample to replay buffer"""
        self.buffer.append((state, action, reward, next_state))
    
    def sample(self, batch_size):
        """Uses a sample from a replay buffer"""
        return random.sample(self.buffer, batch_size)
    
    def size(self):
        """Returns size of replay buffer"""
        return len(self.buffer)
    
class DeepQNetwork:
    """Deep Neural Network to be used for elevator model training"""

    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
        self.model = self.create_model()

    def create_model(self):
        """Creates Deep NN model

        Returns:
            DeepQNetwork model: dqn model
        """
        with tf.device('/device:GPU:0'):
            model = tf.keras.Sequential([
                Dense(128, activation='relu', input_shape=(STATE_SIZE,)),
                Dense(64, activation='relu'),
                Dense(32, activation='relu'),
                Dense(ACTION_SIZE, activation='linear')
            ])
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), loss='mse')
            return model
        
    
    

