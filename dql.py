"""responsible for training the model"""

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization
import numpy as np
from environment import ElevatorEnv
from dqn import DeepQNetwork,ReplayBuffer
from globals_vars import *
from tabulate import tabulate
from logger.logger import log


class DeepQAgent:
    """Responsible for training agent using deep q learning """

    def __init__(self, nf:int, sf:list, sp:list, ef:list, ep:list, 
                 floors:list, ts:int, lr:float, df:float, exp:float, ar:float, 
                 itrs:int, buff_cap:int, bs:int, decay:float, min_exp:float, verbose):
        
        self.verbose = verbose

        # Simulator vars
        self.num_floors = nf
        self.floors = floors
        self.timestep = ts
        self.max_capacity = 2
        """Max number of people in an elevator"""
        self.start_floors = sf
        """Start floors EX: [1]"""
        self.start_probs = sp
        """Probability of landing on start floor. EX: [.2,.2,.2,.2,.2]"""
        self.exit_floors = ef
        """Exit floors EX: [2,3,4,5,6]"""
        self.exit_probs = ep
        """Probability of landing on exit floor. EX: [.2,.2,.2,.2,.2]"""

        # Learner vars
        self.learning_rate = lr
        self.discount_factor = df
        self.epsilon = exp
        """Exploration rate"""
        self.decay = decay
        """Amount which exploration rate decays"""
        self.min_epsilon = min_exp
        """Minimum exploration rate"""
        self.bs = bs
        """Batch size"""
        self.arrival_rate = ar
        """Arrival rate of people in simulator per second"""
        self.buff_cap = buff_cap
        """Maximum size of the replay buffer"""

        # Agent vars
        self.iterations = itrs
        self.avg_rewards = []
        """Average rewards per iteration"""
        self.exact_rewards = []
        """Exact reward per iteration"""

        self.env =  ElevatorEnv(self.num_floors, self.max_capacity, self.start_floors, 
                          self.start_probs, self.exit_floors, self.exit_probs, 
                          self.timestep, self.arrival_rate)
        """Elevator class"""
        log("Num GPUs Available: " + str(len(tf.config.list_physical_devices('GPU'))))


    def train_model(self,model, target_model, replay_buffer):
        """Trains the deep q network using replay and target network

        Args:
            model (DeepQNetwork): training dqn
            target_model (DeepQNetwork): target dqn
            replay_buffer (ReplayBuffer): replay buffer
        """

        # Return if replay is greater than the size of the batch
        if replay_buffer.size() < self.bs:
            return
        
        # Get specified number of samples, and train model using them
        samples = replay_buffer.sample(self.bs)
        for state, action, reward, next_state in samples:
            with tf.device('/device:GPU:0'):

                # Get current predictions of model
                q_target = model.predict(state)

                # Get predictions of next state using target network
                td_targets = target_model.predict(next_state)

                # Replace old q value with new q value from  the td target
                q_target[0][action] = reward + self.discount_factor * np.max(td_targets)

                if self.verbose > 1:
                    actions = [*range(0,ACTION_SIZE)]
                    log('\n' + tabulate([actions, q_target[0]], tablefmt="grid"))
                    log("q[action] = " + str(q_target[0][action]))
                    
                # Fit main network using new target q values
                model.fit(state, q_target, epochs=1, verbose=1)

    def update_target_model(self, model, target_model):
        """Updates target model

        Args:
            model (DeepQNetwork): dqn model
            target_model (DeepQNetwork): target dqn model
        """
        target_model.set_weights(model.get_weights())

    def policy(self, state, model, epsilon):
        """Chooses max action or random action given the epsilon/exploration rate

        Args:
            state (ndarray): state vector   
            model (DeepQNetwork): dqn
            epsilon (float): exploration rate

        Returns:
            int: int range(0,15). One hot coded action
        """

        if np.random.rand() <= epsilon:
            return np.random.randint(0, ACTION_SIZE)
        else:
            Q_values = model.predict(state.reshape(1, -1))
            return np.argmax(Q_values[0])

    def train_agent(self):
        """Trains agent using deep q learning"""

        # Init dqn, target dqn, and replay buffer
        dqn = DeepQNetwork(self.learning_rate)
        target_dqn = DeepQNetwork(self.learning_rate)
        replay_buffer = ReplayBuffer(capacity=self.buff_cap)

        # inital state
        state = self.env.init_state()
        state = np.reshape(state, [1, STATE_SIZE])
        total_reward = 0

        # Until simulation ends ...
        for iteration in range(1, self.iterations+1):

            # Get action accordning to policy
            action = self.policy(state, dqn.model, self.epsilon)

            # Take action, observe reward, next state, and update replay buffer
            next_state, reward = self.env.step(action)
            next_state = np.reshape(next_state, [1, STATE_SIZE])
            replay_buffer.add(state, action, reward, next_state)

            # Update tracking variables
            state = next_state
            total_reward += reward
            self.exact_rewards.append(reward)
            self.avg_rewards.append(sum(self.exact_rewards) / (iteration+1))  

            # Train model using replay buffer samples
            print("Epsilon: ", self.epsilon)
            self.env.print_state_info(action, reward, self.verbose) 
            self.train_model(dqn.model, target_dqn.model, replay_buffer)

            # Update the target dqn every c number of iterations
            if iteration % 10 == 0:
                self.update_target_model(dqn.model, target_dqn.model)

            # Apply epsilon decay
            self.epsilon = max(self.min_epsilon, self.epsilon * self.decay)

        # Remove bias
        self.avg_rewards.pop(0)
        self.exact_rewards.pop(0)
       


