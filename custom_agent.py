from datetime import datetime
from tensorflow.keras.optimizers import Adam, RMSprop
from model import Actor_Model, Critic_Model, Shared_Model
from tensorboardX import SummaryWriter
import copy
import os
import numpy as np
from icecream import ic
import enum
import tensorflow as tf
from keras import layers


class Spaces(enum.Enum):
    out_of_position = 0
    in_position = 1
    # Size of Possible State


class Acts(enum.Enum):
    remain_out_of_position = 0
    enter_long = 1
    exit_long = 2
    remain_in_position = 3


class custom_agent:
    # A custom Bitcoin trading agent
    def __init__(self, lookback_window_size=50, learning_rate=0.00005, epochs=1, optimizer=Adam, batch_size=32, model="", state_size=10):
        self.lookback_window_size = lookback_window_size
        self.model = model  # Currently, only one model is available (CNN)
        self.space = Spaces.out_of_position
        self.next_space = "Mamad"  
        # Action space from 0 to 3
        self.action_space = np.array([Acts.remain_out_of_position,
                                      Acts.enter_long,
                                      Acts.exit_long,
                                      Acts.remain_in_position]
                                     )

        # folder to save models
        self.log_name = datetime.now().strftime("%Y_%m_%d_%H_%M")+"_Crypto_trader"
        ic(self.log_name)

        # State size contains Market+Orders history for the last lookback_window_size steps
        # 10 standard information +9 indicators
        self.state_size = (lookback_window_size, state_size)
        self.space_size = len(Acts)

        # Neural Networks part bellow
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.optimizer = optimizer
        self.batch_size = batch_size



        # Create shared Actor-Critic network model
        self.Actor = self.Critic = Shared_Model(
            input_shape=self.state_size, valid_act=self.space_size, action_space=self.action_space.shape[
                0], learning_rate=self.learning_rate,
            optimizer=self.optimizer, model=self.model)

    def get_onehot_mask(self):
        if self.space == Spaces.out_of_position:
            return [1, 1, 0, 0] #remain_out_of_position, enter_long
        elif self.space == Spaces.in_position:
            return [0, 0, 1, 1] #exit_long, remain_in_position
        else:
            raise Exception('ERROR!')

    # create tensorboard writer

    def calc_next_space(self, action):
        if self.space == Spaces.out_of_position:
            if action == Acts.enter_long:
                self.next_space = Spaces.in_position
            elif action == Acts.remain_out_of_position:
                self.next_space = Spaces.out_of_position
            else:
                raise Exception('ERROR!')
        elif self.space == Spaces.in_position:
            if action == Acts.exit_long:
                self.next_space = Spaces.out_of_position
            elif action == Acts.remain_in_position:
                self.next_space = Spaces.in_position
            else:
                raise Exception('ERROR!')

    def create_writer(self, initial_balance, normalize_value, train_episodes):
        self.replay_count = 0
        self.writer = SummaryWriter('runs/'+self.log_name)

        # Create folder to save models
        if not os.path.exists(self.log_name):
            os.makedirs(self.log_name)
            # global save_folder = self.log_name

        self.start_training_log(
            initial_balance, normalize_value, train_episodes)

    def start_training_log(self, initial_balance, normalize_value, train_episodes):
        # save training parameters to Parameters.txt file for future
        with open(self.log_name+"/Parameters.txt", "w") as params:
            current_date = datetime.now().strftime('%Y-%m-%d %H:%M')
            params.write(f"training start: {current_date}\n")
            params.write(f"initial_balance: {initial_balance}\n")
            params.write(f"training episodes: {train_episodes}\n")
            params.write(
                f"lookback_window_size: {self.lookback_window_size}\n")
            params.write(f"learning_rate: {self.learning_rate}\n")
            params.write(f"epochs: {self.epochs}\n")
            params.write(f"batch size: {self.batch_size}\n")
            params.write(f"normalize_value: {normalize_value}\n")
            params.write(f"model: {self.model}\n")

    def end_training_log(self):
        with open(self.log_name+"/Parameters.txt", "a+") as params:
            current_date = datetime.now().strftime('%Y-%m-%d %H:%M')
            params.write(f"training end: {current_date}\n")

    # Calculating GAEs
    def get_gaes(self, rewards, dones, values, next_values, gamma=0.99, lamda=0.95, normalize=True):
        deltas = [r + gamma * (1 - d) * nv - v for r, d,
                  nv, v in zip(rewards, dones, next_values, values)]
        deltas = np.stack(deltas)
        gaes = copy.deepcopy(deltas)
        for t in reversed(range(len(deltas) - 1)):
            gaes[t] = gaes[t] + (1 - dones[t]) * gamma * lamda * gaes[t + 1]

        target = gaes + values
        if normalize:
            gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)
        return np.vstack(gaes), np.vstack(target)

    def replay(self, states, spaces, actions, rewards, predictions, dones, next_states):
        # reshape memory to appropriate shape for training
        states = np.vstack(states)
        spaces = np.vstack(spaces)
        next_states = np.vstack(next_states)
        actions = np.vstack(actions)
        predictions = np.vstack(predictions)

        # Get Critic network predictions
        values = self.Critic.critic_predict(states)
        next_values = self.Critic.critic_predict(next_states)

        # Compute advantages
        advantages, target = self.get_gaes(
            rewards, dones, np.squeeze(values), np.squeeze(next_values))

        # stack everything to numpy array
        y_true = np.hstack([advantages, predictions, actions])

        # training Actor and Critic networks
        a_loss = self.Actor.Actor.fit(
            [states, spaces], y_true, epochs=self.epochs, verbose=0, shuffle=True, batch_size=self.batch_size)

        c_loss = self.Critic.Critic.fit(
            states, target, epochs=self.epochs, verbose=0, shuffle=True, batch_size=self.batch_size)

        self.writer.add_scalar('Data/actor_loss_per_replay',
                               np.sum(a_loss.history['loss']), self.replay_count)
        self.writer.add_scalar('Data/critic_loss_per_replay',
                               np.sum(c_loss.history['loss']), self.replay_count)
        self.replay_count += 1

        return np.sum(a_loss.history['loss']), np.sum(c_loss.history['loss'])

    def act(self, state):
        # Use the network to predict the upcoming action
        # TODO: Adding Exploration/Exploitation
        prediction = self.Actor.actor_predict(np.expand_dims(state, axis=0),
                                              np.expand_dims(self.get_onehot_mask(), axis=0))[0]

        action = np.random.choice(self.action_space, p=prediction)

        self.calc_next_space(action)

        return action, prediction

    # Saving the weights after finding a better configuration
    def save(self, name="Crypto_trader", score="", args=[]):
        # save keras model weights
        self.file_name = f"{self.log_name}/{score}_{name}"
        self.Actor.Actor.save_weights(
            f"{self.log_name}/{score}_{name}_Actor.h5")
        self.Critic.Critic.save_weights(
            f"{self.log_name}/{score}_{name}_Critic.h5")

        # log saved model arguments to file
        if len(args) > 0:
            with open(f"{self.log_name}/log.txt", "a+") as log:
                current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                atgumets = ""
                for arg in args:
                    atgumets += f", {arg}"
                log.write(f"{current_time}{atgumets}\n")

    # Loading the weights for testing the model
    def load(self, folder, name):
        # load keras model weights
        self.Actor.Actor.load_weights(os.path.join(folder, f"{name}_Actor.h5"))
        self.Critic.Critic.load_weights(
            os.path.join(folder, f"{name}_Critic.h5"))
