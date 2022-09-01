import numpy as np
import gym
import gc
import os
import gym_environments
import random
import criticPPO as critic
import actorPPO as actor
import tensorflow as tf
from collections import deque
#import time as tt
import argparse
import pickle
from keras import backend as K

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

ENV_NAME = 'GraphEnv-v1'
graph_topology = 0 # 0==NSFNET, 1==GEANT2, 2==Small Topology, 3==GBN
EVALUATION_EPISODES = 50
SEED = 37
BUFF_SIZE = 600 # Experience buffer size
PPO_EPOCHS = 7 # 15
TRAIN_BATCH_SIZE = 32

CRITIC_DISCOUNT = 0.8
ENTROPY_BETA = 0.01

clipping_val = 0.2
gamma = 0.99
lmbda = 0.95

differentiation_str = "PPO_NSFNet_agent"
checkpoint_dir = "./models"+differentiation_str

os.environ['PYTHONHASHSEED']=str(SEED)
np.random.seed(SEED)
random.seed(SEED)

tf.random.set_seed(1)

train_dir = "./TensorBoard/"+differentiation_str
summary_writer = tf.summary.create_file_writer(train_dir)
global_step = 0
listofDemands = [8, 32, 64]
n_actions = [0, 1, 2, 3]

hparams = {
    'l2': 0.1,
    #'dropout_rate': 0.05,
    'link_state_dim': 20,
    'readout_units': 35,
    'learning_rate': 0.0001,
    'T': 4, 
    'num_demands': len(listofDemands)
}

MAX_QUEUE_SIZE = 3000

def old_cummax(alist, extractor):
    with tf.name_scope('cummax'):
        maxes = [tf.reduce_max(extractor(v)) + 1 for v in alist]
        cummaxes = [tf.zeros_like(maxes[0])]
        for i in range(len(maxes) - 1):
            cummaxes.append(tf.math.add_n(maxes[0:i + 1]))
    return cummaxes

class PPOActorCritic:
    def __init__(self, env_training):
        self.memory = deque(maxlen=MAX_QUEUE_SIZE)
        self.K = len(n_actions)
        self.listValues = None
        self.softMaxValues = None
        self.global_step = global_step

        self.action = None
        self.capacity_feature = None
        self.bw_allocated_feature = np.zeros((env_training.numEdges,len(env_training.listofDemands)))

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=hparams['learning_rate'])
        self.actor = actor.myModel(hparams)
        self.actor.build()

        self.critic = critic.myModel(hparams)
        self.critic.build()

    def pred_action_distrib(self, env, state, demand, source, destination):
        # List of graphs
        listGraphs = []
        # List of graph features that are used in the cummax() call
        list_k_features = list()
        # Initialize action
        action = 0

        # We get the K-paths between source-destination
        pathList = env.allPaths[str(source) +':'+ str(destination)]
        path = 0
        
        # 2. Allocate (S,D, linkDemand) demand using the K shortest paths
        while path < len(pathList):
            state_copy = np.copy(state)
            currentPath = pathList[path]
            i = 0
            j = 1

            # 3. Iterate over paths' pairs of nodes and allocate demand to bw_allocated
            while (j < len(currentPath)):
                state_copy[env.edgesDict[str(currentPath[i]) + ':' + str(currentPath[j])]][1] = demand
                i = i + 1
                j = j + 1

            # 4. Add allocated graphs' features to the list. Later we will compute it's values using cummax
            listGraphs.append(state_copy)
            features = self.get_graph_features3(env, state_copy)
            list_k_features.append(features)

            path = path + 1

        vs = [v for v in list_k_features]

        # We compute the graphs_ids to later perform the unsorted_segment_sum for each graph and obtain the 
        # link hidden states for each graph.
        graph_ids = [tf.fill([tf.shape(vs[it]['link_state'])[0]], it) for it in range(len(list_k_features))]
        first_offset = old_cummax(vs, lambda v: v['first'])
        second_offset = old_cummax(vs, lambda v: v['second'])

        tensors = ({
            'graph_id': tf.concat([v for v in graph_ids], axis=0),
            'link_state': tf.concat([v['link_state'] for v in vs], axis=0),
            'first': tf.concat([v['first'] + m for v, m in zip(vs, first_offset)], axis=0),
            'second': tf.concat([v['second'] + m for v, m in zip(vs, second_offset)], axis=0),
            'num_edges': tf.math.add_n([v['num_edges'] for v in vs]),
            }
        )        

        r = self.actor(tensors['link_state'], tensors['graph_id'], tensors['first'],
                tensors['second'], tensors['num_edges'], training=False)

        self.listValues = tf.reshape(r, (1, len(r)))
        self.softMaxValues = tf.nn.softmax(self.listValues)

        # Return action distribution
        return self.softMaxValues.numpy()[0], tensors
    
    def get_graph_features3(self, env, copyGraph):
        self.bw_allocated_feature.fill(0.0)
        self.capacity_feature = (copyGraph[:,0] - 100.00000001) / 200.0

        iter = 0
        for i in copyGraph[:, 1]:
            if i == 8:
                self.bw_allocated_feature[iter][0] = 1
            if i == 32:
                self.bw_allocated_feature[iter][1] = 1
            elif i == 64:
                self.bw_allocated_feature[iter][2] = 1
            iter = iter + 1
        
        sample = {
            'num_edges': env.numEdges,  
            'length': env.firstTrueSize,
            'betweenness': tf.convert_to_tensor(value=env.between_feature, dtype=tf.float32),
            'bw_allocated': tf.convert_to_tensor(value=self.bw_allocated_feature, dtype=tf.float32),
            'capacities': tf.convert_to_tensor(value=self.capacity_feature, dtype=tf.float32),
            'first': tf.convert_to_tensor(env.first, dtype=tf.int32),
            'second': tf.convert_to_tensor(env.second, dtype=tf.int32)
        }

        sample['capacities'] = tf.reshape(sample['capacities'][0:sample['num_edges']], [sample['num_edges'], 1])
        sample['betweenness'] = tf.reshape(sample['betweenness'][0:sample['num_edges']], [sample['num_edges'], 1])

        hiddenStates = tf.concat([sample['capacities'], sample['betweenness'], sample['bw_allocated']], axis=1)

        paddings = tf.constant([[0, 0], [0, hparams['link_state_dim'] - 2 - hparams['num_demands']]])
        link_state = tf.pad(tensor=hiddenStates, paddings=paddings, mode="CONSTANT")

        inputs = {'link_state': link_state, 'first': sample['first'][0:sample['length']],
                  'second': sample['second'][0:sample['length']], 'num_edges': sample['num_edges']}

        return inputs

    def critic_get_graph_features(self, env, state):
        """
        We iterate over the converted graph nodes and take the features. The capacity and bw allocated features
        are normalized on the fly.
        """
        copyGraph = np.copy(state)
        self.capacity_feature = (copyGraph[:,0] - 100.00000001) / 200.0

        sample = {
            'num_edges': tf.convert_to_tensor(env.numEdges, dtype=tf.int32),  
            'length': tf.convert_to_tensor(env.firstTrueSize, dtype=tf.int32),
            'betweenness': tf.convert_to_tensor(value=env.between_feature, dtype=tf.float32),
            'capacities': tf.convert_to_tensor(value=self.capacity_feature, dtype=tf.float32),
            'first': tf.convert_to_tensor(env.first, dtype=tf.int32),
            'second': tf.convert_to_tensor(env.second, dtype=tf.int32)
        }

        sample['capacities'] = tf.reshape(sample['capacities'][0:sample['num_edges']], [sample['num_edges'], 1])
        sample['betweenness'] = tf.reshape(sample['betweenness'][0:sample['num_edges']], [sample['num_edges'], 1])

        hiddenStates = tf.concat([sample['capacities'], sample['betweenness']], axis=1)

        paddings = tf.constant([[0, 0], [0, hparams['link_state_dim'] - 2]])
        link_state = tf.pad(tensor=hiddenStates, paddings=paddings, mode="CONSTANT")

        inputs = {'link_state_critic': link_state, 'first_critic': sample['first'][0:sample['length']],
                  'second_critic': sample['second'][0:sample['length']], 'num_edges_critic': sample['num_edges']}

        return inputs
    
    def _write_tf_summary(self, actor_loss, critic_loss, final_entropy):
        with summary_writer.as_default():
            tf.summary.scalar(name="actor_loss", data=actor_loss, step=self.global_step)
            tf.summary.scalar(name="critic_loss", data=critic_loss, step=self.global_step)  
            tf.summary.scalar(name="entropy", data=-final_entropy, step=self.global_step)                      

            tf.summary.histogram(name='ACTOR/FirstLayer/kernel:0', data=self.actor.variables[0], step=self.global_step)
            tf.summary.histogram(name='ACTOR/FirstLayer/bias:0', data=self.actor.variables[1], step=self.global_step)
            tf.summary.histogram(name='ACTOR/kernel:0', data=self.actor.variables[2], step=self.global_step)
            tf.summary.histogram(name='ACTOR/recurrent_kernel:0', data=self.actor.variables[3], step=self.global_step)
            tf.summary.histogram(name='ACTOR/bias:0', data=self.actor.variables[4], step=self.global_step)
            tf.summary.histogram(name='ACTOR/Readout1/kernel:0', data=self.actor.variables[5], step=self.global_step)
            tf.summary.histogram(name='ACTOR/Readout1/bias:0', data=self.actor.variables[6], step=self.global_step)
            tf.summary.histogram(name='ACTOR/Readout2/kernel:0', data=self.actor.variables[7], step=self.global_step)
            tf.summary.histogram(name='ACTOR/Readout2/bias:0', data=self.actor.variables[8], step=self.global_step)
            tf.summary.histogram(name='ACTOR/Readout3/kernel:0', data=self.actor.variables[9], step=self.global_step)
            tf.summary.histogram(name='ACTOR/Readout3/bias:0', data=self.actor.variables[10], step=self.global_step)
            
            tf.summary.histogram(name='CRITIC/FirstLayer/kernel:0', data=self.critic.variables[0], step=self.global_step)
            tf.summary.histogram(name='CRITIC/FirstLayer/bias:0', data=self.critic.variables[1], step=self.global_step)
            tf.summary.histogram(name='CRITIC/kernel:0', data=self.critic.variables[2], step=self.global_step)
            tf.summary.histogram(name='CRITIC/recurrent_kernel:0', data=self.critic.variables[3], step=self.global_step)
            tf.summary.histogram(name='CRITIC/bias:0', data=self.critic.variables[4], step=self.global_step)
            tf.summary.histogram(name='CRITIC/Readout1/kernel:0', data=self.critic.variables[5], step=self.global_step)
            tf.summary.histogram(name='CRITIC/Readout1/bias:0', data=self.critic.variables[6], step=self.global_step)
            tf.summary.histogram(name='CRITIC/Readout2/kernel:0', data=self.critic.variables[7], step=self.global_step)
            tf.summary.histogram(name='CRITIC/Readout2/bias:0', data=self.critic.variables[8], step=self.global_step)
            tf.summary.histogram(name='CRITIC/Readout3/kernel:0', data=self.critic.variables[9], step=self.global_step)
            tf.summary.histogram(name='CRITIC/Readout3/bias:0', data=self.critic.variables[10], step=self.global_step)
            summary_writer.flush()
            self.global_step = self.global_step + 1

    @tf.function
    def _train_step_actor(self, batch):
        #print("ACTOR >>>>>>>>>>>>>>>>>>>>>>")
        with tf.GradientTape() as tape:
            entropies = []
            actor_losses = []
            for sample in batch:
                values, newpolicy_probs = self.actor(sample['link_state'], sample['graph_id'], sample['first'], 
                    sample['second'], sample['num_edges'], training=True)
                newpolicy_probs = tf.math.reduce_sum(sample['old_act'] * newpolicy_probs[0])
                ratio = K.exp(K.log(newpolicy_probs) - K.log(tf.math.reduce_sum(sample['old_act']*sample['old_policy_probs'])))
                surr1 = ratio*sample['advantage']
                surr2 = K.clip(ratio, min_value=1 - clipping_val, max_value=1 + clipping_val) * sample['advantage']
                actor_losses.append(K.minimum(surr1, surr2))
                entropies.append(-(newpolicy_probs * K.log(newpolicy_probs)))
            final_entropy = tf.math.reduce_mean(entropies)
            final_loss = -tf.math.reduce_mean(actor_losses)
            total_loss = final_loss - ENTROPY_BETA * final_entropy
        grad = tape.gradient(total_loss, sources=self.actor.variables)
        self.optimizer.apply_gradients(zip(grad, self.actor.variables))
        del tape
        return total_loss

    @tf.function
    def _train_step_critic(self, batch):
        with tf.GradientTape() as tape:
            critic_square = []
            for sample in batch:
                value = self.critic(sample['link_state_critic'], sample['first_critic'],
                    sample['second_critic'], sample['num_edges_critic'], training=True)[0]
                critic_square.append(K.square(sample['return'] - value)) # !!!!
            critic_loss = CRITIC_DISCOUNT * tf.math.reduce_mean(critic_square) # !!!!

        grad = tape.gradient(critic_loss, sources=self.critic.variables)
        self.optimizer.apply_gradients(zip(grad, self.critic.variables))
        del tape
        return critic_loss
    
    @tf.function
    def _critic_step(self, sample):
        value = self.critic(sample['link_state_critic'], sample['first_critic'],
                    sample['second_critic'], sample['num_edges_critic'], training=True)[0]
        critic_sample_loss = K.square(sample['return'] - value)
        return critic_sample_loss
    
    @tf.function
    def _actor_step(self, sample):
        r = self.actor(sample['link_state'], sample['graph_id'], sample['first'], 
                    sample['second'], sample['num_edges'], training=True)
        values = tf.reshape(r, (1, len(r)))
        newpolicy_probs = tf.nn.softmax(values)
        newpolicy_probs = tf.math.reduce_sum(sample['old_act'] * newpolicy_probs[0])
        ratio = K.exp(K.log(newpolicy_probs) - K.log(tf.math.reduce_sum(sample['old_act']*sample['old_policy_probs'])))
        surr1 = ratio*sample['advantage']
        surr2 = K.clip(ratio, min_value=1 - clipping_val, max_value=1 + clipping_val) * sample['advantage']
        loss_sample = K.minimum(surr1, surr2)
        entropy_sample = newpolicy_probs * K.log(newpolicy_probs)
        return loss_sample, entropy_sample

    def _train_step_combined(self, batch):
        entropies = []
        actor_losses = []
        critic_losses = []
        # Optimize weights
        with tf.GradientTape() as tape:
            for sample in batch:
                # ACTOR
                loss_sample, entropy_sample = self._actor_step(sample)
                actor_losses.append(loss_sample)
                entropies.append(entropy_sample)
                # CRITIC
                critic_sample_loss = self._critic_step(sample)
                critic_losses.append(critic_sample_loss)
            
            critic_loss = tf.math.reduce_mean(critic_losses)
            final_entropy = tf.math.reduce_mean(entropies)
            actor_loss = -(tf.math.reduce_mean(actor_losses)) - ENTROPY_BETA * final_entropy
            total_loss = actor_loss + critic_loss
        
        entropies.clear()
        actor_losses.clear()
        critic_losses.clear()
        grad = tape.gradient(total_loss, sources=self.actor.trainable_weights + self.critic.trainable_weights)
        self.optimizer.apply_gradients(zip(grad, self.actor.trainable_weights + self.critic.trainable_weights))
        del tape
        return actor_loss, critic_loss, final_entropy

    def ppo_update(self, actions, actions_probs, env_training, tensors, critic_features, returns, advantages):

        for pos in range(0, int(BUFF_SIZE)):

            tensor = tensors[pos]
            critic_feature = critic_features[pos]
            action = actions[pos]
            ret_value = returns[pos]
            adv_value = advantages[pos]
            action_dist = actions_probs[pos]
            
            final_tensors = ({
                'graph_id': tensor['graph_id'],
                'link_state': tensor['link_state'],
                'first': tensor['first'],
                'second': tensor['second'],
                'num_edges': tensor['num_edges'],
                'link_state_critic': critic_feature['link_state_critic'],
                'old_act': tf.convert_to_tensor(action, dtype=tf.float32),
                'advantage': tf.convert_to_tensor(adv_value, dtype=tf.float32),
                'old_policy_probs': tf.convert_to_tensor(action_dist, dtype=tf.float32),
                'first_critic': critic_feature['first_critic'],
                'second_critic': critic_feature['second_critic'],
                'num_edges_critic': critic_feature['num_edges_critic'],
                'return': tf.convert_to_tensor(ret_value, dtype=tf.float32),
            })      

            self.memory.append(final_tensors)  

        for i in range(PPO_EPOCHS):
            batch = random.sample(self.memory, TRAIN_BATCH_SIZE)
            actor_loss, critic_loss, final_entropy = self._train_step_combined(batch)
        self.memory.clear()
        self._write_tf_summary(actor_loss, critic_loss, final_entropy)
        gc.collect()
        return actor_loss, critic_loss

def get_advantages(values, masks, rewards):
    returns = []
    gae = 0
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * values[i + 1] * masks[i] - values[i]
        gae = delta + gamma * lmbda * masks[i] * gae
        returns.insert(0, gae + values[i])

    adv = np.array(returns) - values[:-1]
    # Normalize advantages to reduce variance
    return returns, (adv - np.mean(adv)) / (np.std(adv) + 1e-10)

if __name__ == "__main__":
    # Parse logs and get best model
    parser = argparse.ArgumentParser(description='Parse file and create plots')

    parser.add_argument('-i', help='iters', type=int, required=True)
    parser.add_argument('-c', help='counter model', type=int, required=True)
    parser.add_argument('-e', help='episode iterations', type=int, required=True)
    args = parser.parse_args()

    # Get the environment and extract the number of actions.
    env_training = gym.make(ENV_NAME)
    np.random.seed(SEED)
    env_training.seed(SEED)
    env_training.generate_environment(graph_topology, listofDemands)

    env_eval = gym.make(ENV_NAME)
    np.random.seed(SEED)
    env_eval.seed(SEED)
    env_eval.generate_environment(graph_topology, listofDemands)

    agent = PPOActorCritic(env_training)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint_actor = tf.train.Checkpoint(model=agent.actor, optimizer=agent.optimizer)
    checkpoint_critic = tf.train.Checkpoint(model=agent.critic, optimizer=agent.optimizer)

    if args.i>0:
        # -1 because the current value is to store the model that we train in this iteration
        checkpoint_actor = tf.train.Checkpoint(model=agent.actor, optimizer=agent.optimizer)
        checkpoint_actor.restore(checkpoint_dir + "/ckpt_ACT-" + str(args.c-1))
        checkpoint_critic = tf.train.Checkpoint(model=agent.critic, optimizer=agent.optimizer)
        checkpoint_critic.restore(checkpoint_dir + "/ckpt_CRT-" + str(args.c-1))

    fileLogs = open("./Logs/exp" + differentiation_str + "Logs.txt", "a")

    if os.path.exists("./tmp/" + differentiation_str + "tmp.pckl"):
        f = open("./tmp/" + differentiation_str + "tmp.pckl", 'rb')
        max_reward = pickle.load(f)
        f.close()
    else:
        max_reward = 0

    reward_id = 0
    evalMeanReward = 0
    counter_store_model = args.c
    rewards_test = np.zeros(EVALUATION_EPISODES)

    for iters in range(args.e):
        states = []
        critic_features = []
        tensors = []
        actions = []
        values = []
        masks = []
        rewards = []
        actions_probs = []

        number_samples_reached = False

        print("EPISODE: ", args.i+iters)
        while not number_samples_reached:
            state, old_demand, old_source, old_destination = env_training.reset()
            while 1:
                # Used to clean the TF cache
                tf.random.set_seed(1)
                action_dist, tensor = agent.pred_action_distrib(env_training, state, old_demand, old_source, old_destination)
                
                features = agent.critic_get_graph_features(env_training, state)
                q_value = agent.critic(features['link_state_critic'], features['first_critic'],
                        features['second_critic'], features['num_edges_critic'], training=False)[0].numpy()[0]

                action = np.random.choice(n_actions, p=action_dist)
                action_onehot = tf.one_hot(action, depth=agent.K, dtype=tf.float32).numpy()

                new_state, reward, done, new_demand, new_source, new_destination = env_training.make_step(state, action, old_demand, old_source, old_destination)
                mask = not done

                states.append((state, old_demand, old_source, old_destination))
                tensors.append(tensor)
                critic_features.append(features)
                actions.append(action_onehot)
                values.append(q_value)
                masks.append(mask)
                rewards.append(reward)
                actions_probs.append(action_dist)

                state = new_state
                old_demand = new_demand
                old_source = new_source
                old_destination = new_destination

                # If we have enough samples
                if len(states) == BUFF_SIZE:
                    number_samples_reached = True
                
                if done:
                    break

        features = agent.critic_get_graph_features(env_training, state)
        q_value = agent.critic(features['link_state_critic'], features['first_critic'],
                features['second_critic'], features['num_edges_critic'], training=False)[0].numpy()[0]        
        values.append(q_value)
        returns, advantages = get_advantages(values, masks, rewards)
        actor_loss, critic_loss = agent.ppo_update(actions, actions_probs, env_training, tensors, critic_features, returns, advantages)
        fileLogs.write(">," + str(actor_loss.numpy()) + ",\n")
        fileLogs.write("<," + str(critic_loss.numpy()) + ",\n")
        fileLogs.flush()

        for ep in range(EVALUATION_EPISODES):
            state, old_demand, old_source, old_destination = env_eval.reset()
            done = False
            rewardAddTest = 0
            while 1:
                action_dist, _ = agent.pred_action_distrib(env_eval, state, old_demand, old_source, old_destination)
                action = np.argmax(action_dist)

                new_state, reward, done, new_demand, new_source, new_destination = env_eval.make_step(state, action, old_demand, old_source, old_destination)
                state = new_state
                old_demand = new_demand
                old_source = new_source
                old_destination = new_destination

                rewardAddTest += reward
                if done:
                    break
            rewards_test[ep] = rewardAddTest

        evalMeanReward = np.mean(rewards_test)
        fileLogs.write(".," + str(evalMeanReward) + ",\n")
  
        if evalMeanReward>max_reward:
            max_reward = evalMeanReward
            reward_id = counter_store_model
            fileLogs.write("MAX REWD: " + str(max_reward) + " REWD_ID: " + str(reward_id) +",\n")
        
        fileLogs.flush()
        
        # Store trained model
        # Storing the model and the tape.gradient make the memory increase
        checkpoint_actor.save(checkpoint_prefix+'_ACT')
        checkpoint_critic.save(checkpoint_prefix+'_CRT')
        counter_store_model = counter_store_model + 1
        K.clear_session()
        gc.collect()

    f = open("./tmp/" + differentiation_str + "tmp.pckl", 'wb')
    pickle.dump(max_reward, f)
    f.close()

