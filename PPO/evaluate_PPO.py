import numpy as np
import gym
import os
import gym_environments
import networkx as nx
import random
import matplotlib.pyplot as plt
import argparse
import criticPPO as critic
import actorPPO as actor
from collections import deque
import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

ENV_NAME = 'GraphEnv-v2' # RAND, SAP games
ENV_NAME_AGENT = 'GraphEnv-v1' # PPO
SEED = 9
os.environ['PYTHONHASHSEED']=str(SEED)
np.random.seed(SEED)
random.seed(SEED)

tf.random.set_seed(1)

ITERATIONS = 150
# We assume that the number of samples is always larger than the number of demands any agent can ever allocate
NUM_SAMPLES_ITER = 70

graph_topology = 0 # 0==NSFNET, 1==GEANT, 2===GBN
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

class FLUIDAgent:
    def __init__(self):
        self.K = len(n_actions)

    def act(self, env, state, demand, source, destination):
        pathList = env.allPaths[str(source) + ':' + str(destination)]
        new_state = np.copy(state)
        negativeCapacity = False

        if len(pathList) < self.K:
            path_min_cap = np.ones(len(pathList))  # Stores the minimum capacity existing in a link
            missing_to_allocate = np.zeros(len(pathList))
        else:
            path_min_cap = np.ones(self.K)
            missing_to_allocate = np.zeros(self.K)
       
        other_sp = dict() # Count the number of other shortest paths passing through the links from the K-paths
        path = 0
        while path < len(pathList) and path < self.K:
            currentPath = pathList[path]
            i = 0
            j = 1
            while j < len(currentPath):
                if (str(currentPath[i]) + str(currentPath[j])) in other_sp:
                    other_sp[str(currentPath[i]) + str(currentPath[j])] = other_sp[str(currentPath[i]) + str(currentPath[j])] + 1
                else:
                    other_sp[str(currentPath[i]) + str(currentPath[j])] = 1
                i = i + 1
                j = j + 1
            path = path + 1
        while sum(path_min_cap) > 0:
            path = 0
            can_allocate = 1
            # 0. Iterate over k=4 shortest paths and store minimum capacity available
            while path < len(pathList) and path < self.K:
                currentPath = pathList[path]
                # print(currentPath)
                i = 0
                j = 1

                # Iterate over pairs of nodes and allocate the demand
                min_capacity_avbl = 10000
                while j < len(currentPath):
                    val = other_sp[str(currentPath[i]) + str(currentPath[j])]
                    # print(currentPath[i],currentPath[j],new_state[currentPath[i]][currentPath[j]][2],val)
                    div_cap = new_state[env.edgesDict[str(currentPath[i]) + ':' + str(currentPath[j])]][0]/val
                    if div_cap < min_capacity_avbl:
                        min_capacity_avbl = div_cap
                    i = i + 1
                    j = j + 1
                path_min_cap[path] = min_capacity_avbl
                path = path + 1

            # If there is available capacity to allocate
            if sum(path_min_cap)>0:
                # We store the proportions of the demand to allocate for each shortest path
                proportions = path_min_cap / (sum(path_min_cap))
                # We store the amount of the demand to allocate for each shortest path
                demand_to_allocate = proportions * demand
                it = 0
                # 1. Iterate over all minimum capacities and check if what we want to allocate fits
                # if it doesn't fit, we allocate the maximum
                for all in demand_to_allocate:
                    if all > path_min_cap[it] and it<len(pathList):
                        can_allocate = 0
                        # We allocate the maximum we can
                        currentPath = pathList[it]
                        partial_demand = path_min_cap[it]
                        missing_to_allocate[it] = all-partial_demand
                        i = 0
                        j = 1

                        while j < len(currentPath):
                            edge = env.edgesDict[str(currentPath[i]) + ':' + str(currentPath[j])]
                            new_state[edge][0] = new_state[edge][0] - partial_demand
                            if new_state[edge][0] < 0.01:
                                new_state[edge][0] = 0.0
                            i = i + 1
                            j = j + 1
                    it = it + 1
            # If we can't allocate the demand because everything is full
            else:
                currentPath = pathList[0]
                #print(currentPath)
                negativeCapacity = True
                i = 0
                j = 1
                # 2. Iterate over pairs of nodes and allocate the demand
                while j < len(currentPath):
                    new_state[env.edgesDict[str(currentPath[i]) + ':' + str(currentPath[j])]][0] = new_state[env.edgesDict[str(currentPath[i]) + ':' + str(currentPath[j])]][0] - demand
                    i = i + 1
                    j = j + 1
                break
            # 2. If demand fits well, then we allocate and we go out of the loop
            if can_allocate == 1:
                path = 0
                demand_to_allocate = proportions * demand
                while path < len(pathList) and path < self.K:
                    currentPath = pathList[path]
                    demand = demand_to_allocate[path]
                    if demand > 0:
                        i = 0
                        j = 1

                        # 2. Iterate over pairs of nodes and allocate the demand
                        while j < len(currentPath):
                            new_state[env.edgesDict[str(currentPath[i]) + ':' + str(currentPath[j])]][0] = new_state[env.edgesDict[str(currentPath[i]) + ':' + str(currentPath[j])]][0] - demand
                            i = i + 1
                            j = j + 1
                    path = path + 1
                break
            else:
                demand = sum(missing_to_allocate)
                missing_to_allocate.fill(0)

        return new_state, negativeCapacity

class SAPAgent:
    def __init__(self):
        self.K = len(n_actions)

    def act(self, env, state, demand, n1, n2):
        pathList = env.allPaths[str(n1) +':'+ str(n2)]
        path = 0
        negativeCapacity = False
        allocated = 0 # Indicates 1 if we allocated the demand, 0 otherwise
        new_state = np.copy(state)
        while allocated==0 and path < len(pathList) and path<self.K:
            currentPath = pathList[path]
            can_allocate = 1 # Indicates 1 if we can allocate the demand, 0 otherwise
            i = 0
            j = 1

            # 1. Iterate over pairs of nodes and check if we can allocate the demand
            while j < len(currentPath):
                if new_state[env.edgesDict[str(currentPath[i]) + ':' + str(currentPath[j])]][0] - demand < 0:
                    can_allocate = 0
                    break
                i = i + 1
                j = j + 1

            if can_allocate==1:
                i = 0
                j = 1

                # 2. Iterate over pairs of nodes and allocate the demand
                while j < len(currentPath):
                    new_state[env.edgesDict[str(currentPath[i]) + ':' + str(currentPath[j])]][0] -= demand
                    i = i + 1
                    j = j + 1
                allocated = 1
            path = path + 1

        # If we can't allocate it we just do it in the first path
        if allocated==0:
            negativeCapacity = True
            currentPath = pathList[0]
            i = 0
            j = 1

            # 3. Iterate over pairs of nodes and allocate the demand
            while j < len(currentPath):
                new_state[env.edgesDict[str(currentPath[i]) + ':' + str(currentPath[j])]][0] -= demand
                i = i + 1
                j = j + 1

        return new_state, negativeCapacity

class RANDAgent:
    def __init__(self):
        self.K = len(n_actions)

    def act(self, env, state, demand, n1, n2):
        pathList = env.allPaths[str(n1) +':'+  str(n2)]
        new_state = np.copy(state)
        negativeCapacity = False

        free_capacity = 0
        id_last_free = -1 # Indicates the id of the last free path where we will allocate the demand
        path = 0
        # Check if there are at least 2 paths
        while free_capacity < 2 and path < len(pathList) and path<4:
            currentPath = pathList[path]
            can_allocate = 1  # Indicates 1 if we can allocate the demand, 0 otherwise
            i = 0
            j = 1

            # 1. Iterate over pairs of nodes and check if we can allocate the demand
            while j < len(currentPath):
                if new_state[env.edgesDict[str(currentPath[i]) + ':' + str(currentPath[j])]][0] - demand < 0:
                    can_allocate = 0
                    break
                i = i + 1
                j = j + 1

            if can_allocate == 1:
                free_capacity = free_capacity + 1
                id_last_free = path
            path = path + 1

        # If we can't allocate anything
        if free_capacity == 0:
            currentPath = pathList[0]
            negativeCapacity = True
            i = 0
            j = 1

            # 3. Iterate over pairs of nodes and allocate the demand
            while j < len(currentPath):
                new_state[env.edgesDict[str(currentPath[i]) + ':' + str(currentPath[j])]][0] -= demand
                i = i + 1
                j = j + 1

            return new_state, negativeCapacity
        # If there is just one path to allocate we allocate it there
        elif free_capacity == 1:
            currentPath = pathList[id_last_free]
            i = 0
            j = 1

            # 3. Iterate over pairs of nodes and allocate the demand
            while j < len(currentPath):
                new_state[env.edgesDict[str(currentPath[i]) + ':' + str(currentPath[j])]][0] -= demand
                i = i + 1
                j = j + 1

            return new_state, negativeCapacity
        else:
            allocated = 0 # Indicates 1 if we allocated the demand, 0 otherwise
            while allocated==0:
                currentPath = random.choice(pathList)
                can_allocate = 1 # Indicates 1 if we can allocate the demand, 0 otherwise
                i = 0
                j = 1

                # 1. Iterate over pairs of nodes and check if we can allocate the demand
                while j < len(currentPath):
                    if new_state[env.edgesDict[str(currentPath[i]) + ':' + str(currentPath[j])]][0] - demand < 0:
                        can_allocate = 0
                        break
                    i = i + 1
                    j = j + 1

                if can_allocate == 1:
                    i = 0
                    j = 1

                    # 2. Iterate over pairs of nodes and allocate the demand
                    while j < len(currentPath):
                        new_state[env.edgesDict[str(currentPath[i]) + ':' + str(currentPath[j])]][0] -= demand
                        i = i + 1
                        j = j + 1
                    allocated = 1

        return new_state, negativeCapacity

def old_cummax(alist, extractor):
    with tf.name_scope('cummax'):
        maxes = [tf.reduce_max(extractor(v)) + 1 for v in alist]
        cummaxes = [tf.zeros_like(maxes[0])]
        for i in range(len(maxes) - 1):
            cummaxes.append(tf.math.add_n(maxes[0:i + 1]))
    return cummaxes

class PPOActorCritic:
    def __init__(self, env_training):
        self.K = len(n_actions)
        self.listValues = None
        self.softMaxValues = None

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

        # Predict values for all graphs within tensors
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

    def critic_get_graph_features(self, env, copyGraph):
        """
        We iterate over the converted graph nodes and take the features. The capacity and bw allocated features
        are normalized on the fly.
        """
        length = env.numEdges
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

def play_fluid_games(experience_memory, graph_topology):
    env_fluid = gym.make(ENV_NAME)
    env_fluid.seed(SEED)
    env_fluid.generate_environment(graph_topology, listofDemands)

    agent = FLUIDAgent()
    rewards_fluid = np.zeros(ITERATIONS)

    rewardAdd = 0
    reward_it = 0
    iter_episode = 0 # Iterates over samples within the same episode
    new_episode = True
    wait_for_new_episode = False
    new_episode_it = 0 # Iterates over EPISODES
    while iter_episode < len(experience_memory):
        if new_episode:
            new_episode = False
            demand = experience_memory[iter_episode][1]
            source = experience_memory[iter_episode][2]
            destination = experience_memory[iter_episode][3]
            state = env_fluid.reset()

            new_state, negCap = agent.act(env_fluid, state, demand, source, destination)
            reward, done = env_fluid.make_step(negCap, demand)
            rewardAdd = rewardAdd + reward
            state = new_state

            if done:
                rewards_fluid[reward_it] = rewardAdd
                reward_it = reward_it + 1
                wait_for_new_episode = True

            iter_episode = iter_episode + 1
        else:
            if experience_memory[iter_episode][0] != new_episode_it:
                print("FLUID ERROR! The experience replay buffer needs more samples/episode")
                os.kill(os.getpid(), 9)

            demand = experience_memory[iter_episode][1]
            source = experience_memory[iter_episode][2]
            destination = experience_memory[iter_episode][3]
            new_state, negCap = agent.act(env_fluid, state, demand, source, destination)
            reward, done = env_fluid.make_step(negCap, demand)
            rewardAdd = rewardAdd + reward
            state = new_state

            if done:
                rewards_fluid[reward_it] = rewardAdd
                reward_it = reward_it + 1
                wait_for_new_episode = True

            iter_episode = iter_episode + 1
        if wait_for_new_episode:
            rewardAdd = 0
            wait_for_new_episode = False
            new_episode = True
            new_episode_it = new_episode_it + 1
            iter_episode = new_episode_it*NUM_SAMPLES_ITER
    print(rewards_fluid)
    return rewards_fluid

def play_rand_games(experience_memory, graph_topology):
    env_rand = gym.make(ENV_NAME)
    env_rand.seed(SEED)
    env_rand.generate_environment(graph_topology, listofDemands)

    agent = RANDAgent()
    rewards_rand = np.zeros(ITERATIONS)

    rewardAdd = 0
    reward_it = 0
    iter_episode = 0 # Iterates over samples within the same episode
    new_episode = True
    wait_for_new_episode = False
    new_episode_it = 0 # Iterates over EPISODES
    while iter_episode < len(experience_memory):
        if new_episode:
            new_episode = False
            demand = experience_memory[iter_episode][1]
            source = experience_memory[iter_episode][2]
            destination = experience_memory[iter_episode][3]
            state = env_rand.reset()

            new_state, negCap = agent.act(env_rand, state, demand, source, destination)
            reward, done = env_rand.make_step(negCap, demand)
            rewardAdd = rewardAdd + reward
            state = new_state

            if done:
                rewards_rand[reward_it] = rewardAdd
                reward_it = reward_it + 1
                wait_for_new_episode = True

            iter_episode = iter_episode + 1
        else:
            if experience_memory[iter_episode][0] != new_episode_it:
                print("RAND ERROR! The experience replay buffer needs more samples/episode")
                os.kill(os.getpid(), 9)

            demand = experience_memory[iter_episode][1]
            source = experience_memory[iter_episode][2]
            destination = experience_memory[iter_episode][3]
            new_state, negCap = agent.act(env_rand, state, demand, source, destination)
            reward, done = env_rand.make_step(negCap, demand)
            rewardAdd = rewardAdd + reward
            state = new_state

            if done:
                rewards_rand[reward_it] = rewardAdd
                reward_it = reward_it + 1
                wait_for_new_episode = True

            iter_episode = iter_episode + 1
        if wait_for_new_episode:
            rewardAdd = 0
            wait_for_new_episode = False
            new_episode = True
            new_episode_it = new_episode_it + 1
            iter_episode = new_episode_it*NUM_SAMPLES_ITER
    print(rewards_rand)
    return rewards_rand

def play_sap_games(experience_memory, graph_topology):
    env_sap = gym.make(ENV_NAME)
    env_sap.seed(SEED)
    env_sap.generate_environment(graph_topology, listofDemands)

    agent = SAPAgent()
    rewards_sap = np.zeros(ITERATIONS)

    rewardAdd = 0
    reward_it = 0
    iter_episode = 0  # Iterates over samples within the same episode
    new_episode = True
    wait_for_new_episode = False
    new_episode_it = 0  # Iterates over EPISODES
    while iter_episode < len(experience_memory):
        if new_episode:
            new_episode = False
            demand = experience_memory[iter_episode][1]
            source = experience_memory[iter_episode][2]
            destination = experience_memory[iter_episode][3]
            state = env_sap.reset()

            new_state, negCap = agent.act(env_sap, state, demand, source, destination)
            reward, done = env_sap.make_step(negCap, demand)
            rewardAdd = rewardAdd + reward
            state = new_state

            if done:
                rewards_sap[reward_it] = rewardAdd
                reward_it = reward_it + 1
                wait_for_new_episode = True

            iter_episode = iter_episode + 1
        else:
            if experience_memory[iter_episode][0]!=new_episode_it:
                print("SAP ERROR! The experience replay buffer needs more samples/episode")
                os.kill(os.getpid(), 9)

            demand = experience_memory[iter_episode][1]
            source = experience_memory[iter_episode][2]
            destination = experience_memory[iter_episode][3]
            new_state, negCap = agent.act(env_sap, state, demand, source, destination)
            reward, done = env_sap.make_step(negCap, demand)
            rewardAdd = rewardAdd + reward
            state = new_state

            if done:
                rewards_sap[reward_it] = rewardAdd
                reward_it = reward_it + 1
                wait_for_new_episode = True

            iter_episode = iter_episode + 1
        if wait_for_new_episode:
            rewardAdd = 0
            wait_for_new_episode = False
            new_episode = True
            new_episode_it = new_episode_it + 1
            iter_episode = new_episode_it * NUM_SAMPLES_ITER
    print(rewards_sap)
    return rewards_sap

def play_ppo_games(experience_memory, env_eval, agent):
    rewards_ppo = np.zeros(ITERATIONS)

    rewardAdd = 0
    reward_it = 0
    iter_episode = 0  # Iterates over samples within the same episode
    new_episode = True
    wait_for_new_episode = False
    new_episode_it = 0  # Iterates over EPISODES
    while iter_episode < len(experience_memory):
        if new_episode:
            new_episode = False
            old_demand = experience_memory[iter_episode][1]
            old_source = experience_memory[iter_episode][2]
            old_destination = experience_memory[iter_episode][3]
            state = env_eval.eval_sap_reset(demand, source, destination)

            action_dist, _ = agent.pred_action_distrib(env_eval, state, old_demand, old_source, old_destination)
            action = np.argmax(action_dist)

            new_state, reward, done, new_demand, new_source, new_destination = env_eval.make_step(state, action, old_demand, old_source, old_destination)
            state = new_state
            old_demand = new_demand
            old_source = new_source
            old_destination = new_destination

            rewardAdd += reward
            if done:
                rewards_ppo[reward_it] = rewardAdd
                reward_it = reward_it + 1
                wait_for_new_episode = True
            iter_episode = iter_episode + 1
        else:
            if experience_memory[iter_episode][0] != new_episode_it:
                print("PPOACTOR ERROR! The experience replay buffer needs more samples/episode")
                os.kill(os.getpid(), 9)

            old_demand = experience_memory[iter_episode][1]
            old_source = experience_memory[iter_episode][2]
            old_destination = experience_memory[iter_episode][3]

            action_dist, _ = agent.pred_action_distrib(env_eval, state, old_demand, old_source, old_destination)
            action = np.argmax(action_dist)

            new_state, reward, done, new_demand, new_source, new_destination = env_eval.make_step(state, action, old_demand, old_source, old_destination)
            state = new_state
            old_demand = new_demand
            old_source = new_source
            old_destination = new_destination

            rewardAdd += reward
            if done:
                rewards_ppo[reward_it] = rewardAdd
                reward_it = reward_it + 1
                wait_for_new_episode = True
            iter_episode = iter_episode + 1
        if wait_for_new_episode:
            rewardAdd = 0
            wait_for_new_episode = False
            new_episode = True
            new_episode_it = new_episode_it + 1
            print("PPO >>> ", new_episode_it)
            iter_episode = new_episode_it * NUM_SAMPLES_ITER
    return rewards_ppo

if __name__ == "__main__":
    # Parse logs and get best model
    # python evaluate_PPO.py -d ./Logs/expPPO_NSFNet_agentLogs.txt
    parser = argparse.ArgumentParser(description='Parse file and create plots')

    parser.add_argument('-d', help='data file', type=str, required=True, nargs='+')
    args = parser.parse_args()

    aux = args.d[0].split(".")
    aux = aux[1].split("exp")
    differentiation_str = str(aux[1].split("Logs")[0])

    if not os.path.exists("./Images"):
        os.makedirs("./Images")

    model_id = 0
    topo = ""
    if graph_topology==0:
        topo = "NSFNet"
    elif graph_topology==1:
        topo = "GEANT"
    elif graph_topology==2:
        topo = "GBN"
    elif graph_topology==5:
        topo = "GBN50"
    else:
        topo = "AUX"
    # store_experiences = open("../drl_code/NOU_requests_queue_"+str(topo)+"300.txt", "w")

    with open(args.d[0]) as fp:
        for line in reversed(list(fp)):
            arrayLine = line.split(":")
            if arrayLine[0]=='MAX REWD':
                model_id = int(arrayLine[2].split(",")[0])
                break

    env_eval = gym.make(ENV_NAME_AGENT)
    np.random.seed(SEED)
    env_eval.seed(SEED)
    env_eval.generate_environment(graph_topology, listofDemands)

    ppo_agent = PPOActorCritic(env_eval)
    checkpoint_dir = "./models" + differentiation_str
    checkpoint = tf.train.Checkpoint(model=ppo_agent.actor, optimizer=ppo_agent.optimizer)
    # Restore variables on creation if a checkpoint exists.
    checkpoint.restore(checkpoint_dir + "/ckpt_ACT-" + str(model_id))
    print("Load model " + checkpoint_dir + "/ckpt_ACT-" + str(model_id))

    means_sap = np.zeros(ITERATIONS)
    means_ppo = np.zeros(ITERATIONS)
    means_rand = np.zeros(ITERATIONS)
    means_fluid = np.zeros(ITERATIONS)
    iters = np.zeros(ITERATIONS)

    experience_memory = deque(maxlen=200000)

    # Generate lists of determined size of demands. The different agents will iterate over this same list
    for iter in range(ITERATIONS):
        for sample in range(NUM_SAMPLES_ITER):
            demand = random.choice(listofDemands)
            source = random.choice(env_eval.nodes)

            # We pick a pair of SOURCE,DESTINATION different nodes
            while True:
                destination = random.choice(env_eval.nodes)
                if destination != source:
                    # We generate unique demands that don't overlap with existing topology edges
                    experience_memory.append((iter, demand, source, destination))
                    # store_experiences.write(str(iter)+","+str(source)+","+str(destination)+","+str(demand)+"\n")
                    break

    # store_experiences.close()
    rewards_fluid = play_fluid_games(experience_memory, graph_topology)
    rewards_rand = play_rand_games(experience_memory, graph_topology)
    rewards_sap = play_sap_games(experience_memory, graph_topology)
    rewards_ppo = play_ppo_games(experience_memory, env_eval, ppo_agent)

    # rewards_rand.tofile('../drl_code/NOU_rewards_rand'+topo+'300.dat')
    # rewards_ppo.tofile('../drl_code/NOU_rewards_ppo'+topo+'300.dat')
    # rewards_fluid.tofile('../drl_code/NOU_rewards_fluid'+topo+'300.dat')

    plt.rcParams.update({'font.size': 12})
    plt.plot(rewards_ppo, 'r', label="PPO")
    plt.plot(rewards_sap, 'b', label="SAP")
    plt.plot(rewards_rand, 'g', label="RAND")
    plt.plot(rewards_fluid, 'y', label="FLUID")
    mean = np.mean(rewards_ppo) #PPO
    means_ppo.fill(mean)
    plt.plot(means_ppo, 'r', linestyle="-.")
    mean = np.mean(rewards_sap) #SAP
    means_sap.fill(mean)
    plt.plot(means_sap, 'b', linestyle=":")
    mean = np.mean(rewards_rand) #RAND
    means_rand.fill(mean)
    plt.plot(means_rand, 'g', linestyle="--")
    mean = np.mean(rewards_fluid) #FLUID
    means_fluid.fill(mean)
    plt.plot(means_fluid, 'y', linestyle=":")
    plt.xlabel("Games", fontsize=14, fontweight='bold')
    plt.ylabel("Score", fontsize=14, fontweight='bold')
    lgd = plt.legend(loc="lower left", bbox_to_anchor=(-0.1, -0.24),
            ncol=4, fancybox=True, shadow=True)
    
    plt.savefig("./Images/ModelEval" + differentiation_str+topo+".pdf", bbox_extra_artists=(lgd,), bbox_inches='tight')
    #plt.show()

