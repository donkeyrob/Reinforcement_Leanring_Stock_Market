# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 16:40:12 2018

@author: xiang
"""

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import os
path = os.chdir('C:\\Users\\xiang\\OneDrive\\Projects\\Highfreq')

data = pd.read_csv("data2.csv")

#seperate training self.data
train_data = data[data['symbol']=="STOCKB"]
test_data = data[data['symbol']=='STOCKA']

#remove duplicate records
train_data = train_data.drop_duplicates()
test_data = test_data.drop_duplicates()


# ============split dataset into 100 subsets==================
train_sub = []
subsets = 100
for i in range(subsets):
    train_sub.append(train_data[i::subsets])
# =============================================================================


#Visualization
    #plot training data
nozero = train_data[train_data['bid'] != 0]
nozero = nozero[1000:150000]
plt.plot(nozero['time'],nozero['bid'])
plt.plot(nozero['time'],nozero['ask'])
plt.show()
    #plot testing data
nozero_test = test_data[test_data['bid'] != 0]
plt.plot(nozero_test['time'],nozero_test['bid'])
plt.plot(nozero_test['time'],nozero_test['ask'])
plt.show()
# =============================================================================
# #plot bid vs time visualization.
# #replace 0 with trade price
# zeros = self.data['bid']
# zeros[zeros == 0] = self.data.trdpx[self.data['trdpx']!=0]
# #plot
# plt.plot(self.data.time[0:1000],zeros[0:1000])
# 
# max(zeros)
# min(zeros)
# =============================================================================
# =============================================================================
# #improve training efficiency
# #Remove consecutive duplicated values 
# from itertools import groupby
# 
# def unique_d(data):
#     nozerobid = data
#     groups = groupby(nozerobid)
#     groupbid = [label for label in groups]
#     uniquebid = []
#     
#     for i in range(len(groupbid)):
#         uniquebid.append(groupbid[i][0])
#     
#     plt.plot(uniquebid)
#     return uniquebid
# 
# uniqueask = unique_d(nozero['ask'])
# uniquebid = unique_d(nozero['bid'])
# 
# plt.plot(uniquebid)
# plt.plot(uniqueask)
# plt.show()
# =============================================================================
########## Learning Agent##############
class Agent:
    
    def __init__(self, data, epsilon=1.0, alpha=0.5, gamma=0.5, hist=500, a =0.03):
        self.t = 10
        self.valid_actions = ['buy','sell','hold']
        self.data = data
        self.Q = dict()          # Create a Q-table which will be a dictionary of tuples
        self.epsilon = epsilon   # Random exploration factor
        self.alpha = alpha 
        self.gamma = gamma
        self.hist = hist
        self.profit = 0
        self.position= []
        self.learning = True
        self.records = []
        self.a = a #decay rate
        self.turn = 0
        self.done = False
        self.total_profit = []
        
    def reset(self):
        self.t = 10
        self.position = []
        self.profit = 0
        self.records = []
        self.done = False
    
    def build_state(self, t):
        if t < self.hist:
            data_hist = self.data.iloc[:t+1]
        else:
            data_hist = self.data.iloc[(t-self.hist):t+1]
        data_t = self.data.iloc[t]
        data_hist = data_hist[data_hist['bid']!=0]
        data_mix = data_hist['bid'].append(data_hist['ask'])
        min_data = min(data_mix)
        max_data = max(data_mix)
        max_min = max_data-min_data
 
        bid = round((data_t['bid']-min_data)/max_min,1)
        ask = round((data_t['ask']-min_data)/max_min,1)
# =============Test 2 Center and diff : too many empty states=================================================
#         center_data = round((max_data+min_data)/2,0)
#         center_t = round((data_t['bid']+data_t['ask'])/2,0)-center_data
#         diff = data_t['ask'] - data_t['bid']
# =============================================================================
# ================Test 1=============================================
#         min_data = min(data_mix)
#         max_data = max(data_mix)
#         max_min = max_data-min_data
# 
#         bid = round((data_t['bid']-min_data)/max_min,1)
#         ask = round((data_t['ask']-min_data)/max_min,1)
# =============================================================================
# =============================================================================
#          #trade signal - indicating type of trade happen most in the past.
#         if sum(data_hist['trdsd']) > 0:
#             trdsig = 1
#         elif sum(data_hist['trdsd']) == 0:
#             trdsig = 0
#         else:
#             trdsig = -1        
# =============================================================================
        state = (bid,ask)
        return state
    
    def create_Q(self,state):        
        if state not in self.Q:
            self.Q[state] = dict()
            for action in self.valid_actions:
                self.Q[state][action] = 0.0
        return
    
    def get_maxQ(self,state):
        try:
            maxQ = max(self.Q[state].values())
        except:
            maxQ = 0
        return maxQ

    def get_next_state(self):
        return self.build_state(self.t + 1)
       
    def choose_action(self,state):     
        act_list = []
        if random.uniform(0,1) < self.epsilon:
            action = random.choice(self.valid_actions)
        else: #select action with maxQ. if multiple actions, select hold.
            for act in self.Q[state]:
                 if self.Q[state][act] >= self.get_maxQ(state):
                     action = act
                     act_list.append(act)
            if len(act_list) >= 2:
                 action = 'hold'
# ===buy won't be updated to be negative if sell not good==================
#             for act in self.Q[state]:
#                 if self.Q[state][act] >= self.get_maxQ(state):
#                     action = act
#                     act_list.append(act)
#             if len(act_list) >= 2:
#                 action = random.choice(act_list)
# =============================================================================
        return action
        
    def learn(self, state, action, reward): #update Q table
        self.Q[state][action] = (1 - self.alpha)*self.Q[state][action] + \
            self.alpha*(reward + self.gamma * self.get_maxQ(self.get_next_state()))        
        
        return

    def sell(self): #steps to take when selling
        rewards = 0
        bid_left = self.data.iloc[self.t]['bidsz'] 
        #When what we have is less than how many they bid.
        while self.position != [] and bid_left - self.position[0][1] >= 0:
            rewards += (self.data['bid'].iloc[self.t]-self.position[0][0])*self.position[0][1]
            self.profit += rewards
            bid_left -= self.position[0][1]               
            self.position.pop(0)
        #When what we have is more than how many they bid.
        if self.position != [] and bid_left != 0:   
            rewards += (self.data['bid'].iloc[self.t]-self.position[0][0])*bid_left
            self.profit += rewards
            self.position[0][1] -= bid_left
        
        return rewards
    
    def step(self,act): #steps to take after choosing an action
        rewards = 0
        if act == "buy":
            self.position.append(list(self.data[['ask', 'asksz']].iloc[self.t]))
        elif act == "sell" and self.position != []:
            rewards = self.sell()

        return rewards

    def update(self): #
        state = self.build_state(self.t)          # Get current state
        self.create_Q(state)                 # Create 'state' in Q-table
        action = self.choose_action(state)  # Choose an action
        rewards = self.step(action)     # Receive a reward
        if self.learning:
            self.learn(state, action, rewards)   # Q-learn
            self.epsilon = np.exp(-self.a*(self.turn))
            
        self.records.append((self.t,state,action,rewards,self.profit))#Record
        self.t += 1
        if self.data.iloc[self.t]['bid']==0:
            self.t+=1
        if self.t >= len(self.data)-1:
            self.done = True
        if self.done:
            self.total_profit.append(self.profit)
            print("--------------------------------")
            print("Turn : " + str(self.turn))
            print("Total Profits: " + str(self.profit))
            print("--------------------------------")
        return
                
                
t1 = Agent(train_sub)

def train():
    while t1.turn < subsets:
        t1.data = train_sub[t1.turn]
        while not t1.done:
            t1.update()
        t1.reset()
        t1.turn += 1
    return

# ======call train() to start training========================================
# train()
# =============================================================================

# ======================save result=============================================
import json
import csv
os.getcwd()
# as requested in comment
newQ = {}
for key in t1.Q.keys():
  if type(key) is not str:
    try:
      newQ[str(key)] = t1.Q[key]
    except:
      try:
        newQ[repr(key)] = t1.Q[key]
      except:
        pass

exDict = newQ

with open('Qtable.txt', 'w') as file:
     file.write(json.dumps(exDict))
with open('Records.txt', 'w') as f:
    for item in t1.records:
        f.write("%s\n" % item)
import pickle
#write list
with open('Records.txt', 'wb') as fp:
    pickle.dump(t1.records, fp)   
#read list    
with open('Records.txt', 'rb') as fp:
    b = pickle.load(fp)       
with open('Profits.csv','w') as file:
    writer = csv.writer(file,quoting = csv.QUOTE_ALL)
    writer.writerow(t1.total_profit)
# =============================================================================

def test(data,hist):
    t1.reset()
    t1.data = data
    t1.learning = False
    t1.epsilon = 0
    t1.hist = hist
    t1.t = 10
    t1.turn = 0
    while not t1.done:
        t1.update()
    
    return

# ======call test function to start test========================================================
# test(test_data,hist = 1000)
# =============================================================================
    
####Analysing the Result
profits_test = []
for i in range(len(t1.records)):
    profits_test.append(t1.records[i][4])
#plot return 
plt.plot(profits_test)
##plot buy and sell on the price
#buy and sell records
list_action = []
for i in range(len(t1.records)):
    list_action.append(t1.records[i][2])

##create a part of the data frame for plot
data_plot = nozero_test[2000:][['bid','ask']]
#get list of action
list_action = list_action[2000:len(nozero_test)]
#add list of action to data_plot
data_plot['action'] = list_action
#subset the data_plot
sub_plot = data_plot[::200]

plt.plot(sub_plot['ask'],'-bd',markevery = [i for i,x in enumerate(sub_plot['action']=='buy') if x],markersize = 20,mfc = 'red')
plt.savefig('buy.png')
plt.plot(sub_plot['bid'],'-rd',markevery = [i for i,x in enumerate(sub_plot['action']=='sell') if x],markersize = 10,mfc = 'blue')
plt.savefig('sell.png')
plt.show()

# trying to make live graph. failed: response time so long========================================================
# #live graph of return
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# from matplotlib import style
# 
# style.use('fivethirtyeight')
# 
# fig = plt.figure()
# ax1 = fig.add_subplot(1,1,1)
# 
# def animate(i):
#     graph_data = open('example.txt','r').read()
#     lines = graph_data.split('\n')
#     xs = []
#     ys = []
#     for line in lines:
#         if len(line) > 1:
#             x, y = line.split(',')
#             xs.append(x)
#             ys.append(y)
#     ax1.clear()
#     ax1.plot(xs, ys)
# 
# ani = animation.FuncAnimation(fig, animate, interval=1000)
# plt.show()


# #def new test to make live
# def newtest(data,hist):
#     t1.reset()
#     t1.data = data
#     t1.learning = False
#     t1.epsilon = 0
#     t1.hist = hist
#     t1.t = 10
#     t1.turn = 0
#     while not t1.done:
#         t1.update()
#         info = str(t1.t) + ',' + str(t1.profit) + '\n'
#         with open('example.txt','a') as file:
#             file.write(info)
#     return
# 
# =============================================================================


