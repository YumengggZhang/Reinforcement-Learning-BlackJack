import random
from matplotlib import pyplot as plt
import numpy as np

class Blackjack():

    def __init__(self):
        '''mapping from random selcted number of probability 1/13 to card value'''
        self.mapping = {1:11,2:2,3:3,4:4,5:5,6:6,7:7,8:8,9:9,10:10,11:10,12:10,13:10}
        self.returns = {}
        self.buffer = []
        '''state[0] for dealer showing and state[1] for player hand'''
        self.states = []
        for i in range(2,12):
            for j in range(12,23):
                self.states.append((i,j))

        '''self.actions stores policies for each state'''
        self.actions = {}
        '''initialize Q function with all zero, and store q-values for each state-action pair'''
        self.Q_s_a = {}
        for state in self.states:
            self.Q_s_a[(state,'hit')] = 0
            self.Q_s_a[(state,'stick')] = 0

            if state[1] >= 21: #If current player have card exceed 20 we set to stick as default for a quiker convergence
                self.actions[state] = 'stick'
            else:
                self.actions[state] = ['hit','stick'][random.randint(0,1)] #otherwise initalize actions randomly



    def deck(self):
        return random.randint(1,13)

    def dealer_action(self,state):
        '''dealer will keep to hit until exceed 16'''
        if state[0] > 16:
            if state[0] > 22:
                # dealer bust
                return 1
            else: #compare dealers card with players final hand
                if state[0] > state[1]:
                    return -1
                elif state[0] == state[1]:
                    return 0
                else:
                    return 1
        else:
            state = (state[0] + self.mapping[self.deck()], state[1])
            return self.dealer_action(state)

    def play(self,state,action):
        if action == 'hit':
            state = (state[0], state[1] + self.mapping[self.deck()])
            if state[1] > 22:
                return -1
            else:
                '''player will take next action based on current policy'''
                action = self.actions[state]
                '''update new state action pair to buffer'''
                self.buffer.append((state, action))
                return self.play(state,action)
        else:
            '''after player stick dealer start to play'''
            return self.dealer_action(state)

    def deal_cards(self):
        card = self.mapping[self.deck()]
        while card < 12:
            card += self.mapping[self.deck()]
        return card

    def train_step(self):
        '''clear buffer for each episode'''
        self.buffer = []
        # state = self.states[random.randint(0,len(self.states)-1)]
        '''random pick cards for player and dealer'''
        player_hand = self.deal_cards()
        dealer_showing = self.mapping[self.deck()]
        state = (dealer_showing,player_hand)
        action = ['hit','stick'][random.randint(0,1)]
        '''buffer the inital state and action'''
        self.buffer.append((state,action))
        '''get reward for each episode by calling play'''
        episode_return = self.play(state,action)
        '''update the q value for each state in the buffer'''
        for (state,action) in self.buffer:
            if (state,action) not in self.returns.keys():
                self.returns[(state,action)] = []
            self.returns[(state,action)].append(episode_return)
            self.Q_s_a[(state,action)] = sum(self.returns[(state,action)])/len(self.returns[(state,action)])
            '''update optimal action'''
            if self.Q_s_a[(state,'hit')] > self.Q_s_a[(state,'stick')]:
                self.actions[state] = 'hit'
            else:
                self.actions[state] = 'stick'

    def train(self,n):
        for i in range(1000*n):
            self.train_step()

    def evaluate(self):
        '''evaluate current policy using average return of 1ooo episodes'''
        return_lissy = []
        for i in range(1000):
            player_hand = self.deal_cards()
            dealer_showing = self.mapping[self.deck()]
            state = (dealer_showing, player_hand)
            action = self.actions[state]
            return_lissy.append(self.play(state,action))
        return sum(return_lissy)/1000

    def plot_policy(self,ax,fig):
        '''plot heatmap for optimal policy'''
        Pi = np.zeros((11, 10))
        for i in range(10,-1,-1):
            for j in range(10):
                if self.actions[(j + 2, i + 12)] == 'hit':
                    Pi[i][j] = 1
                else:
                    Pi[i][j] = 0

        ax.set(xticks = (range(10)),
                xticklabels = ['2','3','4','5','6','7','8','9','10','A'],
                yticks= (range(11,)),
                yticklabels = (range(12,23)),
                xlabel = "Dealer's showing",
                ylabel = "Player's hand",
                title = 'Optimal Policy')
        c = ax.imshow(Pi,cmap ='copper_r',interpolation='nearest',origin='lower')
        fig.colorbar(c,ax=ax)

    def plot_avg_reward(self,ax,result_lissy):
        ax.set(title='Average Reward after running the n-th 1000 steps')
        ax.plot(result_lissy)

if __name__ == '__main__':
    model = Blackjack()
    result_lissy = []
    for n in range(1,101):
        model.train(n)
        result = model.evaluate()
        print('The training return after training',str(n*1000),'steps are',result)
        result_lissy.append(result)

    print(model.actions)
    fig, (ax1,ax2) = plt.subplots(2)
    model.plot_policy(ax1,fig)
    model.plot_avg_reward(ax2,result_lissy)
    plt.show()

