
from helper import *
import numpy as np
import time
from collections import defaultdict
import random

class AIPlayer:
    def __init__(self, player_number, timer):
        self.player_number = player_number
        self.opponent_number = 2 if player_number == 1 else 1
        self.timer = timer
        self.time_limit = 10  # Time to aim for per move (seconds)
        self.type = 'ai'
        self.player_string = 'Player {}: ai4'.format(player_number)
        self.expanded_list = set()
        
        # RAVE-specific dictionaries
        self.q_values = defaultdict(float)  # Store Q-values for state-action pairs
        self.n_values = defaultdict(int)    # Store visit counts for state-action pairs
        self.q_rave = defaultdict(float)    # Store RAVE Q-values for state-action pairs
        self.n_rave = defaultdict(int)      # Store RAVE visit counts

    def get_move(self, state):
        self.dimension = state.shape[0]//2 + 1
        valid_actions = get_valid_actions(state)

        for move in valid_actions:
            player_number = self.player_number
            new_state = self.simulate_move(state, move, player_number)
            ans, way = check_win(new_state,move,player_number)
            if (ans):
                return move
        for move in valid_actions:
            player_number = self.opponent_number
            new_state = self.simulate_move(state, move, player_number)
            ans, way = check_win(new_state,move,player_number)
            if (ans):
                return move
        
        if(self.dimension == 6):
            tri_threat =  self.triangle_defense(state)

            if(tri_threat != None):
                return tri_threat
        
        # Set a safe default move if we run out of time
        best_move = (0,0)

        # Running MCTS Algorithm with RAVE Properties
        best_move = self.mcts(state, valid_actions,500)  # Run MCTS with RAVE
        if(best_move):
            return (int(best_move[0]), int(best_move[1]))
        else:
            return (0,0)

    
    def triangle_defense(self,state):
        blocks = [[(0,0),(1,2),(0,3)],[(0,1),(1,3),(0,4)],[(0,2),(1,4),(0,5)],
                [(0,5),(1,6),(0,8)],[(0,6),(1,7),(0,9)],[(0,7),(1,8),(0,10)],
                [(0,10),(2,9),(3,10)],[(1,10),(3,9),(4,10)],[(2,10),(4,9),(5,10)],
                [(5,10),(6,8),(8,7)],[(6,9),(7,7),(9,6)],[(7,8),(8,6),(10,5)],
                [(10,5),(8,4),(7,2)],[(9,4),(7,3),(6,1)],[(8,3),(6,2),(5,0)],
                [(5,0),(4,1),(2,0)],[(4,0),(3,1),(1,0)],[(3,0),(2,1),(0,0)]]
        
        move = None
        
        for triads in blocks:
            cnt = 0
            move = None
            for ele in triads:
                if state[ele[0]][ele[1]] == self.opponent_number:
                    cnt += 1

            if cnt == 2:
                for ele in triads:
                    if state[ele[0]][ele[1]] == 0:
                        move = ele
                        return move         
        return move


    def mcts(self, state, valid_actions, total_simulations):
        state_tuple = tuple(state.flatten())  # Convert state to a hashable tuple
        expanded_list = set()
        # print("mcts running")

        for move in valid_actions:
            player_number = self.player_number
            new_state = self.simulate_move(state, move, player_number)
            ans, way = check_win(new_state,move,player_number)
            if (ans):
                return move
        for move in valid_actions:
            player_number = self.opponent_number
            new_state = self.simulate_move(state, move, player_number)
            ans, way = check_win(new_state,move,player_number)
            if (ans):
                return move


        for _ in range(total_simulations):
            self.simulate_mcts(state, valid_actions, expanded_list)

        best_move = max(valid_actions, key=lambda move: self.q_values[(state_tuple, move)] / self.n_values[(state_tuple, move)] if self.n_values[(state_tuple, move)] > 0 else 0)
        # print(best_move, self.q_values[(state_tuple, best_move)]/self.n_values[(state_tuple, best_move)] )
        return best_move

    def is_not_terminal_node(self,current_state, expanded_list):
        if(current_state in expanded_list):
            return True
        else:
            expanded_list.add(current_state)
            return False

    
    def simulate_move(self, state, move, player):
        # Deep Copy of the State to incorporate the change
        new_state = np.copy(state)
        new_state[move[0], move[1]] = player
        return new_state

    def simulate_mcts(self, state, valid_actions, expanded_list):
        path = []
        state_tuple = tuple(state.flatten())  # Convert state to tuple
        current_state = state
        current_state_tuple = tuple(current_state.flatten())
        current_player = self.player_number

        move = valid_actions[0]
        winner = -1
        while(self.is_not_terminal_node(current_state_tuple, expanded_list)):
            if not valid_actions:
                break
            move = self.select_move(current_state, valid_actions, current_player)
            path.append((current_state_tuple, move))
            current_state = self.simulate_move(current_state, move, current_player)
            current_state_tuple = tuple(current_state.flatten())
            ans, way = check_win(current_state,move,current_player)
            if(ans):
                winner = current_player 
                break
            valid_actions = get_valid_actions(current_state)
            current_player = 2 if current_player == 1 else 1
        
        # Explansion State
        valid_actions = get_valid_actions(current_state)
        if(len(valid_actions) !=0 and winner == -1):
            move = valid_actions[np.random.randint(len(valid_actions))]
            path.append((current_state_tuple, move))
            current_state = self.simulate_move(current_state,move,current_player)
            current_player = 2 if current_player == 1 else 1
            winner = self.random_playout(current_state, current_player)

        for state_tuple, move in path:
            self.n_values[(state_tuple, move)] += 1
            self.n_rave[(state_tuple, move)] += 1
            if winner == self.player_number:
                self.q_values[(state_tuple, move)] += 1
                self.q_rave[(state_tuple, move)] += 1
            elif (self.dimension <= 5 and winner == self.opponent_number):
                self.q_values[(state_tuple, move)] -= 1
                self.q_rave[(state_tuple, move)] -= 1

    def select_move(self, state, valid_actions, current_player):
        # Select move based on UCT + RAVE
        best_move = set()
        best_value = -float('inf')
        state_tuple = tuple(state.flatten())
        total_simulations = sum(self.n_values[(state_tuple, move)] for move in valid_actions)


        constact_c = 1.8
        R_value = 50

        min_value = float('inf')
        min_states = set()
        for move in valid_actions:
            # if self.n_values[(state_tuple, move)] == 0:
            #     return move  # Exploration: Unvisited moves
            denom = self.n_values[(state_tuple, move)] + 1
            uct_value = (self.q_values[(state_tuple, move)] / denom) + constact_c * np.sqrt(np.log(total_simulations + 1) / denom)
            
            # Incorporate RAVE heuristic
            beta = R_value / (R_value + self.n_values[(state_tuple, move)] + 1)
            rave_value = (self.q_rave[(state_tuple, move)] / (self.n_rave[(state_tuple, move)] + 1)) if self.n_rave[(state_tuple, move)] > 0 else 0
            # hybrid_value = (1 - beta) * uct_value + beta * rave_value
            hybrid_value = uct_value
            
            if hybrid_value > best_value:
                best_value = hybrid_value
                best_move = {move}
            elif hybrid_value == best_value:
                best_move.add(move)

            # For opponent
            if hybrid_value < min_value:
                min_value = hybrid_value
                min_states = {move}
            elif hybrid_value == min_value:
                min_states.add(move)


        # Random selection if all moves have the same value
        opponents_moves = self.potential_winning_moves(state_tuple,self.opponent_number)
        if(opponents_moves):
            return opponents_moves

        if(current_player == self.player_number):
            if best_value == -float('inf'):
                return np.random.choice(valid_actions)
            return random.choice(list(best_move))

        # Incase of oppoent Chance
        if min_value == float('inf'):
            return np.random.choice(valid_actions)
        return random.choice(list(min_states))


    def update_rave_values(self,path, winner):
        for state_tuple, move in path:
            self.n_rave[(state_tuple, move)] += 1
            if winner == self.player_number:
                self.q_rave[(state_tuple, move)] += 1



    def random_playout(self, state, current_player):
        # Perform random moves until the game ends
        path = []
        current_player = current_player
        while True:
            valid_actions = get_valid_actions(state)
            if not valid_actions:
                break
            try:
                move = valid_actions[np.random.randint(len(valid_actions))]
            except:
                print("Error Encountered")
                print(state)
                return 1
            state = self.simulate_move(state, move, current_player)
            state_tuple = tuple(state.flatten())
            path.append((state_tuple, move))
            if(self.is_terminal(move,state,current_player)):
                self.update_rave_values(path, current_player)
                return current_player
            current_player = 2 if current_player == 1 else 1

        return 2 if current_player == 1 else 1

    def potential_winning_moves(self, state, player_number):
        valid_actions = get_valid_actions(state)
        winning_move = None
        for move in valid_actions:
            if self.is_terminal(move, state, player_number):
                return winning_move
        return winning_move

    def is_potential_win(self, state,move,player):
        var, text =  check_win(state,move,player)
        return var

    def is_terminal(self, move,state,player_number):
        # Implement a terminal state check (winning or full board)
        new_state = self.simulate_move(state, move, player_number)
        return check_win(new_state,move,player_number)[0]
    

