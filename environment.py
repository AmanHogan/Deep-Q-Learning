"""Models the environment"""

import numpy as np
from globals_vars import *
from tabulate import tabulate
from logger.logger import log
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


class ElevatorEnv:
    """Models the elevator environment"""

    def __init__(self, num_floors=6, max_capacity=2, start_floors=[1], 
                 start_prob=[1], exit_floors=[2, 3, 4, 5, 6], exit_prob=[0.2, 0.2, 0.2, 0.2, 0.2], 
                 timestep=5, arrival_rate=.1):
        
        self.num_floors = num_floors

        self.start_floors = start_floors
        """Start floors EX: [1]"""
        self.start_prob = start_prob
        """Probability of landing on start floor. EX: [.2,.2,.2,.2,.2]"""
        self.exit_floors = exit_floors
        """Exit floors EX: [2,3,4,5,6]"""
        self.exit_prob = exit_prob
        """Probability of landing on exit floor. EX: [.2,.2,.2,.2,.2]"""
        self.people = 0
        """Total people who entered simualtion"""
        self.a_occupancy = 0
        """Total people in elevator A"""
        self.b_occupancy = 0
        """Total people in elevator B"""
        self.max_occupancy = max_capacity
        """Max occupancy in elevator"""

        self.timestep = timestep
        self.arrival_rate = arrival_rate
        self.state = self.init_state()

        # Tracking variables
        self.exit_count = 0
        self.entered_sim_count = 0
        self.max_bandwidth_count = 0
        self.current_time = 0

    def init_state(self):
        """Initializes start state vector

        Returns:
            ndarray start_state: start state vector
        """
        
        start_state  = np.zeros(STATE_SIZE)
        start_state[0] = 1
        start_state[2] = 1
        return start_state

    def step(self, action):
        """Takes one step in the environment

        Args:
            action (int): one hot encoded action

        Returns:
            ndarray:next_state, float:reward : next state and reward
        """

        # Run simulation
        action = ACTION_MAPPING[action]
        state = self.state.copy()
        _state = self.simulate_people(state)
        __state = state.copy()

        # Take action and get next state
        next_state = self.state_transition(__state, action)

        # Observer reward given state, action, next state
        reward = self.reward_func(_state, action, next_state)
        self.state = next_state.copy()
        return next_state, reward

    def reward_func(self, state, action, new_state):
        """Calculates reward given state, action, and new state

        Args:
            state (ndarray): snapshot of environment
            action (int): verbal converted one hot encoded action
            new_state (ndarray): new state

        Returns:
            float:reward : reward
        """

        # Extract old state info
        old_a_floor, old_a_door, old_b_floor, old_b_door = state[:4]
        old_passengers = state[4:].reshape(-1, 3)

        # Extract New state info
        new_a_floor, new_a_door, new_b_floor, new_b_door = new_state[:4]
        new_passengers = new_state[4:].reshape(-1, 3)
        
        reward = 0
        
        # Evaluate elevator movement and door operations
        for i, (old_p, new_p) in enumerate(zip(old_passengers, new_passengers)):
            old_call, old_exit, old_loc = old_p
            new_call, new_exit, new_loc = new_p


            # MOVEMENT FOR A
            if action[0] == 'UP' or action[0] == 'DOWN':
                
                # If in A, and elevator moved closer to exit floor, add reward, else penalty
                if new_loc == 1:
                    if abs(new_a_floor - new_exit) < abs(old_a_floor - old_exit):
                        reward += MOVEMENT_REWARD
                    else:  
                        reward -= MOVEMENT_PENALTY

                # If a passenger is waiting and Elevator A moved closer to them, give reward, else penalty
                if new_loc == 0 and new_call != 0 and new_exit != 0:
                    if abs(new_a_floor - new_call) < abs(old_a_floor - old_call):
                        reward += MOVEMENT_REWARD
                    else:  
                        reward -= MOVEMENT_PENALTY

            # MOVEMENT FOR B
            if action[1] == 'UP' or action[1] == 'DOWN':
                
                # If in B, and elevator moved closer to exit floor, add reward, else penalty
                if new_loc == 2:
                    if abs(new_b_floor - new_exit) < abs(old_b_floor - old_exit):
                        reward += MOVEMENT_REWARD
                    else:  
                        reward -= MOVEMENT_PENALTY

                # If a passenger is waiting and Elevator B moved closer to them, give reward, else penalty
                if new_loc == 0 and new_call != 0 and new_exit != 0:
                    if abs(new_b_floor - new_call) < abs(old_b_floor - old_call):
                        reward += MOVEMENT_REWARD
                    else:  
                        reward -= MOVEMENT_PENALTY

            # DOORS A
            if action[0] == 'DOORS':
            
                # If was waiting before, now in elevator, give reward
                if old_loc == 0 and old_call != 0 and old_exit != 0 and new_loc == 1:
                    reward += PASSENGER_PICKUP_REWARD 
                
                # If was waiting before, and still waiting, penalty
                if old_loc == 0 and old_call != 0 and old_exit != 0 and new_loc == 0:
                    reward -= DOOR_OPEN_PENALTY

                # If was in elevtaor and now not in elevator, give reward
                if old_loc == 1 and new_loc == 0:
                    reward += PASSENGER_DROP_OFF_REWARD

            # DOORS B
            if action[1] == 'DOORS':
            
                # If was waiting before, now in elevator, give reward
                if old_loc == 0 and old_call != 0 and old_exit != 0 and new_loc == 2:
                    reward += PASSENGER_PICKUP_REWARD 
                
                # If was waiting before, and still waiting, penalty
                if old_loc == 0 and old_call != 0 and old_exit != 0 and new_loc == 0:
                    reward -= DOOR_OPEN_PENALTY

                # If was in elevtaor and now not in elevator, give reward
                if old_loc == 2 and new_loc == 0:
                    reward += PASSENGER_DROP_OFF_REWARD

            # HOLD A
            if action[0] == 'HOLD':
            
                # If held doors and was wiating and now in elevator, reward
                if old_loc == 0 and new_loc == 1:
                    reward += PASSENGER_PICKUP_REWARD 
                
                # If held doors, and still waiting, penalty
                if old_loc == 0 and old_call != 0 and old_exit != 0 and new_loc == 0:
                    reward -= DOOR_HOLD_PENALTY

                # If was in elevaotr, held doors, and off elevator , reward
                if old_loc == 1 and new_loc == 0:
                    reward += PASSENGER_DROP_OFF_REWARD

            # HOLD B
            if action[1] == 'HOLD':
            
                # If held doors and was wiating and now in elevator, reward
                if old_loc == 0 and new_loc == 2:
                    reward += PASSENGER_PICKUP_REWARD 
                
                # If held doors, and still waiting, penalty
                if old_loc == 0 and old_call != 0 and old_exit != 0 and new_loc == 0:
                    reward -= DOOR_HOLD_PENALTY

                # If was in elevaotr, held doors, and off elevator , reward
                if old_loc == 2 and new_loc == 0:
                    reward += PASSENGER_DROP_OFF_REWARD

        return reward

    def simulate_people(self, state):
        """Simulates people entering the simulation

        Args:
            state (ndarray): snapshot of environment

        Returns:
            ndarray:new_state : state with simulated people
        """

        elevator_A_state = state[0:2]  # State of Elevator A
        elevator_B_state = state[2:4]  # State of Elevator B
        passengers = state[4:]  # Passenger information in the format [call_floor, exit_floor, location, ...]
        max_people = MAX_PEOPLE_IN_SIM # Max people at any time
        num_attributes_per_passenger = 3  # Each passenger has 3 attributes (call floor, exit floor, location)

        # Count current valid people (ignoring those who have exited and are marked with zeros in all fields)
        passenger_matrix = passengers.reshape(-1, num_attributes_per_passenger)
        valid_passengers = np.any(passenger_matrix > 0, axis=1)
        num_current_people = np.sum(valid_passengers)

        # Simulate new arrivals only if there's room for more people
        if num_current_people < max_people:
            for _ in range(self.timestep):
                if np.random.random() < self.arrival_rate and num_current_people < max_people:
                    empty_indices = np.where(~valid_passengers)[0]
                    if empty_indices.size > 0:
                        idx = empty_indices[0]
                        call_floor = np.random.choice(self.start_floors, p=self.start_prob)
                        exit_floor = np.random.choice(self.exit_floors, p=self.exit_prob)
                        passenger_matrix[idx] = [call_floor, exit_floor, 0]  # 0 denotes WAITING location
                        valid_passengers[idx] = True  # Mark this passenger as valid
                        num_current_people += 1
                        self.entered_sim_count += 1
        else:
            self.max_bandwidth_count += 1
        
        updated_passengers = passenger_matrix.flatten()
        self.current_time += self.timestep
        new_state = np.concatenate((elevator_A_state, elevator_B_state, updated_passengers))
        
        return new_state
    
    def state_transition(self, state, action):
        """
        Transitions the environment to a new state given actions.
        Handles up to 10 passengers within a flat list state of 34 elements.

        Args:
            state (list): snapshot of environment state including elevators and passengers
            action (list): actions chosen for each elevator (e.g., ['UP', 'HOLD'])

        Returns:
            list: new snapshot of environment
        """
        
        # Elevator states
        a_floor, a_door = state[0], state[1]
        b_floor, b_door = state[2], state[3]

        # Apply actions to elevators
        state[0], state[1] = self.update_elevator_state(a_floor, a_door, action[0])
        state[2], state[3] = self.update_elevator_state(b_floor, b_door, action[1])


        # Update each passenger's state
        for i in range(4, len(state), 3):
            call_floor, exit_floor, location = state[i], state[i+1], state[i+2]
            state[i], state[i+1], state[i+2] = self.update_passenger_state(
                call_floor, exit_floor, location, (state[0], state[1]), (state[2], state[3]))

        return state

    def update_elevator_state(self, floor, door, action):
        """
        Update the state of an elevator based on the action.
        """
        if action == 'UP' and not door and floor < self.num_floors:
            return [floor + 1, door]
        elif action == 'DOWN' and not door and floor > 1:
            return [floor - 1, door]
        elif action == 'DOORS':
            return [floor, 1 - door]
        elif action == 'HOLD':
            return [floor, door]
        return [floor, door]

    def update_passenger_state(self, call_floor, exit_floor, location, elevator_A, elevator_B):
        """
        Update the state of a passenger based on the elevator states and actions.
        """
        a_floor, a_door = elevator_A
        b_floor, b_door = elevator_B

        # Passenger is waiting and Elevator A door is open and not full
        if location == 0 and a_door == 1 and call_floor == a_floor and self.a_occupancy < 2:
            self.a_occupancy += 1  # Increase occupancy of Elevator A
            return [call_floor, exit_floor, 1]  # Enter Elevator A
                
        # Passenger is in Elevator A and door is open at their exit floor
        elif location == 1 and a_door == 1 and exit_floor == a_floor:
            self.a_occupancy -= 1  # Decrease occupancy of Elevator A
            self.exit_count +=1
            return [0, 0, 0]  # Exit Elevator A

        # Passenger is waiting and Elevator B door is open and not full
        elif location == 0 and b_door == 1 and call_floor == b_floor and self.b_occupancy < 2:
            self.b_occupancy += 1  # Increase occupancy of Elevator B
            return [call_floor, exit_floor, 2]  # Enter Elevator B

        # Passenger is in Elevator B and door is open at their exit floor
        elif location == 2 and b_door == 1 and exit_floor == b_floor:
            self.b_occupancy -= 1  # Decrease occupancy of Elevator B
            self.exit_count +=1
            return [0, 0, 0]  # Exit Elevator B

        return [call_floor, exit_floor, location]  # No change if none of the above conditions are met

    def print_state_info(self, action=None, reward=None, verbose=0):
        
        print("TIME:", self.current_time, "ACTION TAKEN: ", 'None' if action is None else ACTION_MAPPING[action])
        print("REWARD: ", 'None' if action is None else reward)

        if verbose > 0:

            # Elevator information
            elev_a = self.state[0:2]  # elevator A info [floor, door status]
            elev_b = self.state[2:4]  # elevator B info [floor, door status]
            ps = self.state[4:]       # all passenger info

            # Create a list of passenger information where each entry is [call floor, exit floor, status]
            p = [ps[i:i+3].astype(int) for i in range(0, len(ps), 3) if ps[i] != 0]  # filter out empty passenger slots


            # Set up floors for visualization
            floors = list(range(1, 7))
            floor_mapping = {floor: [] for floor in floors}

            # Map passengers to their current locations or waiting status
            for passenger in p:
                call, exit, status = passenger
                if status == 0:  # Waiting
                    floor_mapping[call].append(f'W: E {exit}')
                elif status == 1:  # In Elevator A
                    floor_mapping[elev_a[0]].append(f'In A: E {exit}')
                elif status == 2:  # In Elevator B
                    floor_mapping[elev_b[0]].append(f'In B: E {exit}')

            # Create a visualization table for each floor
            n_table = []
            for floor in floors:
                elevator_info = []
                if floor == elev_a[0]:
                    elevator_info.append(f'Elev A: {"Open" if elev_a[1] == 1 else "Closed"}')
                if floor == elev_b[0]:
                    elevator_info.append(f'Elev B: {"Open" if elev_b[1] == 1 else "Closed"}')
                passenger_info = ' | '.join(floor_mapping[floor])
                n_table.append([floor, ', '.join(elevator_info), passenger_info or '-'])

            n_table.reverse()
            # Print the comprehensive table showing floors, elevators, and passengers
            if verbose > 1:
                log('\n' + tabulate(n_table, headers=["Floor", "Elevator", "Passengers"], tablefmt="grid"))
            else:
                print(tabulate(n_table, headers=["Floor", "Elevator", "Passengers"], tablefmt="grid"))

        
if __name__ == '__main__':
    # Init elevator object and breakdown state information
    iterations = 100
    total_rewards = []
    elev = ElevatorEnv()
    elev.print_state_info()
    for i in range(iterations):
        action = np.random.randint(0,16)
        next_state, reward = elev.step(action)
        elev.print_state_info(action, reward,1)
        total_rewards.append(reward)
    
    print("NUMEBER OF PEOPLE WHO FOUND EXIT FLOOR: ", elev.exit_count)
    print("NUMBER OF PEOPLE WHO ENTERED SIMUALTION: ", elev.entered_sim_count)
    print("NUMBER OF TIMES SIM HIT MAX BANDWIDTH: ", elev.max_bandwidth_count)
    print("TOTAL ITERATIONS", iterations)
    print("SIMULATION UTILIZATION", elev.max_bandwidth_count/iterations)
    print(f"Total avg rewards: {sum(total_rewards)/iterations}")



