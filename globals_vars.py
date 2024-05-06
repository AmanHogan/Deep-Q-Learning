"""contains globals"""

STATE_SIZE = 34
"""The size of the state vector"""

ACTION_SIZE = 16
"""The size of the action vector"""

MOVEMENT_REWARD = 10
""" Reward for elevator moving the right direction towards a passenger's target floor or towards a call """

MOVEMENT_PENALTY = 1/5
""" Penalty for elevator moving away from a passenger's target floor or opposite to a call direction"""

PASSENGER_PICKUP_REWARD = 10
"""Reward for picking up a passenger, highly valued to encourage prompt responses to calls"""

DOOR_OPEN_PENALTY = 2/5
""" Penalty for opening the door when there is no one to pick up or drop off at that floor"""

PASSENGER_DROP_OFF_REWARD = 10
""" Reward for dropping off a passenger on their desired floor, incentivizing efficient route planning"""

DOOR_HOLD_PENALTY = 2/5
""" Penalty for holding the door open when it's not necessary, promoting energy efficiency and time management """

MAX_PEOPLE_IN_SIM = 10


ACTION_MAPPING = {
    0: ('UP', 'UP'),
    1: ('UP', 'DOWN'),
    2: ('UP', 'HOLD'),
    3: ('UP', 'DOORS'),
    4: ('DOWN', 'UP'),
    5: ('DOWN', 'DOWN'),
    6: ('DOWN', 'HOLD'),
    7: ('DOWN', 'DOORS'),
    8: ('HOLD', 'UP'),
    9: ('HOLD', 'DOWN'),
    10: ('HOLD', 'HOLD'),
    11: ('HOLD', 'DOORS'),
    12: ('DOORS', 'UP'),
    13: ('DOORS', 'DOWN'),
    14: ('DOORS', 'HOLD'),
    15: ('DOORS', 'DOORS')
}
"""One hot encoding for the elevator """