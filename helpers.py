"""Includes helper modules"""
import argparse
import matplotlib.pyplot as plt

class ParseCMD:
    """Parses command line"""
    
    def __init__(self):
        parser = argparse.ArgumentParser(description='Performs Deep Q Learning')
        parser.add_argument('--num_floors', '-nf', type=int, help='Number of Floora', default=6)
        parser.add_argument('--start_floors', '-sf', type=int, help='start floors', default=[1], nargs='+')
        parser.add_argument('--start_prob', '-sp', type=int, help='start floors chance', default=[1], nargs='+')
        parser.add_argument('--exit_floors', '-ef', type=int, help='exit floors', default=[2,3,4,5,6], nargs='+')
        parser.add_argument('--exit_prob', '-ep', type=int, help='exit floors chance', default=[.20,.20,.20,.20,.20], nargs='+')
        parser.add_argument('--floors', '-f', type=int, help='floors', default=[1,2,3,4,5,6], nargs='+')
        parser.add_argument('--floors_zero', '-fz', type=int, help='floors including zero', default=[0,1,2,3,4,5,6], nargs='+')
        parser.add_argument('--iterations','-i' ,type=int, help='number of iterations', default=1000)
        parser.add_argument('--timestep', '-t', type=int, help='timestep', default=5)
        parser.add_argument('--epsilon', '-e', type=float, help='exploration rate', default=1)
        parser.add_argument('--gamma', '-g', type=float, help='discount factor', default=.9)
        parser.add_argument('--alpha', '-a', type=float, help='learning rate', default=.05)
        parser.add_argument('--arrival_rate', '-ar', type=float, help='arrival rate', default=.1)
        parser.add_argument('--epochs', type=int, help='epochs', default=800)
        parser.add_argument('--max_steps', type=int, help='max steps', default=5)
        parser.add_argument('--buffer_capacity', type=int, help='max replay buffer size', default=100)
        parser.add_argument('--batch_size', type=int, help='size of batch', default=2)
        parser.add_argument('--epsilon_decay', type=float, help='epsilon decay', default=.999)
        parser.add_argument('--min_epsilon', type=float, help='minimum epsilon', default=.5)
        parser.add_argument('--verbose', '-v', type=float, help='verbose', default=0)
        args = parser.parse_args()
        self.num_floors = args.num_floors
        self.start_floors = args.start_floors
        self.start_prob = args.start_prob
        self.exit_floors = args.exit_floors
        self.exit_prob = args.exit_prob
        self.floors = args.floors
        self.floors_zero = args.floors_zero
        self.iterations = args.iterations
        self.timestep = args.timestep
        self.alpha = args.alpha
        self.gamma = args.gamma
        self.epsilon = args.epsilon
        self.arrival_rate = args.arrival_rate
        self.epochs = args.epochs
        self.max_steps = args.max_steps
        self.buffer_capacity = args.buffer_capacity
        self.batch_size = args.batch_size
        self.epsilon_decay = args.epsilon_decay
        self.min_epsilon = args.min_epsilon
        self.verbose = args.verbose
        self.all = args
    
class Grapher:
    """Graphs agent data"""

    def graph_avg_rewards(agent):
        """Graphs average rewards"""

        plt.plot([*range(1, agent.iterations)], agent.avg_rewards)
        plt.title("Average Rewards per Iteration")
        plt.xlabel("Iterations (x) [#]")
        plt.ylabel("Average Rewards (y) [#]")
        plt.grid(True)
        plt.savefig('./output/average_rewards.png')
        plt.show()

    def graph_exact_rewards(agent):
        """Graphs average rewards"""

        plt.plot([*range(1, agent.iterations)], agent.exact_rewards)
        plt.title("Exact Rewards per Iteration")
        plt.xlabel("Iterations (x) [#]")
        plt.ylabel("Rewards (y) [#]")
        plt.grid(True)
        plt.savefig('./output/exact_rewards.png')
        plt.show()

    def display_agent_data(agent):
        """Displays agent data"""
        print("--------------------------------------")
        print(f"Iterations used: {agent.iterations}")
        print(f"Final epsilon: {agent.epsilon}, Epsilon decay: {agent.decay}", )
        print(f"Number of people who exited: {agent.env.exit_count}")
        print(f"Number who entered the simulation: {agent.env.entered_sim_count}")
        print(f"Average reward {round(sum(agent.exact_rewards)/agent.iterations,4)}")
        print("--------------------------------------")
        