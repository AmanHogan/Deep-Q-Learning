from logger.logger import log
from helpers import ParseCMD, Grapher
from dql import DeepQAgent
import time

start_time = time.time()
cmd = ParseCMD()
log(str(cmd.all))

agent = DeepQAgent(cmd.num_floors,cmd.start_floors,cmd.start_prob,cmd.exit_floors,
                     cmd.exit_prob,cmd.floors,
                     cmd.timestep,cmd.alpha,cmd.gamma,cmd.epsilon,
                     cmd.arrival_rate,cmd.epochs, cmd.buffer_capacity,cmd.batch_size,
                     cmd.epsilon_decay, cmd.min_epsilon,cmd.verbose)

agent.train_agent()
print(f"{(time.time() - start_time)} s over {cmd.epochs} iterations" )
print(f"Time per iteration: {(time.time() - start_time)/cmd.epochs}")


Grapher.graph_avg_rewards(agent)
Grapher.graph_exact_rewards(agent)
Grapher.display_agent_data(agent)


