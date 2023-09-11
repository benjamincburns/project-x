import os,sys
import random
random.seed(0)
import numpy as np
np.random.seed(0)
import torch
torch.manual_seed(0)
import logging
logging.basicConfig(
    format="%(asctime)s %(levelname)s - %(processName)s:%(process)d - %(message)s",
    level=logging.INFO
)

from distrib_rl.policy_optimization.distrib_policy_gradients import Server
from distrib_rl.experiments import ExperimentManager
import traceback

from project_x.register_customizations import register_customizations

def main():
    if len(sys.argv) == 1:
        experiment_path = "resources/experiments/test_experiments/project_x_mvp.json"
    if len(sys.argv) == 2:
        experiment_path = sys.argv[1]
        if not os.path.exists(experiment_path):
            raise FileNotFoundError(f"No experiment file found at location '{experiment_path}'")

    register_customizations()
    server = Server()
    experiment_manager = ExperimentManager(server)
    experiment_manager.load_experiment(experiment_path)

    try:
        experiment_manager.run_experiments()
    except:
        print("\nFAILURE IN SERVER!\n")
        print(traceback.format_exc())
    finally:
        try:
            server.cleanup()
        except:
            print("\n!!!CRITICAL FAILURE!!!\nUNABLE TO SET REDIS STATE TO STOPPING AFTER EXCEPTION IN CLIENT\n")
            print(traceback.format_exc())

if __name__ == "__main__":
    main()
