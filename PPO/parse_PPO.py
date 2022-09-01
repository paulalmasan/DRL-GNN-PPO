import argparse
import numpy as np
import os
import matplotlib.pyplot as plt
from operator import add, sub
from scipy.signal import savgol_filter

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parse file and create plots')

    parser.add_argument('-d', help='data file', type=str, required=True, nargs='+')
    args = parser.parse_args()

    aux = args.d[0].split(".")
    aux = aux[1].split("exp")
    differentiation_str = str(aux[1].split("Logs")[0])

    actor_loss = []
    critic_loss = []
    total_loss = []
    eval_rewards = []
    entropy = []

    if not os.path.exists("./Images"):
        os.makedirs("./Images")

    with open(args.d[0]) as fp:
        for line in fp:
            arrayLine = line.split(",")
            if arrayLine[0]==">":
                actor_loss.append(float(arrayLine[1]))
            elif arrayLine[0]=="-":
                total_loss.append(float(arrayLine[1]))
            elif arrayLine[0]==".":
                eval_rewards.append(float(arrayLine[1]))
            elif arrayLine[0]=="<":
                critic_loss.append(float(arrayLine[1]))
            elif arrayLine[0]=="_":
                entropy.append(float(arrayLine[1]))

        plt.plot(actor_loss)
        plt.xlabel("Training Episode")
        plt.ylabel("ACTOR Loss")
        plt.savefig("./Images/ACTORLoss" + differentiation_str)
        plt.close()

        plt.plot(critic_loss)
        plt.xlabel("Training Episode")
        plt.ylabel("CRITIC Loss (MSE)")
        plt.yscale("log")
        plt.savefig("./Images/CRITICLoss" + differentiation_str)
        plt.close()

        plt.plot(eval_rewards)
        plt.xlabel("Training Episode")
        plt.ylabel("Average test reward")
        plt.savefig("./Images/TestReward" + differentiation_str)
        plt.close()

