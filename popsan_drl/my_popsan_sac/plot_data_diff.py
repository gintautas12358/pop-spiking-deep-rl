#
#BSD 3-Clause License
#
#
#
#Copyright 2022 fortiss, Neuromorphic Computing group
#
#
#All rights reserved.
#
#
#
#Redistribution and use in source and binary forms, with or without
#
#modification, are permitted provided that the following conditions are met:
#
#
#
#* Redistributions of source code must retain the above copyright notice, this
#
#  list of conditions and the following disclaimer.
#
#
#
#* Redistributions in binary form must reproduce the above copyright notice,
#
#  this list of conditions and the following disclaimer in the documentation
#
#  and/or other materials provided with the distribution.
#
#
#
#* Neither the name of the copyright holder nor the names of its
#
#  contributors may be used to endorse or promote products derived from
#
#  this software without specific prior written permission.
#
#
#
#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#
#AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#
#IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#
#DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
#
#FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#
#DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#

from cProfile import label
import matplotlib.pyplot as plt



# file_path_g = "agent_guiding_40e_single_goal.txt"
# file_path_ga = "agent_guiding_activity_40e_single_goal.txt"
# file_path_gca = "agent_guiding_corner_activity_40e_single_goal.txt"
# file_path_vae = "agent_guiding_vae_40e_single_goal.txt"

# no_coord_
# file_path_g = "agent_guiding_no_coord_40e_single_goal.txt"
# file_path_ga = "agent_guiding_activity_no_coord_40e_single_goal.txt"
# file_path_gca = "agent_guiding_corner_activity_no_coord_40e_single_goal.txt"
# file_path_vae = "agent_guiding_vae_no_coord_40e_single_goal.txt"

# rand
# file_path_g = "agent_guiding_200e_rand_goal.txt"
# file_path_ga = "agent_guiding_activity_200e_rand_goal.txt"
# file_path_gca = "agent_guiding_corner_activity_200e_rand_goal.txt"
# file_path_vae = "agent_guiding_vae_200e_rand_goal.txt"

# rand no_coord
file_path_g = "agent_guiding_no_coord_200e_rand_goal.txt"
file_path_ga = "agent_guiding_activity_no_coord_200e_rand_goal.txt"
file_path_gca = "agent_guiding_corner_activity_no_coord_200e_rand_goal.txt"
file_path_vae = "agent_guiding_vae_no_coord_200e_rand_goal.txt"

results = {"mean pixel distance": file_path_g, "mean activity coordinates": file_path_ga, "corner extraction": file_path_gca, "vae latent space": file_path_vae}

def read_data(file_path):
    agents_accuracy = []
    with open(file_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            str_array = line.split(",")
            int_array = list(map(float, str_array))
            agents_accuracy.append(int_array)

    return agents_accuracy



def get_min_max(agents_accuracy):
    min_run_length = 300
    max_run_length = 0
    for a in agents_accuracy:
        if len(a) < min_run_length:
            min_run_length = len(a)

        if len(a) > max_run_length:
            max_run_length = len(a)

    return min_run_length, max_run_length


def process(agents_accuracy, max_run_length):

    processed_runs = []

    failed_tries = 0
    for a in agents_accuracy:
        size = len(a)
        if a[size - 1] > 2:
            failed_tries += 1
            continue

        if size < max_run_length:
            for i in range(max_run_length - size):
                a.append(a[size-1])

        processed_runs.append(a)

    print("Success rate:", (len(agents_accuracy) - failed_tries) * 1.0 / len(agents_accuracy))

    return processed_runs

def mean_run(agents_accuracy, max_run_length):
    runs_mean = []
    for step in range(max_run_length):
        sum = 0
        for r in agents_accuracy:
            if step >= len(r):
                continue

            sum += r[step]
        
        runs_mean.append(1.0 * sum / len(agents_accuracy))

    return runs_mean

def plot_data(agents):

    for k, a in agents.items():

        plt.xlabel('episode steps')
        plt.ylabel('pixel error')

        min_run_length, max_run_length = get_min_max(a)

        a = process(a, max_run_length)

        runs_mean = mean_run(a, max_run_length)

        plt.plot(runs_mean, label=k)

    plt.grid()
    plt.legend()
    plt.show()

agents = {}
for k, p in results.items():
    agents_accuracy = read_data(p)
    agents[k] = agents_accuracy

plot_data(agents)