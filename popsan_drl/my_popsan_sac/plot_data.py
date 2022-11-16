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

import matplotlib.pyplot as plt



# file_path = "collected_data.txt"
# file_path = "agent_guiding_activity_no_coord_40e_single_goal.txt"
file_path = "agent_guiding_vae_no_coord_40e_single_goal.txt"


def read_data(file_path):
    agents_accuracy = []
    with open(file_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            str_array = line.split(",")
            int_array = list(map(float, str_array))
            agents_accuracy.append(int_array)

    return agents_accuracy

def plot_data(agents_accuracy):

    min_run_length = 300
    max_run_length = 0
    for a in agents_accuracy:
        if len(a) < min_run_length:
            min_run_length = len(a)

        if len(a) > max_run_length:
            max_run_length = len(a)

    failed_tries = 0
    for a in agents_accuracy:
        size = len(a)
        if a[size - 1] > 2:
            failed_tries += 1
        if size < max_run_length:
            for i in range(max_run_length - size):
                a.append(a[size-1])

    print("Success rate:", (len(agents_accuracy) - failed_tries) * 1.0 / len(agents_accuracy))

    fig, axs = plt.subplots(2)

    plt.xlabel('episode steps')
    plt.ylabel('pixel error')

    for i, a in enumerate(agents_accuracy):
        axs[0].plot(range(len(a)), a, label = "line" + str(i))

    min_run_length = 300
    max_run_length = 0
    for a in agents_accuracy:
        if len(a) < min_run_length:
            min_run_length = len(a)

        if len(a) > max_run_length:
            max_run_length = len(a)


    runs_mean = []
    for step in range(max_run_length):
        sum = 0
        for r in agents_accuracy:
            if step >= len(r):
                continue

            sum += r[step]
        
        runs_mean.append(1.0 * sum / len(agents_accuracy))

    axs[1].plot(runs_mean)

    axs[1].set_xlabel('episode steps')
    axs[1].set_ylabel('pixel error')

    axs[0].set_xlabel('episode steps')
    axs[0].set_ylabel('pixel error')

    plt.legend()
    plt.show()


agents_accuracy = read_data(file_path)
plot_data(agents_accuracy)