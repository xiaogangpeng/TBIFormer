import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# data=np.load('/home/ericpeng/DeepLearning/Datasets/HumanMotion/Mocap/two_train_4seconds_2.npy',allow_pickle=True)
data = np.load('/home/ericpeng/DeepLearning/Projects/MotionPrediction/MRT_nips2021/mix_data/mix2_10persons.npy', allow_pickle=True)

eg = 800
data_list = data[eg]

# data_list=data_list.reshape(-1,120,15,3)
# data_list=data_list*0.1*1.8/3 # scale
# #no need to scale if using the mix_mocap data
#
# body_edges = np.array(
# [[0,1], [1,2],[2,3],[3,4],
# [4,5],[0,6],[6,7],[7,8],[8,9],[9,10],[0,11],[11,12],[12,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[21,22],[20,23],[13,24],[24,25],[25,26],[26,27],[27,28],[28,29],[27,30]]
# )

'''
if use the 15 joints in common
use=[0,1,2,3,6,7,8,14,16,17,18,20,24,25,27]
data_list=data_list.reshape(-1,120,15,3)
data_list=data_list[:,:,[0,1,4,7,2,5,8,12,15,16,18,20,17,19,21],:]
body_edges = np.array(
[[0,1], [1,2],[2,3],[0,4],
[4,5],[5,6],[0,7],[7,8],[7,9],[9,10],[10,11],[7,12],[12,13],[13,14]]
)
'''


fig = plt.figure(figsize=(10, 4.5))

ax = fig.add_subplot(111, projection='3d')


for eg in range(len(data)):
    data_list = data[eg]
    use = [0, 1, 2, 3, 6, 7, 8, 14, 16, 17, 18, 20, 24, 25, 27]
    data_list = data_list.reshape(-1, 75, 15, 3)
    body_edges = np.array(
        [[0, 1], [1, 2], [2, 3], [0, 4],
         [4, 5], [5, 6], [0, 7], [7, 8], [7, 9], [9, 10], [10, 11], [7, 12], [12, 13], [13, 14]]
    )

    plt.ion()
    length_ = data_list.shape[1]


    i = 0

    p_x = np.linspace(-15, 15, 15)
    p_y = np.linspace(-15, 15, 15)
    X, Y = np.meshgrid(p_x, p_y)

    while i < length_:

        for j in range(data_list.shape[0]):

            x = data_list[j, i, :, 0]
            y = data_list[j, i, :, 1]
            z = data_list[j, i, :, 2]

            ax.plot(z, x, y, 'y.')

            plot_edge = True
            if plot_edge:
                for edge in body_edges:
                    alpha = 1

                    x = [data_list[j, i, edge[0], 0], data_list[j, i, edge[1], 0]]
                    y = [data_list[j, i, edge[0], 1], data_list[j, i, edge[1], 1]]
                    z = [data_list[j, i, edge[0], 2], data_list[j, i, edge[1], 2]]
                    if j == 0:
                        ax.plot(z, x, y, zdir='z', c='green', alpha=alpha)
                    elif j == 1:
                        ax.plot(z, x, y, zdir='z', c='red', alpha=alpha)  # double person
                    else:
                        ax.plot(z, x, y, zdir='z', c='red', alpha=alpha)  # double person

            ax.set_xlim3d([-3, 3])
            ax.set_ylim3d([-3, 3])
            ax.set_zlim3d([0, 3])
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            plt.title("batch_"+str(eg)+" :"+str(i), y=-0.1)
        plt.pause(0.0001)
        ax.cla()
        i += 1

plt.ioff()
plt.show()
plt.close()