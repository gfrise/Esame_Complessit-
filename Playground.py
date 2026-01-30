# from calendar import c
# import numpy as np
# import matplotlib.pyplot as plt

# uni = np.random.uniform(-10,10,100)
# norm = np.random.normal(0,3,100)
# ins = np.random.randint(-10,10,100)

# plt.hist(uni, label="Uniforme")
# plt.hist(norm, label="Normale")
# plt.hist(ins,label="Interi")
# plt.legend()
# plt.xlabel("valore")
# plt.ylabel("frequenza")
# plt.title("distribuzioni")
# plt.grid(True)
# # plt.show()

# def wiener(n):
#     x = np.zeros(n)
#     x[0]=0.1

#     for i in range(1,n):
#         x[i]=x[i-1]+np.random.normal()

#     return x

# def gbm(n):
#     x = np.zeros(n)
#     x[0]=0
#     dt = 0.1
#     for i in range(1,n):
#         x[i]=x[i-1]+0.17*dt+np.random.normal(0,dt**0.5)

#     return x

# plt.plot(wiener(100), label="wiener")
# plt.plot(gbm(100),label="geometric brownian motion")
# plt.grid(True)
# plt.legend()
# plt.xlabel("tempo")
# plt.ylabel("Prezzo")
# plt.title("Fake stocks")

# # np.random.seed(34)
# # y = np.random.rand()
# # z = np.random.rand(3)
# # t = (100-80)*np.random.rand(20)+80
# # print(t)

# import numpy as np
# from numpy import random

# z = np.array([[1.1,2,3],[4,5,6]])
# # print(z.shape, z.astype('i'), z.reshape(2,3), z.reshape(-1))
# # print(z[1:3], z[-3:-1], z[1:5:2], z[::2])
# # print(z.reshape(2,3,-1))
# # print(np.concatenate((z[0],z[1])))
# # x = np.array([1,2,3,4,34,5,6,77,0,11,11,12])
# # print(np.where(x%2==0))
# # print(np.sort(x))

# # arr = np.array([38,39,40,41,42,43,55,44])
# # newarr = arr[arr>=42]
# # newestarr = arr[arr%2==0]
# # print(newarr, newestarr)

# t = random.randint(3)
# tz = random.rand()
# xt = random.randint(100,size=(5,3))
# # print(t,tz,xt)
# print(np.random.choice([3,5,6,7]), np.random.shuffle([0,1,2,3]),np.random.permutation([1,2,3,4,5]))

# paretiana = np.random.pareto(2,100000)

# plt.hist(paretiana, label="pareto")
# plt.title("Paretiana")
# plt.xlabel("reddito")
# plt.ylabel("frequenza")
# plt.grid(False)
# plt.legend()
# # plt.show()

# import seaborn as sns

# import seaborn as sns
# import numpy as np
# import matplotlib.pyplot as plt

# # Generate some sample data
# data = np.random.pareto(100,10000)

# # Create a histogram using sns.histplot
# sns.displot(data, bins=30, 
#             #kde=True
#             )
# plt.title('Histogram of Sample Data')
# plt.xlabel('Value')
# plt.ylabel('Frequency')
# plt.show()

# x = np.array([0,1,2,3,4])
# y = np.array([0,2,4,6,8])

# ##Matplotlib

# import matplotlib.pyplot as plt
# import numpy as np

# x = np.array([0,5,10,15,20,25,30])
# y = np.array([0,2,4,6,8,10,12])

# plt.plot(x, y, 'o')
# plt.plot(y)
# plt.show()

# plt.plot(y,'o:r')
# plt.show()

# plt.plot(y,'*')
# plt.show()

# plt.plot(y, marker='.')
# plt.show()

# plt.plot(y, marker=',')
# plt.show()

# plt.plot(y, marker='x')
# plt.show()

# plt.plot(y, marker='^')
# plt.show()

import numpy as np
import matplotlib.pyplot as plt

arr1 = np.arange(0,11)
arr2 = np.linspace(-10,1,11)
print(arr1, arr2, len(arr2))
print(np.sum(arr1))

newarr = np.absolute(arr1)
arr3 = np.sum([arr1,arr2])
print(newarr)
print(arr3)
newarr = np.sum([arr1, arr2], axis=1)
newarr = np.cumsum(arr1)

arr = np.array([10, 15, 25, 5])

newarr = np.diff(arr1)

x = np.sin(np.pi/2) #tan #arcsin #sinh
newarr = np.union1d(arr1, arr2)
newarr = np.intersect1d(arr1, arr2, assume_unique=True)


print(arr1-arr2)
