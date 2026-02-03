import numpy as np
import matplotlib.pyplot as plt

# --- Parametri ---
T = 10**5         # lunghezza serie temporale
dt = 0.1
gamma = 0.1
mu = 0.0
sigma = 1.0
N_boot = 100      # numero di bootstrap

# --- Genera processo OU ---
x = np.zeros(T)
x[0] = 0.0
for t in range(1, T):
    x[t] = x[t-1] + gamma*(mu - x[t-1])*dt + sigma*np.sqrt(2*dt)*np.random.randn()

# --- Bootstrap ---
# bootstrap_samples = np.random.choice(x, size=(N_boot, T), replace=True)

# # --- PDF: serie originale ---
# plt.hist(x, bins=30, density=True, alpha=0.5, label='Original OU')

# # --- PDF: bootstrap ---
# for i in range(N_boot):
#     plt.hist(bootstrap_samples[i], bins=30, density=True, alpha=0.05, color='orange')

# plt.xlabel('x')
# plt.ylabel('PDF')
# plt.title('PDF OU process vs bootstrap samples')
# plt.legend()
# plt.show()


boots = np.random.choice(x,size=(N_boot,T),replace=True)

plt.hist(x,bins=30,density=False,alpha=0.5,label="Original OU")

for i in range(N_boot):
    plt.hist(boots[i],bins=30,alpha=0.4, density=False,color="orange")

plt.legend()
plt.show()

# import numpy as np
# import matplotlib.pyplot as plt

# T_tot = 10**3
# dt = 0.1
# y = 0.1
# T = int(T_tot/dt)

# x = np.zeros(T)
# x[0] = 0.1
# for i in range(1,T):
#     x[i] = x[i-1] - y*dt*x[i-1] + np.random.normal(0,np.sqrt(dt))

# N = 100
# bootstap_sample = np.random.choice(x, size=(N,T),replace=True)

# for i in range(N):
#     plt.hist(bootstap_sample[i],density=True,alpha=0.5,color="orange")

# plt.hist(x,bins=30,density=True,label="OU")
# plt.show()

