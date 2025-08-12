import numpy as np, matplotlib.pyplot as plt

n, l, m, sd = 10**3, 10**3, 3, 0.14
means = np.empty(n)

for i in range(n):
    x = np.random.normal(m,sd,l)
    means[i] = np.mean(x)

print("Media:",np.mean(means))
print("Sigma stimata:",np.std(means))
print("Sigma teorica:",sd/(l)**0.5)
plt.hist(means,bins=30)
plt.title("PDF Medie")
plt.show()
#campionamente con ripetizione, si ricampiona da un solo campione
#invece di generarne tanti
original = np.random.normal(m,sd,l)

for i in range(n):
    x = np.random.choice(original,l,replace=True)
    # idx = np.random.randint(0,l,l) array di l interi random da 0 a l-1
    # x = original[idx]     
    means[i] = np.mean(x)

print("Medie:",np.mean(means))
print("Sigma stimata:", np.std(means))
print("Sigma teorica:", sd/(l)**0.5)
plt.hist(means,bins=30)
plt.title("PDF Medie con ripetizione")
plt.show()    

#Nel secondo caso il bootstrap eredita maggiormente le caratteristiche 
#del campione originale. Con rimpiazzo [a,a,c],[a,b,b] mentre senza
#rimpiazzo solo permutazioni [a,b,c],[a,c,b]
#Il bootstrap con rimpiazzo riproduce maggior variabilità e incertezza
#che stimano meglio il comportamento della popolazione, ereditando 
#caratteristiche e difetti dell'originale