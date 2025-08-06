from Bio import SeqIO
import numpy as np

fasta_file = "NC_00195.fasta" 

for record in SeqIO.parse(fasta_file, "fasta"):
    print("ID:", record.id)
    print("Descrizione:", record.description)
    print("Lunghezza:", len(record.seq))

    # Converti la sequenza in array di lettere
x_lett = np.array(list(record.seq))

# Stampa le lettere uniche presenti nella sequenza
print(np.unique(x_lett))

# Codifica la sequenza in numeri
x = np.zeros(len(x_lett), dtype=int)

for i in range(len(x)):
    if x_lett[i] == 'A':
        x[i] = 0
    elif x_lett[i] == 'C':
        x[i] = 1
    elif x_lett[i] == 'G':
        x[i] = 2
    else:
        x[i] = 3

n = 4
L = np.array([0, 1, 2, 3])

mat2 = np.zeros((n, n)) # Matrice congiunta

for k in range(len(x) - 1):
    i = x[k]
    j = x[k + 1]
    mat2[i, j] += 1

print(mat2)
mat2 /= (len(x) - 1)
print(mat2)
print(np.sum(mat2, axis=1), np.sum(mat2))
#il primo somma ogni riga, il secondo tutti gli elementi della matrice