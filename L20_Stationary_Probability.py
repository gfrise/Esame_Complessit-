import sympy as sp

def compute_transition_matrix_power():
    # Definisci i simboli
    alfa, beta, n = sp.symbols('alfa beta n', real=True, nonnegative=True)
    
    # Matrice di transizione PP
    PP = sp.Matrix([
        [1 - alfa, alfa],
        [beta, 1 - beta]
    ])
    
    # Calcola autovalori e autovettori
    eigenvectors = PP.eigenvects()
    
    # Estrai autovalori e autovettori
    lambda1 = eigenvectors[0][0]  # 1
    lambda2 = eigenvectors[1][0]  # 1 - alfa - beta
    u1 = eigenvectors[0][2][0]    # [1, 1]
    u2 = eigenvectors[1][2][0]    # [-alfa/beta, 1]
    
    # Costruisci Umat (matrice degli autovettori)
    Umat = sp.Matrix.hstack(u1, u2)
    
    # Calcola Vmat (inversa di Umat)
    Vmat = Umat.inv()
    
    # Matrice diagonale LAMBDAn
    LAMBDAn = sp.Matrix.diag(sp.Pow(lambda1, n), sp.Pow(lambda2, n))
    
    # Calcola PP^n = Umat * LAMBDAn * Vmat
    PPn = Umat @ LAMBDAn @ Vmat
    PPn_simplified = PPn.simplify()
    
    return PPn_simplified

# Esegui il calcolo
result = compute_transition_matrix_power()
sp.pprint(result)

# ⎡   β        α⋅(1 - α - β)ⁿ       α        α⋅(1 - α - β)ⁿ  ⎤
# ⎢  ───── + ────────────────    ─────── - ────────────────⎥
# ⎢  α + β       α + β          α + β         α + β        ⎥
# ⎢                                                         ⎥
# ⎢      β⋅(1 - α - β)ⁿ        α        β⋅(1 - α - β)ⁿ     ⎥
# ⎢   - ──────────────── + ───────   ──────────────── + ───────⎥
# ⎣        α + β           α + β         α + β          α + β  ⎦

#Questo codice calcola la matrice di transizione dopo n passi 
# per un canale binario, utile per studiare distribuzioni stazionarie 
# in catene di Markov.