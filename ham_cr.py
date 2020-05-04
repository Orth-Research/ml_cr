# Module creates crystal field Hamiltonian matrix for point group (PG) Oh and J=4.
# Number of crystal field variables is 2: {x0, x1}.
import numpy as np
def ham_cr_PG_Oh_J_4(x0, x1):
	J = 4
	dim=2*J+1
	ham = np.arange(dim*dim, dtype=np.float)
	ham = ham.reshape(dim,dim)
	ham[0][0] = 0.2860387767736777*x0 + 2.1507845372965946*x0*x1 - 0.2860387767736777*x0*np.abs(x1)
	ham[0][1] = 0.
	ham[0][2] = 0.
	ham[0][3] = 0.
	ham[0][4] = -3.5897581584785954*x0 + 1.285339605745463*x0*x1 + 3.5897581584785954*x0*np.abs(x1)
	ham[0][5] = 0.
	ham[0][6] = 0.
	ham[0][7] = 0.
	ham[0][8] = 0.
	ham[1][0] = 0.
	ham[1][1] = -1.2156648012881304*x0 - 3.226176805944892*x0*x1 + 1.2156648012881304*x0*np.abs(x1)
	ham[1][2] = 0.
	ham[1][3] = 0.
	ham[1][4] = 0.
	ham[1][5] = 0.5675906014982022*x0 + 2.0323003604892547*x0*x1 - 0.5675906014982022*x0*np.abs(x1)
	ham[1][6] = 0.
	ham[1][7] = 0.
	ham[1][8] = 0.
	ham[2][0] = 0.
	ham[2][1] = 0.
	ham[2][2] = 1.5732132722552274*x0 - 1.6899021364473241*x0*x1 - 1.5732132722552274*x0*np.abs(x1)
	ham[2][3] = 0.
	ham[2][4] = 0.
	ham[2][5] = 0.
	ham[2][6] = 3.003407156123616*x0 + 2.3044120042463514*x0*x1 - 3.003407156123616*x0*np.abs(x1)
	ham[2][7] = 0.
	ham[2][8] = 0.
	ham[3][0] = 0.
	ham[3][1] = 0.
	ham[3][2] = 0.
	ham[3][3] = 0.07150969419341943*x0 + 1.3826472025478107*x0*x1 - 0.07150969419341943*x0*np.abs(x1)
	ham[3][4] = 0.
	ham[3][5] = 0.
	ham[3][6] = 0.
	ham[3][7] = 0.5675906014982022*x0 + 2.0323003604892547*x0*x1 - 0.5675906014982022*x0*np.abs(x1)
	ham[3][8] = 0.
	ham[4][0] = -3.5897581584785954*x0 + 1.285339605745463*x0*x1 + 3.5897581584785954*x0*np.abs(x1)
	ham[4][1] = 0.
	ham[4][2] = 0.
	ham[4][3] = 0.
	ham[4][4] = -1.4301938838683885*x0 + 2.7652944050956214*x0*x1 + 1.4301938838683885*x0*np.abs(x1)
	ham[4][5] = 0.
	ham[4][6] = 0.
	ham[4][7] = 0.
	ham[4][8] = -3.5897581584785954*x0 + 1.285339605745463*x0*x1 + 3.5897581584785954*x0*np.abs(x1)
	ham[5][0] = 0.
	ham[5][1] = 0.5675906014982022*x0 + 2.0323003604892547*x0*x1 - 0.5675906014982022*x0*np.abs(x1)
	ham[5][2] = 0.
	ham[5][3] = 0.
	ham[5][4] = 0.
	ham[5][5] = 0.07150969419341943*x0 + 1.3826472025478107*x0*x1 - 0.07150969419341943*x0*np.abs(x1)
	ham[5][6] = 0.
	ham[5][7] = 0.
	ham[5][8] = 0.
	ham[6][0] = 0.
	ham[6][1] = 0.
	ham[6][2] = 3.003407156123616*x0 + 2.3044120042463514*x0*x1 - 3.003407156123616*x0*np.abs(x1)
	ham[6][3] = 0.
	ham[6][4] = 0.
	ham[6][5] = 0.
	ham[6][6] = 1.5732132722552274*x0 - 1.6899021364473241*x0*x1 - 1.5732132722552274*x0*np.abs(x1)
	ham[6][7] = 0.
	ham[6][8] = 0.
	ham[7][0] = 0.
	ham[7][1] = 0.
	ham[7][2] = 0.
	ham[7][3] = 0.5675906014982022*x0 + 2.0323003604892547*x0*x1 - 0.5675906014982022*x0*np.abs(x1)
	ham[7][4] = 0.
	ham[7][5] = 0.
	ham[7][6] = 0.
	ham[7][7] = -1.2156648012881304*x0 - 3.226176805944892*x0*x1 + 1.2156648012881304*x0*np.abs(x1)
	ham[7][8] = 0.
	ham[8][0] = 0.
	ham[8][1] = 0.
	ham[8][2] = 0.
	ham[8][3] = 0.
	ham[8][4] = -3.5897581584785954*x0 + 1.285339605745463*x0*x1 + 3.5897581584785954*x0*np.abs(x1)
	ham[8][5] = 0.
	ham[8][6] = 0.
	ham[8][7] = 0.
	ham[8][8] = 0.2860387767736777*x0 + 2.1507845372965946*x0*x1 - 0.2860387767736777*x0*np.abs(x1)
	return ham
