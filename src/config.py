import numpy as np

########################################################################################
##################################### Design Config: ###################################

# Set a random seed?
# np.random.seed(seed=1234)

LEN       = 45     # sequence length
AA_WEIGHT = 1.0    # weight for the aa composition biasing loss term
RM_AA     = ""     # disable specific amino acids from being sampled (ex: '' or C' or 'W,Y,F')
n_models  = 1      # How many structure prediction models to ensemble? [1-5]

# MCMC schedule:
MCMC = {}
MCMC['BETA_START']    = 10     # Energy multiplier for the metropolis criterion
MCMC['N_STEPS']       = 500   # Number of steps for each MCMC optimization
MCMC['COEF']          = 1.25   # Divide BETA by COEF
MCMC['M']             = MCMC['N_STEPS'] // 10    # Adjust beta every M steps

num_simulations     = 1000     # Number of sequences to design

#seed_filepath       = 'trdesign-seeds.txt' # Optionally, start from a .txt file with sequences
seed_filepath       = None # Sample starting sequences 100% at random

#sequence_constraint = '--C------------------------------------------------------C--'
sequence_constraint = None

# Constraint can be specified as an .npz file containing ['dist', 'omega', 'theta', 'phi'] target arrays
#target_motif_path   = 'target_motifs/sso7d/from_pdb/sso7d_model.npz'
target_motif_path   = None

experiment_name = 'proteins_len_%d_%d_steps' %(LEN, MCMC['N_STEPS'])

#############################################################################################################################
#############################################################################################################################


####################################### Constants: #####################################

# These settings are specific to the trRosetta Model implementation (might need to change for AF2)

limits = {
    'dist':  [2,      20   ],
    'omega': [-np.pi, np.pi],
    'theta': [-np.pi, np.pi],
    'phi':   [0,      np.pi]
}

bin_dict_np = {
    'dist':  np.linspace(*limits['dist'],  num = 37)[:-1] + 0.25,     #Inter-residue distances (Angstroms)
    'omega': np.linspace(*limits['omega'], num = 25)[:-1] + np.pi/24, #Omega-angles     (radians)
    'theta': np.linspace(*limits['theta'], num = 25)[:-1] + np.pi/24, #Theta-angles     (radians)
    'phi':   np.linspace(*limits['phi'],   num = 13)[:-1] + np.pi/24, #Phi-angles       (radians)
}

# Add "no-contact" values:
no_contact_value = np.inf
for key in bin_dict_np.keys():
    bin_dict_np[key] = np.insert(bin_dict_np[key], 0, no_contact_value)

print_dist_bins = 0
if print_dist_bins:
    for i, bin_centre in enumerate(bin_dict_np['dist']):
        print('Bin %02d: [%.2f - %.2f] (%.2f)' %(i, bin_centre - 0.25, bin_centre + 0.25, bin_centre))

########################################################################################
################################## AA_type Alphabet: ###################################

ALPHABET_core_str = "ARNDCQEGHILKMFPSTWYV"  #doesn't include '-' gap char

ALPHABET_full_str = ALPHABET_core_str + "-"
MAX_AA_INDEX      = len(ALPHABET_full_str) - 1

ALPHABET_core = np.array(list(ALPHABET_core_str), dtype='|S1').view(np.uint8) #doesn't include '-' gap char
ALPHABET_full = np.array(list(ALPHABET_full_str), dtype='|S1').view(np.uint8)

########################################################################################
##################### Target AA_composition distribution: ##############################

# ALPHABET_core_str = "ARNDCQEGHILKMFPSTWYV"
# Using all of PDB:
native_freq = np.array([0.078926, 0.049790, 0.045148, 0.060338, 0.012613,
                        0.037838, 0.065925, 0.071221, 0.023248, 0.056478,
                        0.093113, 0.059803, 0.020729, 0.041453, 0.046319,
                        0.061237, 0.054742, 0.014891, 0.037052, 0.069127])

# Using PDB filtered to [40-100] residue range (total of 47'693 sequences):
native_freq = np.array([0.075905, 0.070035, 0.039181, 0.045862, 0.023332,
                        0.035662, 0.066048, 0.064150, 0.021644, 0.059121,
                        0.089042, 0.084882, 0.031276, 0.035995, 0.038211,
                        0.060108, 0.053137, 0.008422, 0.026804, 0.071172])

print(" ---- Using PDB [40-100] native frequencies! ----")
########################################################################################
