
#--------------------- Settings ------------------------
NUMBER_GAMES = 190
GAMMA = 0.99

#----------- Model ----------------------------------

EPSILON_START = 1
EPSILON_END = 0.1 #should be twice of training length
EPSILON_DECAY = 1e-4

NEURONS_1=128
NEURONS_2=128
LEARNING_RATE=0.001
OPTIMIZER='Adam' #SGD
LOSS='mse'       #
BATCH_SIZE =  16 #32 was original

#----------------------Others ---------------------------
MEMORY_SIZE=1000000 #buffer reply memory length
REPLACE=5 # how many games play bofere replace target with policy -- It is calculated based on training lenthg * REPLACE
