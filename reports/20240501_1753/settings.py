
#--------------------- Settings ------------------------
NUMBER_GAMES = 900
GAMMA = 0.999

#----------- Model ----------------------------------

EPSILON_START = 1
EPSILON_END = 0.0001 #should be twice of training length
EPSILON_DECAY = 1e-3

NEURONS_1=512
NEURONS_2=512
LEARNING_RATE=0.001
OPTIMIZER='Adam' #SGD
LOSS='mse'       #
BATCH_SIZE =  32 #32 was original

#----------------------Others ---------------------------
MEMORY_SIZE=1000000 #buffer reply memory length
REPLACE=10 # how many games play bofere replace target with policy -- It is calculated based on training lenthg * REPLACE
