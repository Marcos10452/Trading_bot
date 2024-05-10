
#--------------------- Settings ------------------------
NUMBER_GAMES = 200
BATCH_SIZE =  32 #32 was original
GAMMA = 0.999

#----------- Model ----------------------------------
LEARNING_RATE=0.001
EPSILON_START = 1
EPSILON_END = 0.01
EPSILON_DECAY = 1e-3
NEURONS_1=256
NEURONS_2=256
OPTIMIZER='Adam' #SGD
LOSS='mse'       #


#----------------------Others ---------------------------
MEMORY_SIZE=1000000
REPLACE=1000
