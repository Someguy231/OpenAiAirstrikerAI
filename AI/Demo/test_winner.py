import retro
import numpy as np
import cv2 
import neat
import pickle
#80 generations winner
env = retro.make('Airstriker-Genesis','Level1')

imgarray = []

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-feedforward')

p = neat.Population(config)


p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)


with open('winner.pkl', 'rb') as input_file:
    genome = pickle.load(input_file)


ob = env.reset()
ac = env.action_space.sample()

inx, iny, inc = env.observation_space.shape

inx = int(inx/8)
iny = int(iny/8)

net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)

current_max_fitness = 0
fitness_current = 0
frame = 0
counter = 0
counter_end=0
fitness = 0
done = False

while not done:
    
    env.render()
    frame += 1

    ob = cv2.resize(ob, (inx, iny))
    ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
    ob = np.reshape(ob, (inx,iny))

    for x in ob:
        for y in x:
            imgarray.append(y)
    
    action = net.activate(imgarray)
    
    ob, rew, done, info = env.step(action)
    imgarray.clear()
    
    lives=info["lives"]
    fitness_current += rew
    
    
    if fitness_current>current_max_fitness:
        current_max_fitness=fitness_current
        counter_end=0
    else:
        counter_end+=1
        counter+=1
        #counter+=1
            
    if lives<3 or counter_end==1500:
        done = True
        if counter_end==1500 and fitness_current+counter>=15000:
            fitness_current+=10000
        fitness=fitness_current    
        fitness+=counter
                
        print(fitness)
