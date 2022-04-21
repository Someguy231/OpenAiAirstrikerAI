import retro        
import numpy as np  
import cv2          
import neat         
import pickle       


class Worker(object):
    def __init__(self, genome, config):
        self.genome = genome
        self.config = config
        
    def work(self):
        
        self.env = retro.make('Airstriker-Genesis','Level1')
        
        self.env.reset()
        
        ob, _, _, _ = self.env.step(self.env.action_space.sample())
        
        inx = int(ob.shape[0]/8)
        iny = int(ob.shape[1]/8)
        done = False
        
        net = neat.nn.FeedForwardNetwork.create(self.genome, self.config)
        
        fitness = 0
        fitness_current=0  
        fitness_current_max=0
        counter_end=0      
        imgarray = []
        counter=0
        lives=0
        while not done:
            self.env.render()
            ob = cv2.resize(ob, (inx, iny))
            ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
            ob = np.reshape(ob, (inx, iny))
            
            imgarray = np.ndarray.flatten(ob)
            actions = net.activate(imgarray)
            
            ob, rew, done, info = self.env.step(actions)
            
            lives=info["lives"]
            fitness_current += rew
            #fitness += rew
            if fitness_current>fitness_current_max:
                fitness_current_max=fitness_current
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
        return fitness

def eval_genomes(genome, config):
    
    worky = Worker(genome, config)
    return worky.work()


config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, 
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-feedforward')

p = neat.Population(config)
p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-85')
p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)
p.add_reporter(neat.Checkpointer(10))


if __name__ ==  '__main__':
    pe = neat.ParallelEvaluator(10, eval_genomes)
    winner = p.run(pe.evaluate)
    #with open('winner.pkl', 'wb') as output:
     #   print('writing winner gen to ', output)
      #  pickle.dump(winner, output, 1)


