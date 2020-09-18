import numpy as np

class Perceptron:
    def __init__(self, input_values , output_values , learning_rate , activation_function):
        oneColumns = np.ones((len(input_values), 1 )) * -1
        self.input_values = np.append(oneColumns,input_values,axis=1)
        self.output_values = output_values
        self.learning_rate = learning_rate
        self.activation_function = activation_function
        self.W = np.random.rand(self.input_values.shape[1])
    
    def train(self):
        epochs = 1
        error = True
        
        print(f'Actual W: {self.W}')
        while error and epochs <= 10000:
            error = False
            print('')
            print(f'Epoch {epochs}')
            for x,d in zip(self.input_values,self.output_values):
                u =  np.dot(x,self.W)
                y = self.activation_function.g(u)
                print(f'input: {x} , output: {y} , expected: {d}')
                if y != d:
                    print('Output is different from expected recalculating W')
                    self.W = self.W + self.learning_rate * (d - y) * x
                    print(f'New W: {self.W}')
                    print('')
                    error = True
                    break
            epochs +=1
            
    def evaluate(self,input_value):
        u = np.dot(input_value,self.W)
        return self.activation_function.g(u)