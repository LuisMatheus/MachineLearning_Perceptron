import numpy as np

class Perceptron:
    def __init__(self, input_values , output_values , learning_rate , activation_function):
        self.input_values = input_values
        self.output_values = output_values
        self.learning_rate = learning_rate
        self.activation_function = activation_function
        self.W = np.random.rand(input_values.shape[1])
        self.theta = np.random.rand(1)[0]
    
    def train(self):
        epochs = 1
        error = True
        
        print(f'Actual W: {self.W}, Actual Theta: {self.theta}')
        while error and epochs <= 10000:
            error = False
            print('')
            print(f'Epoch {epochs}')
            for x,d in zip(self.input_values,self.output_values):
                u =  np.dot(x,self.W) - self.theta
                y = self.activation_function.g(u)
                print(f'input: {x} , output: {y} , expected: {d}')
                if y != d:
                    print('Output is different from expected recalculating W')
                    self.W = self.W + self.learning_rate * (d - y) * x
                    self.theta = self.theta + self.learning_rate * (d - y) * -1
                    print(f'New W: {self.W}, New Theta: {self.theta}')
                    print('')
                    error = True
                    break
            epochs +=1
            
    def evaluate(self,input_value):
        u = np.dot(input_value,self.W) - self.theta
        return self.activation_function.g(u)