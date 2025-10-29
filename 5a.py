import numpy as np
from functools import reduce

# Function to compute perceptron output for a single input vector
def perceptron(weight, bias, x):
    model = np.add(np.dot(x, weight), bias)   # Linear combination: (w · x + b)
    print('Model (w.x + b): {}'.format(model))
    
    # Sigmoid activation
    logit = 1 / (1 + np.exp(-model))
    print('Sigmoid Output: {}'.format(logit))
    
    # Threshold to binary output (0 or 1)
    return np.round(logit)

# Function to compute perceptron outputs for a dataset for a given logic gate
def compute(logictype, weightdict, dataset):
    # Extract weights for the given logic type
    weights = np.array([weightdict[logictype][w] for w in weightdict[logictype].keys()])
    
    # Compute perceptron output for each input in the dataset
    output = np.array([perceptron(weights, weightdict['bias'][logictype], val) for val in dataset])
    
    print(f"\nComputed Logic Gate: {logictype.upper()}\n")
    return logictype, output

def main():
    # Define weights and biases for each logic gate
    logic = {
        'logic_and':  {'w0': -0.1, 'w1': 0.2, 'w2': 0.2},
        'logic_or':   {'w0': -0.1, 'w1': 0.7, 'w2': 0.7},
        'logic_not':  {'w0': 0.5,  'w1': -0.7},
        'logic_nand': {'w0': 0.6,  'w1': -0.8, 'w2': -0.8},
        'logic_nor':  {'w0': 0.5,  'w1': -0.7, 'w2': -0.7},
        'logic_xor':  {'w0': -5,   'w1': 20,  'w2': 10},
        'logic_xnor': {'w0': -5,   'w1': 20,  'w2': 10},
        'bias': {
            'logic_and': -0.2,
            'logic_or': -0.1,
            'logic_not': 0.1,
            'logic_xor': 1,
            'logic_xnor': 1,
            'logic_nand': 0.3,
            'logic_nor': 0.1
        }
    }

    # Define dataset (with bias term as first input = 1)
    dataset = np.array([
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1]
    ])

    # Compute AND and NAND gate outputs
    logic_and = compute('logic_and', logic, dataset)
    logic_nand = compute('logic_nand', logic, dataset)
    # You can uncomment others if needed
    # logic_or = compute('logic_or', logic, dataset)
    # logic_nor = compute('logic_nor', logic, dataset)
    # logic_xor = compute('logic_xor', logic, dataset)

    # Function to print results in a formatted truth table
    def template(dataset, name, data):
        print("\nLogic Function: {}".format(name[6:].upper()))
        print("X0\tX1\tX2\tY")
        rows = ["{1}\t{2}\t{3}\t{0}".format(output, *inputs) for inputs, output in zip(dataset, data)]
        for r in rows:
            print(r)

    # Combine all computed gates
    gates = [logic_and, logic_nand]

    # Display results
    for g in gates:
        template(dataset, *g)

if __name__ == '__main__':
    main()
