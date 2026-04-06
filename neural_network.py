import numpy as np


class Node:
    def __init__(self, name, type_of_node, func, layer, in_nodes, out_nodes, x=None, y=None, weights=None, bias=None):
        """
        Constructor
        -----------
            - name: the name of the node (for debugging purposes)
            - func: activation function for the node
            - layer: the layer number this node belongs to
            - in_nodes: the nodes connected to this particular node
            - out_nodes: the nodes connected from this particular node
            - x: the input to the node (optional, will be set during forward pass)
            - y: the output of the node (optional, will be set during forward pass)
            - weights: the weights for the connections to the next layer (optional, will be initialized randomly if not provided)
            - bias: the bias term for the node (optional, will be initialized randomly if not provided)
        """
        if type_of_node not in ['input', 'hidden', 'output']:
            raise ValueError("type_of_node must be 'input', 'hidden', or 'output'")
        if type_of_node == 'input':
            func = lambda x: x  # Identity function for input layer
            in_nodes = []  # No incoming connections for input layer
        elif type_of_node == 'output':
            out_nodes = []  # No outgoing connections for output layer
        self.func = func
        self.layer = layer
        self.in_nodes = in_nodes
        self.out_nodes = out_nodes
        self.x = x
        self.y = y
        self.weights = weights if weights is not None else np.random.rand(len(out_nodes))
        self.bias = bias if bias is not None else np.random.rand(1)
        self.name = name

    def activate(self):
        """
        Apply the activation function to the input and return the output
        """
        return self.func(self.x + self.bias)
    
    def calculate_output(self):
        """
        Calculate the output of the node based on the input, weights, and bias
        """
        return self.activate() * self.weights
    

class NeuralNetwork:
    def __init__(self, num_of_nodes_per_layer, activation_function, learning_rate=0.01):
        """
        Constructor
        -----------
            - input_layer: list of nodes in the input layer
            - hidden_layer: list of nodes in the hidden layer
            - output_layer: list of nodes in the output layer
            - learning_rate: the learning rate for training the network
            - num_of_nodes_per_layer: a list containing the number of nodes in each layer
            - activation_function: the activation function to be used in the network
        """
        self.input_layer = None
        self.hidden_layer = None
        self.output_layer = None
        self.num_of_nodes_per_layer = num_of_nodes_per_layer
        self.activation_function = activation_function
        self.learning_rate = learning_rate

        self.normalise_activation_functions()

    def normalise_activation_functions(self):
        """
        Ensure that the activation function is in the correct format (list of functions for each layer)
        """
        def normalise_layer(func, num_nodes):
            if not isinstance(func, list):
                return [func] * num_nodes
            elif len(func) != num_nodes:
                raise ValueError(
                    f"Length of activation function list must match the number of "
                    f"nodes in the layer. Expected {num_nodes}, got {len(func)}."
                    )
            return func
        
        layers = self.num_of_nodes_per_layer
        af = self.activation_function

        if not isinstance(af, list):
            af = [af] * len(layers)

        elif len(af) != len(layers):
            raise ValueError(
                f"Length of activation function list must match the number of "
                f"layers. Expected {len(layers)}, got {len(af)}"
            )

        self.activation_function = [normalise_layer(val, size) for val, size in zip(af, layers)]

    def create_network(self):
        """
        Create the connections between the layers of the network
        """
        output_layer = []
        input_layer = []
        hidden_layer = []
        out_nodes = []
        depth = len(self.num_of_nodes_per_layer)
        node_types = ['input'] + ['hidden'] * (len(self.num_of_nodes_per_layer) - 2) + ['output']
        for node, af, node_type in zip(reversed(self.num_of_nodes_per_layer), reversed(self.activation_function), reversed(node_types)):
            layer_nodes = [Node(f"{node_type}_{depth}_{i}", node_type, af[i], depth, [], out_nodes) for i in range(node)]
            for out_node in out_nodes:
                out_node.in_nodes.extend(layer_nodes)
            out_nodes = layer_nodes
            depth -= 1

            if node_type == 'output':
                output_layer = layer_nodes
                print("Output layer created with nodes:", [node.name for node in output_layer])
                print("depth = ", depth)
            elif node_type == 'input':
                input_layer = layer_nodes
                print("Input layer created with nodes:", [node.name for node in input_layer])
                print("depth = ", depth)
            else:
                hidden_layer.insert(0, layer_nodes)
                print(f"Hidden layer {len(hidden_layer)} created with nodes:", [node.name for node in layer_nodes])
                print("depth = ", depth)
        
        self.output_layer = output_layer
        self.input_layer = input_layer
        self.hidden_layer = hidden_layer
        return 
            
    def print_network(self):
        """
        Print the structure of the network
        """
        print("Input Layer:")
        for node in self.input_layer:
            print(f"  Node {node.name}: activation function: {node.func.__name__}, weights: {node.weights}, bias: {node.bias}, in_nodes: {len(node.in_nodes)}, out_nodes: {len(node.out_nodes)}")
        
        for i, layer in enumerate(self.hidden_layer):
            print(f"Hidden Layer {i+1}:")
            for node in layer:
                print(f"  Node {node.name}: activation function: {node.func.__name__}, weights: {node.weights}, bias: {node.bias}, in_nodes: {len(node.in_nodes)}, out_nodes: {len(node.out_nodes)}")
        
        print("Output Layer:")
        for node in self.output_layer:
            print(f"  Node {node.name}: activation function: {node.func.__name__}, weights: {node.weights}, bias: {node.bias}, in_nodes: {len(node.in_nodes)}, out_nodes: {len(node.out_nodes)}")

    def forward_pass(self, x):
        """
        Perform a forward pass through the network
        """
        # Initialises the input layer
        for i, node in enumerate(self.input_layer):
            node.x = x[i]
            node.y = node.activate()
        
        # Forward pass through the hidden layer
        for layer in self.hidden_layer:
            for node in layer:
                node.x = sum(in_node.calculate_output() for in_node in node.in_nodes)
                node.y = node.activate()

        # Forward pas through the output layer
        for node in self.output_layer:
            node.x = sum(in_node.calculate_output() for in_node in node.in_nodes)
            node.y = node.activate()

        return [node.y for node in self.output_layer]
    
    def backward_pass(self, y_true):
        """
        Perform a backward pass through the netwrok and update the weights and biases
        """
        pass



nn = NeuralNetwork(num_of_nodes_per_layer=[3, 1,4, 2], activation_function=[lambda x: max(0, x), lambda x: max(0, x), lambda x: x, lambda x: x])
nn.create_network()
nn.print_network()
