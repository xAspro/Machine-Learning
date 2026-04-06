import numpy as np


class Node:
    def __init__(self, name, type_of_node, func, layer, in_nodes, out_nodes, x=None, y=None, weights=None, bias=None, derivative_func=None):
        """
        Node constructor.

        Parameters
        ----------
        name : str
            Name of the node for debugging.
        type_of_node : str
            'input', 'hidden', or 'output'.
        func : callable
            Activation function.
        layer : int
            Layer number this node belongs to.
        in_nodes : list[Node]
            Incoming nodes to this node.
        out_nodes : list[Node]
            Outgoing nodes from this node.
        x : float, optional
            Weighted input (computed during forward pass).
        y : float, optional
            Node output (computed during forward pass).
        weights : np.ndarray, optional
            Weights for incoming connections.
        bias : float, optional
            Bias term for the node.
        """
        if type_of_node not in ['input', 'hidden', 'output']:
            raise ValueError("type_of_node must be 'input', 'hidden', or 'output'")
        if type_of_node == 'input':
            if func not in [None, 'linear']:
                print(f"Warning: Input layer activation ignored for {name}. Using identity.")
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
        self.name = name
        self.derivative_func = derivative_func

        if type_of_node != 'input':
            self.weights = weights if weights is not None else np.random.rand(len(in_nodes))
            self.bias = bias if bias is not None else np.random.rand()
        else:
            self.weights = None
            self.bias = None

        if isinstance(func, str):  # If it's a name, fetch the corresponding function and its derivative
            self.func, self.derivative_func = self._get_activation_function(func)
        elif not callable(func) or not callable(self.derivative_func):
            raise ValueError("If not a name, the activation function and its derivative must be callable.")
        

    def _get_activation_function(self, name):
        if name == 'relu':
            return lambda x: max(0, x), lambda x: 1.0 if x > 0 else 0.0
        elif name == 'sigmoid':
            def sigmoid(x):
                return 1 / (1 + np.exp(-x))
            def sigmoid_derivative(x):
                s = sigmoid(x)
                return s * (1 - s)
            return sigmoid, sigmoid_derivative
        elif name == 'tanh':
            return lambda x: np.tanh(x), lambda x: 1 - np.tanh(x) ** 2
        elif name == 'linear':
            return lambda x: x, lambda x: 1.0
        else:
            raise ValueError(f"Unsupported activation function name: {name}")
        
    

class NeuralNetwork:
    def __init__(self, num_of_nodes_per_layer, activation_function, task, learning_rate=0.01, derivative_func=None):
        """
        Constructor
        -----------
            - input_layer: list of nodes in the input layer
            - hidden_layer: list of nodes in the hidden layer
            - output_layer: list of nodes in the output layer
            - learning_rate: the learning rate for training the network
            - num_of_nodes_per_layer: a list containing the number of nodes in each layer
            - activation_function: the activation function to be used in the network
            - loss_function: the loss function to be used for training the network (determined by the task)
        """
        self.input_layer = None
        self.hidden_layer = None
        self.output_layer = None
        self.num_of_nodes_per_layer = num_of_nodes_per_layer
        self.activation_function = activation_function
        self.learning_rate = learning_rate
        self.derivative_func = derivative_func

        self.normalise_functions()
        self.loss_function = self._get_loss_function(task)

    def normalise_functions(self):
        """
        Ensure that the activation function and derivative of the function is in 
        the correct format (list of functions for each layer)
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
        df = self.derivative_func

        if not isinstance(af, list):
            af = [af] * len(layers)
        elif len(af) != len(layers):
            raise ValueError(
                f"Length of activation function list must match the number of "
                f"layers. Expected {len(layers)}, got {len(af)}"
            )

        if df is None:
            df = [None] * len(layers)
        elif not isinstance(df, list):
            df = [df] * len(layers)
        elif len(df) != len(layers):
            raise ValueError(
                f"Length of derivative function list must match the number of "
                f"layers. Expected {len(layers)}, got {len(df)}"
            )

        self.activation_function = [normalise_layer(val, size) for val, size in zip(af, layers)]
        self.derivative_func = [normalise_layer(val, size) for val, size in zip(df, layers)]

    def create_network(self):
        """
        Create the connections between the layers of the network
        """
        output_layer = []
        input_layer = []
        hidden_layer = []
        in_nodes = []
        depth = 1

        n_input = self.num_of_nodes_per_layer[0]
        n_output = self.num_of_nodes_per_layer[-1]

        node_types = ['input'] + ['hidden'] * (len(self.num_of_nodes_per_layer) - 2) + ['output']

        for node, af, df, node_type in zip(self.num_of_nodes_per_layer, self.activation_function, self.derivative_func, node_types):
            layer_nodes = []
            for i in range(node):
                new_node = Node(
                    name=f"{node_type}_{depth}_{i}",
                    type_of_node=node_type,
                    func=af[i],
                    layer=depth,
                    in_nodes=in_nodes,
                    out_nodes=[],
                    derivative_func=df[i]
                )
                layer_nodes.append(new_node)

            for in_node in in_nodes:
                in_node.out_nodes.extend(layer_nodes)
            
            in_nodes = layer_nodes
            depth += 1


            if node_type == 'input':
                input_layer = layer_nodes
            elif node_type == 'output':
                output_layer = layer_nodes
            else:
                hidden_layer.append(layer_nodes)

        self.output_layer = output_layer
        self.input_layer = input_layer
        self.hidden_layer = hidden_layer
        return 
            
    def print_network(self, with_outputs=False):
        """
        Print the structure of the network
        """
        print("\nInput Layer:")
        for node in self.input_layer:
            print(f"  Node {node.name}: weights: {node.weights}, bias: {node.bias}, in_nodes: {len(node.in_nodes)}, out_nodes: {len(node.out_nodes)}", end="")
            if with_outputs:
                print(f", output: {node.y}")
            else:
                print()

        for i, layer in enumerate(self.hidden_layer):
            print(f"\nHidden Layer {i+1}:")
            for node in layer:
                print(f"  Node {node.name}: weights: {node.weights}, bias: {node.bias}, in_nodes: {len(node.in_nodes)}, out_nodes: {len(node.out_nodes)}", end="")
                if with_outputs:
                    print(f", output: {node.y}")
                else:
                    print()
        
        print("\nOutput Layer:")
        for node in self.output_layer:
            print(f"  Node {node.name}: weights: {node.weights}, bias: {node.bias}, in_nodes: {len(node.in_nodes)}, out_nodes: {len(node.out_nodes)}", end="")
            if with_outputs:
                print(f", output: {node.y}")
            else:
                print()

    def _forward_input_layer(self, x):
        """
        Forward pass through the input layer (simply sets the output of the input nodes to the input values)
        """
        for i, node in enumerate(self.input_layer):
            node.x = x[i]
            node.y = node.func(node.x)
    
    def _forward_layer(self, layer):
        """
        Forward pass through a hidden or output layer
        """
        for node in layer:
            node.x = sum(in_node.y * node.weights[i] for i, in_node in enumerate(node.in_nodes)) + node.bias
            node.y = node.func(node.x)

    def forward_pass(self, x):
        """
        Perform a forward pass through the network
        """
        # Initialises the input layer
        self._forward_input_layer(x)

        # Forward pass through hidden layers
        for layer in self.hidden_layer:
            self._forward_layer(layer)

        # Forward pass through output layer
        self._forward_layer(self.output_layer)

        return [node.y for node in self.output_layer]
    
    def backward_pass(self, y_true):
        """
        Perform a backward pass through the netwrok and update the weights and biases
        """
        pass

    def _get_loss_function(self, task):
        if task == 'classification':
            def loss(y_true, y_pred, eps=1e-12):
                p = np.clip(y_pred, eps, 1 - eps)
                return -np.mean((y_true * np.log(p)) + ((1 - y_true) * np.log(1 - p)))
        if task == 'regression':
            def loss(y_true, y_pred):
                return np.mean((y_true - y_pred) ** 2)
        return loss



nn = NeuralNetwork(num_of_nodes_per_layer=[3, 1,4, 2], activation_function=[lambda x: max(0, x), lambda x: max(0, x), lambda x: x, lambda x: x], task='regression', derivative_func=[lambda y: y * (1 - y), lambda y: y * (1 - y), lambda y: 1, lambda y: 1])
nn.create_network()
nn.print_network()
nn.forward_pass([1, 2, 3])
nn.print_network(with_outputs=True)
