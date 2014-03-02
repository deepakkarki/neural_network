import random
import math 

class neural_net(object):
	'''
	Neural network which has following fixed topology:
		- one input layer
		- one hidden layer
		- one output layer
	'''
	def __init__(self, in_num=1, hid_num=1, out_num=1):
		'''
		creates the network topology, initializes weights, bias of all nodes.
		*_num : num of nodes for respective layer
		'''
		self.input  = [new Node() for i in range(0, in_num)]
		self.hidden = [new Node() for i in range(0, hid_num)]
		self.output = [new Node() for i in range(0, out_num)]

		#sets up links b/w input layer and hidden layer
		for child in self.hidden:
			for parent in self.input:
				wt= random.uniform(-1,1)
				child.add_parent(parent, wt)
				parent.add_child(child, wt)

		#sets up links b/w hidden layer and input layer
		for child in self.output:
			for parent in self.hidden:
				wt= random.uniform(-1,1)
				child.add_parent(parent, wt)
				parent.add_child(child, wt)

		
	def build(self, data_set, log=True):
		'''
		Updates the neural network from the given list of data
		data_set (list of tuples) : contains data for learning
		log (Boolean) : if True, logs the changes in weights and bias values for each node for each input tuple
		'''

		for data in data_set:
			
			assert len(self.input) + len(self.output) == len(data)
			attributes = data[0:len(self.input)]
			output_real = data[len(self.input):]
			for i in attributes
				self.input[i] = i

			for node in self.hidden:
				Ij = map(lambda x : x[0].out * x[1], node.parents) + node.bias
				node.out = sigmoid(Ij)

			for node in self.out:
				Ij = map(lambda x : x[0].out * x[1], node.parents) + node.bias
				node.out = sigmoid(Ij)

			# backpropogate the errors now
			j = 0
			for node in self.out:
				node.error = node.out * (1-node.out) * (output_real[j] - node.out)




		
	
class Node:
	Node.id = 0

	def __init__(self):
		self.id = node.id
		self.parents = []
		#tuple of link and corresponding weight
		self.children = []
		self.bias = random.random()
		self.out = None
		self.error = None
		Node.id += 1

	def add_parent(self, link, wt):
		self.parents.append((link,wt))

	def add_child(self, link, wt):
		self.children.append((link,wt))
