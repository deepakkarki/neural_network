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
		self.input  = [Node(), Node(), Node()]
		self.hidden = [Node(-0.4), Node(0.2)]
		self.output = [Node(0.1)]

		#set up the connection
		self.input[0].add_child(self.hidden[0], 0.2)
		self.input[0].add_child(self.hidden[1], -0.3)
		self.input[1].add_child(self.hidden[0], 0.4)
		self.input[1].add_child(self.hidden[1], 0.1)
		self.input[2].add_child(self.hidden[0], -0.5)
		self.input[2].add_child(self.hidden[1], 0.2)

		self.hidden[0].add_child(self.output[0], -0.3)
		self.hidden[1].add_child(self.output[0], -0.2)

		self.output[0].add_parent(self.hidden[0], -0.3)
		self.output[0].add_parent(self.hidden[1], -0.2)

		self.hidden[0].add_parent(self.input[0], 0.2)
		self.hidden[0].add_parent(self.input[1], 0.4)
		self.hidden[0].add_parent(self.input[2], -0.5)
		self.hidden[1].add_parent(self.input[0], -0.3)
		self.hidden[1].add_parent(self.input[1], 0.1)
		self.hidden[1].add_parent(self.input[2], 0.2)

	def __str__(self):
		nl = '\n'
		tb = '\t'
		sp = ' '
		ind = nl+tb
		s = 'Neural network structure' + ind
		
		s += 'Input layer :' +ind
		for node in self.input:
			s += str(node.id) + ' , '

		s += ind + 'Hidden layer :' +ind
		for node in self.hidden:
			s += str(node.id) + ' , '

		s += ind + 'Output layer :' +ind
		for node in self.output:
			s += str(node.id) + ' , '

		s += nl

		s += ind + 'Weights :' + ind

		for child in self.hidden:
			for parent in child.parents:
				s += tb + 'W('+ str(child.id) + '-' + str(parent[0].id) + ') : ' + str(parent[1]) + ind

		s += ind

		for child in self.output:
			for parent in child.parents:
				s += tb + 'W('+ str(child.id) + '-' + str(parent[0].id) + ') : ' + str(parent[1]) + ind

		s += ind + 'Bias :' + ind

		for node in self.hidden:
			s += tb + "Node # " + str(node.id) + ' : ' + str(node.bias) + ind

		for node in self.output:
			s += tb + "Node # " + str(node.id) + ' : ' + str(node.bias) + ind


		return s

	def build(self, data_set, log=True):
		'''
		Updates the neural network from the given list of data
		data_set (list of tuples) : contains data for learning
		log (Boolean) : if True, logs the changes in weights and bias values for each node for each input tuple
		'''

		for data in data_set:
			
			assert len(self.input) + len(self.output) == len(data), "input tuple has bad format"
			attributes = data[0:len(self.input)]
			output_real = data[len(self.input):]

			for i in range(0, len(attributes)):
				self.input[i].out = attributes[i]

			for node in self.hidden:
				Ij = sum(map(lambda x : x[0].out * x[1], node.parents)) + node.bias
				node.out = sigmoid(Ij)

			for node in self.output:
				Ij = sum(map(lambda x : x[0].out * x[1], node.parents)) + node.bias
				node.out = sigmoid(Ij)

			# backpropogate the errors now
			j = 0
			for node in self.output:
				node.error = node.out * (1-node.out) * (output_real[j] - node.out)
				j = j + 1

			for node in self.hidden:
				node.error = node.out * (1-node.out) * sum( map(lambda x : (x[0].error * x[1] ) , node.children) )

			l = 0.9 #gradient something - I didn't really understand

			count = 0
			# change weights and bias
			for node in self.output:
				err = node.error
				for parent in node.parents:
					dw = l * err * parent[0].out 
					#dw : delta wt - change to be applied
					parent[1] += dw
					#update children list in the parent with the new weight
					parent[0].children[count][1] = parent[1] 
				db =  l * err
				node.bias += db
				count += 1

			count = 0
			for node in self.hidden:
				err = node.error
				for parent in node.parents:
					dw = l * err * parent[0].out 
					#dw : delta wt - change to be applied
					parent[1] += dw
					#update children list in the parent with the new weight
					parent[0].children[count][1] = parent[1] 
				db =  l * err
				node.bias += db
				count += 1

			#simple - two more iterations : 1st loop update hidden; second update outer layer.


def sigmoid(Ij):
	var = math.pow(math.e, -Ij)
	return 1/(1+var)
		
	
class Node:
	id = 1

	def __init__(self, bias=None):
		self.id = Node.id
		self.parents = []
		#tuple of link and corresponding weight
		self.children = []
		self.bias = bias or random.random()
		self.out = None
		self.error = None
		Node.id += 1

	def add_parent(self, link, wt):
		self.parents.append([link,wt])

	def add_child(self, link, wt):
		self.children.append([link,wt])


if __name__ == '__main__':
	n = neural_net(3, 2, 2)
	print n
	n.build([(1,0,1,1)])
	print n