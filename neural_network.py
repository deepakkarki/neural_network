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
		self.input  = [Node() for i in range(0, in_num)]
		self.hidden = [Node() for i in range(0, hid_num)]
		self.output = [Node() for i in range(0, out_num)]

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

	def build(self, data_set):
		'''
		Updates the neural network from the given list of data
		data_set (list of tuples) : contains data for learning
		log (Boolean) : if True, logs the changes in weights and bias values for each node for each input tuple
		'''
		delta_epoch = 0
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

			l = 0.1 #gradient - not too sure how to use this!

			count = 0
			# change weights and bias
			for node in self.output:
				err = node.error
				for parent in node.parents:
					dw = l * err * parent[0].out 
					#dw : delta wt - change to be applied
					parent[1] += dw
					delta_epoch += math.fabs(dw)
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
					delta_epoch += math.fabs(dw) #update overall change for dataset
					#update children list in the parent with the new weight
					parent[0].children[count][1] = parent[1] 
				db =  l * err
				node.bias += db
				count += 1

		return delta_epoch



	def train(self, data_set, it=10, treshold=0.025):
		'''
		Trains the neural network iteratively
		default iterations : it = 10
		default treshold : treshold = 0.025 
		'''
		for i in range(it):
			val = self.build(data_set)
			if math.fabs(val) < treshold:
				return "change in wt's too small", i

		return "Number of iterations run out", it



def sigmoid(Ij):
	var = math.pow(math.e, -Ij)
	return 1/(1+var)
		
	
class Node:
	id = 1

	def __init__(self):
		self.id = Node.id
		self.parents = []
		#list of link and corresponding weight
		self.children = []
		self.bias = random.random()
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
	res, val = n.train([(1,0,1,1,0)])
	print n
	print res
	print "Number of iterations : ", val