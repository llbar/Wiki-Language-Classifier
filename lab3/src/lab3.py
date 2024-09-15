# Lachlan Bartle Lab3 Language Classifier, Intro to AI
import pickle
import math
import sys

class Node:
    #representing a single node in a decision tree
    def __init__(self, value, is_leaf=False):
        #initializing the node
        #value: nodes value
        #is_leaf: boolean for knowing if it is a leaf node
        self.is_leaf, self.value, self.children = is_leaf, value, {}

    def add(self, label, d_node):
        #adds child to node
        #label: branch label
        #d_node: new node
        self.children[label] = d_node

    def classify(self, instance):
        #classifying instance
        #instance: instance being classified
        node = self
        while node:
            if node.is_leaf:
                return node.value
            branch = instance.features[node.value]
            node = node.children.get(branch, None)
        return None
    
def plurality_value(examples):
    #gets the classification from examples
    #examples: the list of examples
    count = {}
    for x in examples:
        count[x.goal] = count.get(x.goal, 0) + (x.weight if x.weight else 1)
    return max(count, key=count.get)

def entropy(examples):
    #entropy calc
    #examples: list of examples
    count = {}
    for x in examples:
        count[x.goal] = count.get(x.goal, 0) + (x.weight if x.weight else 1)
    total = -sum((p / len(examples)) * math.log2(p / len(examples)) for p in count.values())
    return total

def dec_tree(examples, features, parent_examples, depth=20):
    #building the dt, using information gain
    #examples: training examples
    #features: the features
    #parent_examples: parents example set
    #depth: max depth
    if not examples:
        return Node(plurality_value(parent_examples), is_leaf=True)
    if all(x.goal == examples[0].goal for x in examples):
        return Node(examples[0].goal, is_leaf=True)
    if not features:
        return Node(plurality_value(examples), is_leaf=True)
    feature, childdren = max_gain(examples, features)
    root = Node(feature)
    if depth < 1:
        depth = 1
    for value in childdren:
        x = childdren[value]
        subtree = Node(plurality_value(x), is_leaf=True) if depth == 1 else dec_tree(x, features - {feature}, examples, depth - 1)
        root.add(value, subtree)
    return root

class Instance:
    #single instance from data (one line)
    def __init__(self, line, preserve=False):
        #initializes the instance, gets the features for the instance
        #line: line from input
        #preserve: if line needs to be striped
        self.goal, self.value = (None, line) if preserve else (line[:2], line[2:])
        self.features, self.weight = get_features(line), None

class WeightedInstance:
    #weighted instance
    def __init__(self, instances):
        #initializes list of instances
        #instances: list of instances
        self.data = instances
        self.sum = 0
        for instance in self.data:
            instance.weight = 1
            self.sum += instance.weight
        self.dist_sum = self.sum

    def normalize(self):
        #weights change for distribution
        x = self.dist_sum/self.sum
        self.sum = 0
        for instance in self.data:
            instance.weight *= x
            self.sum += instance.weight

    def update_weight(self, i, new_weight):
        #updating weight of instance in data
        #i: index of instance
        #new_weight: updated weight
        self.sum -= self.data[i].weight
        self.data[i].weight = new_weight
        self.sum += new_weight

def max_gain(examples, features):
    #finds split in data to find most info gain
    #examples: list of examples
    #features: features
    ent = entropy(examples)
    max_val, max_feature, children = -1, None, None
    for feature in features:
        gains, childdren = gain(examples, feature, ent)
        if gains > max_val:
            max_val, max_feature, children = gains, feature, childdren
    return max_feature, children

def gain(examples, feature, ent):
    #calculates info gain after split
    #examples: list of examples
    #feature: split on feature
    #ent: entropy
    childdren = split(examples, feature)
    total = sum((len(x) / len(examples)) * entropy(x) for x in childdren.values())
    gains = ent - total
    return gains, childdren

def split(examples, feature):
    #splits list on feature
    #examples:list of examples
    #feature: feature split on
    result = {}
    for x in examples:
        value = x.features[feature]
        result.setdefault(value, []).append(x)
    return result

class DecisionTree:
    #class for decision tree model
    def __init__(self, train_file, test_file, hyp_file):
        files = (train_file, test_file)
        lines = parse(files)
        self.data = {"train": lines[0], "test": lines[1]}
        self.hyp_file = hyp_file
        self.tree = None

    def train(self):
        #training for dt, saves to file
        examples, features = self.data["train"], set(self.data["train"][0].features)
        self.tree = dec_tree(examples, features, [], 7)
        with open(self.hyp_file, "wb") as f:
            pickle.dump(self, f)

    def test(self, test_file=None):
        #tests model for dt
        #test_file: the file with tests
        if not self.tree:
            self.train()
        examples = parse([test_file])[0] if test_file else self.data["test"]
        result = [{"value": x.value, "result": self.tree.classify(x), "goal": x.goal} for x in examples]
        for res in result:
            print(res["result"])

class AdaBoost:
    #class for adaboost model
    def __init__(self, train_file, test_file, hyp_file):
        #initializing model
        #train_file: train data
        #test_file: test data
        #hyp_file: where the train stuff goes
        filenames = (train_file, test_file)
        lines = parse(filenames)
        self.data, self.hyp_file, self.ensemble, self.tree = {"train": lines[0], "test": lines[1]}, hyp_file, [], None

    def train(self, ensemble_size=2):
        #train fn for model, learns ensemble using adaboost then saves it to a file
        #ensemble_size: ensemble size
        examples, features = self.data["train"], set(self.data["train"][0].features.keys())
        sample, self.ensemble = WeightedInstance(examples), []
        for i in range(ensemble_size):
            stump = dec_tree(examples, features, [], 3)
            error = sum(x.weight if x.weight is not None else 1 for x in examples if stump.classify(x) != x.goal)
            for j, x in enumerate(examples):
                if stump.classify(x) == x.goal:
                    denom = sample.dist_sum - error
                    new_weight = (x.weight if x.weight is not None else 1) * error / denom if denom != 0 else 0
                    sample.update_weight(j, new_weight)
            sample.normalize()
            stump.weight = math.log(sample.dist_sum - error) / error 
            self.ensemble.append(stump)
        with open(self.hyp_file, "wb") as f:
            pickle.dump(self, f)

    def test(self, test_file=None):
        #test fn for model
        #test_file: test data
        if not self.ensemble:
            self.train()
        examples = parse([test_file])[0] if test_file else self.data["test"]
        result = [{"value": x.value, "result": self.vote(x), "goal": x.goal} for x in examples]
        for res in result:
            print(res["result"])

    def vote(self, instance):
        #classifies an instance from votes from ensemble
        #instance: instance to classify
        count = {}
        for stump in self.ensemble:
            decision = stump.classify(instance)
            count[decision] = count.get(decision, 0) + stump.weight
        return max(count, key=count.get)

def parse(files):
    #parses the lines in the file
    #files: files
    lines = [[] for _ in files]
    for i, filename in enumerate(files):
        if filename:
            with open(filename, encoding='utf-8') as file:
                lines[i].extend(Instance(line) for line in file if len(line.strip()) > 3)
    return lines

def get_features(line):
    #getting the features of the line
    #line: the line
    words = set(line.split())
    return {
        "av-len": average_word_len(line),
        "contains-het": "het" in words,
        "contains-een": "een" in words,
        "contains-en": "en" in words,
        "contains-de": "de" in words,
        "contains-the": "the" in words,
        "contains-and": "and" in words,
        "contains-in": "in" in words,
        "contains-of": "of" in words,
    }

def average_word_len(line):
    #finds range of average length of words
    #line: line its called on
    total = 0
    x = 0, 4
    y = 5, 8
    z = 8, None
    for _ in line:
        total += 1
    average = total//len(line.split())
    if average <= 4:
        return x
    if 4 < average <= 8:
        return y
    return z

def train(train_file, hyp_file, dtorada):
    #handles train; determines if it is dt or ada and creates corresponding model
    #train_file: file to use to train
    #hyp_file: file that will be used to test
    #dtorada: dt or ada
    if dtorada == "dt":
        model = DecisionTree(train_file=train_file, test_file=None, hyp_file=hyp_file)
    else:
        model = AdaBoost(train_file=train_file, test_file=None, hyp_file=hyp_file)
    model.train()


def predict(hypothesis_file, test_file):
    #handles predict; opens hypothesiss file and loads it into model test
    #hypothesis_file: the model that was trained
    #test_file: the file with the test instances
    hypothesis_file = open(hypothesis_file, "rb")
    model = pickle.load(hypothesis_file)
    hypothesis_file.close()
    model.test(test_file)

def main():
    #Main function handles train and predict command from command line args
    if len(sys.argv) < 2:
        exit(1)
    command = sys.argv[1]
    if command == "train": #what to do for train command
        if len(sys.argv) < 5:
            print("Usage: python3 lab3.py train <examples> <hypothesisOut> <learning-type>")
            exit(1)
        examples = sys.argv[2]
        hyp_file = sys.argv[3]
        dtorada = sys.argv[4]
        train(examples, hyp_file, dtorada)
    elif command == "predict": #what to do for predict command
        if len(sys.argv) < 4:
            print("Usage: python3 lab3.py predict <hypothesis> <file>")
            exit(1)
        hypothesis_file = sys.argv[2]
        test_file = sys.argv[3]
        predict(hypothesis_file, test_file)

if __name__ == "__main__":
    main()