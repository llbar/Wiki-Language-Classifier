Language Classification
Lachlan Bartle
Feature Selection:
The features I chose to obtain for classification were obtained by looking at common
words in each language from the sample train.dat and the English and Dutch Wikipedia
provided in the project page. The selected features include:
Average Word Length (av-len):
Gets the average length of words in a line, categorized into three ranges - short (0-4),
medium (5-8), and long (8 or more).
Common Words: Indicates whether specific words ("het", "een", "en", "de", "the", "and",
"in", "of") are present in the line.
I chose these features because they help to determine the language the input is in
because Dutch words and English words have different average lengths, and most
Dutch/English statements contains certain distinct function words.
Decision Tree:
Decision Tree Node:
The Node class represents a single node in the decision tree. It contains information
about the node's value, whether it is a leaf node, and its children.
Entropy:
Entropy is computed to measure impurity in the dataset. The function calculates the
entropy of a set of examples using the formula: −∑i(pi*log2(pi))
Information Gain: Information gain is calculated for each feature, indicating the
reduction in entropy after the split. The feature with the maximum gain is chosen for
splitting.
Tree Construction:
The dec_tree function recursively constructs the decision tree using a top-down,
recursive approach. It uses information gain for feature selection and stops building the
tree when the specified maximum depth is reached.
Max Depth:
To avoid overfitting, a maximum depth parameter is used during tree construction. If the
depth is set to 1, the algorithm creates leaf nodes based on the plurality value of the
examples. I have the depth set to 20 in my code. If the depth was too low then it may not
catch underlying patterns in the data and underfit whereas if the tree was too deep then
it could overfit by including noise and outliers on the data memorization. (overfitting
would lead to issues on unseen data beyond the training data).
Testing Decision Tree:
The testing process involves classifying instances based on the trained decision tree,
and the results are printed. I tested depth lower and higher and had to find a nice spot in
the middle to get accuracy.
AdaBoost:
Weighted Instances:
The WeightedInstance class represents a weighted set of instances. The weights are
adjusted during the training process to focus on misclassified instances.
AdaBoost Training:
The AdaBoost algorithm is implemented in the AdaBoost class. Key features include:
Weighted Sample Handling:
The weights of instances are updated based on misclassifications, and the
sample is normalized to ensure proper weight distribution.
Ensemble Size:
The train method specifies the ensemble size. Generally, a higher ensemble size,
the better the performance of the model because the weak learner concentrates
on correcting the priors one’s mistakes, which can lead to more accuracy.
However, optimal ensemble size depends on the dataset’s characteristics
because there is a trade-off between model complexity and generalization.
Voting:
The vote method classifies instances by aggregating votes from the ensemble,
considering the weight of each decision tree.
Testing AdaBoost:
The testing process involves classifying instances based on the ensemble of decision
trees, and the results are printed. I found that changing the max depth in dec_tree for the
stumps in Adaboost helped it become more accurate. I decided to leave it at 3.
Files and Main:
Training and Prediction:
The train function handles training for both decision tree and AdaBoost models based on
the specified learning type. The predict function loads a trained model and performs
predictions on test instances.
Command Line:
The main function is the entry point and processes the command-line arguments to
execute either the training or prediction functionality.
