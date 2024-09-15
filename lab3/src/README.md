### Usage documentation:

2 entry points cmd: (everything is in same file so pretty basic):

train <examples> <hypothesisOut> <learning-type> should read in labeled examples and perform some sort of training.
-examples is a file containing labeled examples.
-hypothesisOut specifies the file name to write model to.
-learning-type specifies the type of learning algorithm to run, it is either "dt" or "ada".

predict <hypothesis> <file> Classify each line as either English or Dutch using the specified model. Note that this must not do any training, but should take a model and make a prediction about the input. For each input example, it should simply print its predicted label on a newline. It should not print anything else.
-hypothesis is a trained decision tree or ensemble created by train program
-file is a file containing lines of 15 word sentence fragments in either English or Dutch.

### Train/Test Data format:

Train:
<label>|<data>
example:
en|description languages. In 2005, the 1985 paper in which the Yale shooting scenario was first 
nl|zijn de loketten van dit station gesloten en is het een stopplaats geworden. Voor de 
en|tuples. A decision tree is a flow-chart-like structure, where each internal (non-leaf) node denotes a 
en|effect is that alive changes value (since alive was true before, this corresponds to alive 
en|decision trees due to their added sparsity, permit non-greedy learning methods and monotonic constraints to 
en|of occlusion, which formalizes the â€œpermission to changeâ€ for a fluent. The effect of an 
en|one implication involved.) A solution proposed by Erik Sandewall was to include a new condition 

Test:
<data>
example:
be imposed. Decision tree learning is the construction of a decision tree from class-labeled training 
decision tree, so that every internal node has exactly 1 leaf node and exactly 1 
werd het dienstgebouw opgetrokken, dat zich eveneens onder een schilddak bevindt, langs de straatzijde verspringend 
internal node as a child (except for the bottommost node, whose only child is a 
of shooting are correctly formalized. (Predicate completion is more complicated when there is more than 
beperken, zorgde de NMBS in 2004 voor 60 extra parkeerplaatsen aan het station van Duffel. 
root node. There are many specific decision-tree algorithms. Notable ones include: While the Yale shooting 
