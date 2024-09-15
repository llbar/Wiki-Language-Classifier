import sys
import math

def read_data(file_path):
    with open(file_path) as file:
        data = [line.strip().split() for line in file]

    # Convert boolean attributes to boolean values
    data = [[True if attr.lower() == 'true' else False for attr in entry[:-1]] + [entry[-1]] for entry in data]

    return data

def calculate_entropy(data):
    class_counts = {}
    total_samples = len(data)

    for entry in data:
        label = entry[-1]
        class_counts[label] = class_counts.get(label, 0) + 1

    entropy = 0.0
    for count in class_counts.values():
        probability = count / total_samples
        entropy -= probability * math.log2(probability)

    return entropy

def calculate_information_gain(data, attribute_index, entropy_before_split):
    attribute_values = set(entry[attribute_index] for entry in data)
    entropy_after_split = 0.0

    for value in attribute_values:
        subset = [entry for entry in data if entry[attribute_index] == value]
        probability = len(subset) / len(data)
        entropy_after_split += probability * calculate_entropy(subset)

    information_gain = entropy_before_split - entropy_after_split
    return information_gain

def find_best_attribute(data):
    num_attributes = len(data[0]) - 1  # Assuming the last column is the class label
    entropy_before_split = calculate_entropy(data)
    
    best_attribute_index = None
    max_information_gain = 0.0

    for i in range(num_attributes):
        information_gain = calculate_information_gain(data, i, entropy_before_split)
        if information_gain > max_information_gain:
            max_information_gain = information_gain
            best_attribute_index = i

    return best_attribute_index

def get_class_distribution(data):
    class_counts = {}
    for entry in data:
        label = entry[-1]
        class_counts[label] = class_counts.get(label, 0) + 1
    return class_counts

def print_decision_tree(data, attribute_index):
    true_branch = [entry for entry in data if entry[attribute_index]]
    false_branch = [entry for entry in data if not entry[attribute_index]]

    true_class_distribution = get_class_distribution(true_branch)
    false_class_distribution = get_class_distribution(false_branch)

    root_attribute = f"A{attribute_index + 1}"
    print(f"Root Node: {root_attribute}")

    print(f"If {root_attribute} = True:")
    print(f"  Class Distribution: {true_class_distribution}")

    # Print the next level for when A4 is true
    next_attribute_true = find_best_attribute(true_branch)
    if next_attribute_true is not None:
        print(f"  Next Level True: A{next_attribute_true + 1}")
        print_decision_tree_level(true_branch, next_attribute_true)

    print(f"If {root_attribute} = False:")
    print(f"  Class Distribution: {false_class_distribution}")

    # Print the next level for when A4 is false
    next_attribute_false = find_best_attribute(false_branch)
    if next_attribute_false is not None:
        print(f"  Next Level False: A{next_attribute_false + 1}")
        print_decision_tree_level(false_branch, next_attribute_false)

def print_decision_tree_level(data, attribute_index):
    true_branch = [entry for entry in data if entry[attribute_index]]
    false_branch = [entry for entry in data if not entry[attribute_index]]

    true_class_distribution = get_class_distribution(true_branch)
    false_class_distribution = get_class_distribution(false_branch)

    attribute = f"A{attribute_index + 1}"
    print(f"  If {attribute} = True:")
    print(f"    Class Distribution: {true_class_distribution}")

    # Print the next level for when A6 (or any other attribute) is true
    next_attribute_true = find_best_attribute(true_branch)
    if next_attribute_true is not None:
        print(f"    Next Level True: A{next_attribute_true + 1}")
        print_decision_tree_level(true_branch, next_attribute_true)

    print(f"  If {attribute} = False:")
    print(f"    Class Distribution: {false_class_distribution}")

    # Print the next level for when A6 (or any other attribute) is false
    next_attribute_false = find_best_attribute(false_branch)
    if next_attribute_false is not None:
        print(f"    Next Level False: A{next_attribute_false + 1}")
        print_decision_tree_level(false_branch, next_attribute_false)

def main():
    # Check if command-line arguments are provided
    if len(sys.argv) != 2:
        print("Usage: python hw3.py <input_file_path>")
        sys.exit(1)

    # Extract the input file path from the command-line arguments
    input_file_path = sys.argv[1]

    # Read data from input file
    data = read_data(input_file_path)

    # Find the best attribute for the root node
    root_attribute_index = find_best_attribute(data)
    
    # Print the decision tree for the first two levels
    print_decision_tree(data, root_attribute_index)

if __name__ == "__main__":
    main()
