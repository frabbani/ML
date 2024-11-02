I recently explored decision trees and found the experience to be very informative.

Decision trees are a tree-based data structure designed to model decision-making processes. They represent hierarchical relationships among attributes (features) in a dataset, capturing the underlying patterns and relationships. By recursively splitting the data based on the values of given features, decision trees create a visual representation that can be used for classification or regression tasks.
For my C++ implementation, I chose to create a decision tree without prior knowledge of the underlying patterns in the training dataset. I built the tree based on the "purity" of the features and their values to evaluate its performance in classifying test data. Purity measures a feature’s ability to correctly classify data, with the Gini impurity method often utilized for this assessment. Additionally, I used K-means clustering to bin large quantities of data and applied the Gini method to select pure feature values, resulting in a pruned tree that yields optimal results.
More on Gini Impurity

If we assume the following statements: “AI engineers are smart” and “Some men are smart and some aren’t,” the best question to ask to assess if a person of interest is smart would be, “Are you an AI engineer?” rather than, “Are you a man?”
By inspecting and scoring our test data—specifically a collection of men and women, including AI engineers and non-engineers—using the Gini impurity method helps us ask the right questions.

![gini](https://github.com/user-attachments/assets/34c75cca-232c-42f0-bded-ad2e3f5574e8)


#DecisionTrees
#MachineLearning
#DataScience
#ArtificialIntelligence
#DataAnalysis
#GiniImpurity
#C++
#PredictiveModeling 
