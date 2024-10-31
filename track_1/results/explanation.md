# Explanations


## SecSVM
When you use CClassifierSVM.predict(self, X, return_decision_function=True) in the SecML library, the method returns two outputs:

labels: The predicted class labels for each sample in X.
scores: The decision function scores, which represent the confidence level of the classifier in its predictions.
Explanation of the Scores:
The scores returned by setting return_decision_function=True are the decision function values (or distance from the hyperplane) for each sample in X. These scores indicate how confidently the classifier assigns a sample to each class.

In the case of binary classification:

A positive score indicates that the sample is predicted as belonging to the positive class.
A negative score indicates that the sample is predicted as belonging to the negative class.
The magnitude of the score represents the confidence in the classification: the farther away from zero, the more confident the classifier is.
For multi-class classification, the scores will contain decision function values for each class, often computed as the distance to the separating hyperplanes between each pair of classes (in a one-vs-one or one-vs-all scheme, depending on the classifier setup).