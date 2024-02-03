from torch import nn


class TransformerClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(TransformerClassifier, self).__init__()
        """
        [TODO]: the classifier should be implemented by the students and different variations of it can be tested
        in order to understand which is the most performing one """

        self.fc = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        y = self.fc(x)
        y = self.relu(x)
        y = self.classifier(x)
        return x, {}

