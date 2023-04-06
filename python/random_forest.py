import numpy as np
from decision_tree import DecisionTree


# N v 1: Each specific tree is only responsible for judging whether the data belongs to a specific classification
class RandomForest:
    def __init__(
        self,
        max_depth = 20,
        min_pct = 0.05,
        threshold = 17,
        worst_gini = 0.95
    ):
        self.root_list = [
            DecisionTree(
                cls = i,
                max_depth = max_depth,
                min_pct = min_pct,
                threshold = threshold,
                worst_gini = worst_gini
            )       for i in range(10)
        ]


    def fit(self, data_train, label_train):
        for tree_index, decision_tree in enumerate(self.root_list):
            decision_tree.fit(data_train, label_train)


    def predict(self, data_test):
        predict = []
        for i in data_test:
            pro_list = []
            for dt in self.root_list:
                point = dt.root
                while 1:
                    if isinstance(point.value, float):
                        pro_list.append({'class':dt.cls, 'probability':point.value})
                        break
                    elif isinstance(point.value, int):
                        if i[point.value] < dt.threshold:
                            point = point.left_child
                        else:
                            point = point.right_child
                    else:
                        print('error:', str(type(point.value)))
            predict.append(max(pro_list,key=lambda x:x['probability'])['class'])
        return np.array(predict)