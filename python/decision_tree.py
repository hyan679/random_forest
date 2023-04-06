import numpy as np


class node:
    def __init__(self, depth=0, value=None):
        self.depth = depth  # not necessary but can simplify bulid code
        self.left_child = None
        self.right_child = None
        self.value = value
        # If the data type of value is float -> it is a leaf node, the value is the probability
        # If the data type of value is int  -> it is a branch node, the value is the column name


# Binary tree
class DecisionTree:
    def __init__(
        self,
        cls = 0,
        max_depth = 12,
        min_pct = 0.05,
        threshold = 0.32,
        worst_gini = 0.95
    ):
        self.root = node(depth=0)
        self.cls = cls
        self.max_depth = max_depth
        self.min_pct = min_pct
        self.threshold = threshold
        self.worst_gini = worst_gini
        self.data = None
        self.label = None


    def build(self, point, column, row):
        gini_list = [{'gini':1},]
        row_len = len(row)
        min_set_num = round(self.min_pct * row_len) + 1 # Ensure > 0
        # Scan each pixel in order to find out whether the pixel can effectively divide the data
        for i in column:
            condition = self.data[row,i] < self.threshold
            condition_true = sum(condition)
            condition_false = row_len - condition_true
            # If the data set cannot be divided into two parts,
            # The condition is useless
            if condition_true > min_set_num and condition_false > min_set_num:
                t_t = sum(condition & (self.label[row]==self.cls))
                f_t = sum(~condition & (self.label[row]==self.cls))
                p_tt = t_t / condition_true
                p_ft = f_t / condition_false
                gini = (1 - p_tt**2 - ((condition_true - t_t) / condition_true)**2)*(condition_true / row_len)\
                    + (1 - p_ft**2 - ((condition_false - f_t) / condition_false)**2)*(condition_false / row_len)
                gini_list.append({'gini':gini, 'column':i, 'condition':condition})
        best_res = min(gini_list, key=lambda x:x.get('gini'))
        if best_res.get('gini') < self.worst_gini and point.depth < self.max_depth:
            best_column = int(best_res.get('column'))
            point.value = best_column
            point.left_child = node(depth=point.depth + 1)
            point.right_child = node(depth=point.depth + 1)
            
            condition = best_res.get('condition')
            new_col = list(column)
            new_col.remove(best_column)
            self.build(point.left_child, tuple(new_col), row[condition])
            self.build(point.right_child, tuple(new_col), row[~condition])
        else:
            point.value = float(sum(self.label[row]==self.cls)/row_len)
            
            
    def fit(self, data, label):
        self.data = data
        self.label = label
        self.build(self.root, tuple(range(data.shape[1])), np.arange(data.shape[0]))