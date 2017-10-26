import graphviz
import pydotplus
from sklearn.tree import export_graphviz

def plot_decision_tree(dtc, feature_names, class_names):
    
    export_graphviz(dtc, out_file="tmp.dot", 
                    feature_names=feature_names, class_names=class_names, 
                    filled = True, impurity = False)
    
    with open("tmp.dot") as f:
        dot_graph = f.read()
        
    return graphviz.Source(dot_graph)