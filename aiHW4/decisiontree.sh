#!/bin/bash

echo "----------------------------------###########----"
python classify.py --data data/house-votes-84.train --mode train --model-file decision_tree.model --algorithm decision_tree

python classify.py --data data/house-votes-84.test --mode test --model-file decision_tree.model --algorithm decision_tree
echo "----------------------------------###########----"
python classify.py --data data/iris.train --mode train --model-file decision_tree.model --algorithm decision_tree

python classify.py --data data/iris.test --mode test --model-file decision_tree.model --algorithm decision_tree
echo "----------------------------------###########----"
python classify.py --data data/monks-1.train --mode train --model-file decision_tree.model --algorithm decision_tree

python classify.py --data data/monks-1.test --mode test --model-file decision_tree.model --algorithm decision_tree
echo "----------------------------------###########----"
python classify.py --data data/monks-2.train --mode train --model-file decision_tree.model --algorithm decision_tree

python classify.py --data data/monks-2.test --mode test --model-file decision_tree.model --algorithm decision_tree
echo "----------------------------------###########----"
python classify.py --data data/monks-3.train --mode train --model-file decision_tree.model --algorithm decision_tree

python classify.py --data data/monks-3.test --mode test --model-file decision_tree.model --algorithm decision_tree




