from simpleautodiff import *

Node.verbose = True

# create root nodes
print("============================Forward Primal Trace========================")
x1 = Node(2)
x2 = Node(5)

# create computational graph and evaluate function value
y = sub(add(log(x1), mul(x1, x2)), sin(x2))
# create visualization of computational graph via Graphviz (Optional, requires Graphviz)
debug_computational_graph(y, "vis.png")

print("===============================Forward mode=============================")
# perform forward-mode autodiff
forward(x1)

# resets the intermediate partial derivatives stored. (Can be removed in this case)
reset(rootNode=y)

print("===============================Reverse mode=============================")
# perform reverse-mode autodiff
reverse(y)
