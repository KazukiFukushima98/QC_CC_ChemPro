import gurobipy as gp
from gurobipy import *
import time

model = Model(name="Sample1")

xlb = 0
xub = 5
x = model.addVars(2, vtype=GRB.CONTINUOUS, lb=xlb, ub=xub, name="x")
y = model.addVars(2, vtype=GRB.BINARY, name="y")

c = [1.5, 1, 6.1, 5.9]
alpha = [0.75, 0.8]
d = 3

model.setObjective(c[0] * x[0]**2 + c[1] * x[1]**2 + c[2] * y[0] + c[3] * y[1], sense=gp.GRB.MINIMIZE)

c_1 = model.addConstr(xlb <= x[0], name="const_1")
c_2 = model.addConstr(xub >= x[0], name="const_2")
c_3 = model.addConstr(xlb <= x[1], name="const_3")
c_4 = model.addConstr(xub * y[1] >= x[1], name="const_4")
c_5 = model.addConstr(y[0] * alpha[0] * x[0] + y[1] * alpha[1] * x[1] >= d, name="const_5")

st = time.time()
model.optimize()
ed = time.time()

if model.Status == gp.GRB.OPTIMAL:
    x_opt = [x[0].X, x[1].X]
    y_opt = [y[0].X, y[1].X]
    val_opt = model.ObjVal

