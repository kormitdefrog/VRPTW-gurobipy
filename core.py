import numpy as np
from gurobipy import GRB, Model, quicksum


def solve_VRPTW(
    coordinate: np.ndarray,
    time_window: np.ndarray,
    demand: np.ndarray,
    service_duration: np.ndarray,
    vehicle_quantity: int,
    vehicle_capacity: float,
    cost_per_distance: float,
    time_per_distance: float,
    early_penalty_weight: np.ndarray,
    late_penalty_weight: np.ndarray,
    big_m: float,
    timelimit: float
):
    """
    node quantity = customer quantity + 2 = n + 2

    the starting depot is node 0 and the ending depot is node n + 1

    time window for node 0 should be [0, 0] and for node n + 1 should be [0, max operating time]

    return: is_feasible, objective value, arc matrix, arrival time matrix

    """
    
    # define sets
    node_quantity = coordinate.shape[0]
    customer_quantity = node_quantity - 2

    N = range(node_quantity)
    C = range(1, customer_quantity + 1)
    V = range(vehicle_quantity)

    # calculate traveling distance and time from node to node
    distance = np.zeros([node_quantity, node_quantity])
    for i in N:
        for j in N:
            if i == j:
                distance[i, j] = big_m
            else:
                distance[i, j] = np.hypot(coordinate[i, 0] - coordinate[j, 0], coordinate[i, 1] - coordinate[j, 1])
    
    travel_time = np.zeros([node_quantity, node_quantity])
    for i in N:
        for j in N:
            travel_time[i, j] = time_per_distance * np.hypot(coordinate[i, 0] - coordinate[j, 0], coordinate[i, 1] - coordinate[j, 1])

    # writing mathematical formulation in code
    model = Model("VRPTW")
    x = model.addVars(node_quantity, node_quantity, vehicle_quantity, vtype=GRB.BINARY)
    s = model.addVars(node_quantity, vehicle_quantity, vtype=GRB.CONTINUOUS)
    y = model.addVars(node_quantity, vehicle_quantity, vtype=GRB.BINARY)
    e = model.addVars(node_quantity, vtype=GRB.CONTINUOUS)
    l = model.addVars(node_quantity, vtype=GRB.CONTINUOUS)
    

    model.modelSense = GRB.MINIMIZE
    model.setObjective(
        quicksum(x[i, j, k] * distance[i, j] * cost_per_distance for i in N for j in N for k in V) +
        quicksum(early_penalty_weight[i] * e[i] for i in C) +
        quicksum(late_penalty_weight[i] * l[i] for i in C)
    )
    
    # Constraints
    # Each customer is visited exactly once
    model.addConstrs(quicksum(x[i, j, k] for j in N for k in V) == 1 for i in C)
    
    # Each vehicle starts from depot
    model.addConstrs(quicksum(x[0, j, k] for j in N) == 1 for k in V)
    
    # Each vehicle ends at depot
    model.addConstrs(quicksum(x[i, customer_quantity + 1, k] for i in N) == 1 for k in V)
    
    # Flow conservation
    model.addConstrs(quicksum(x[i, h, k] for i in N) - quicksum(x[h, j, k] for j in N) == 0 for h in C for k in V)
    
    # Vehicle capacity
    model.addConstrs(quicksum(demand[i] * quicksum(x[i, j, k] for j in N) for i in C) <= vehicle_capacity for k in V)
    
    # Define y variables based on x variables
    model.addConstrs(quicksum(x[i, j, k] for j in N) == y[i, k] for i in C for k in V)
    
    # Time consistency
    model.addConstrs(s[i, k] + travel_time[i, j] + service_duration[i] - big_m * (1 - x[i, j, k]) <= s[j, k] for i in N for j in N for k in V)
    
    # Soft time window constraints
    # Early arrival: s[i, k] + e[i] >= time_window[i, 0] * y[i, k]
    for i in C:
        for k in V:
            model.addConstr(s[i, k] + e[i] >= time_window[i, 0] * y[i, k])
    
    # Late arrival: s[i, k] - l[i] <= time_window[i, 1] + big_m * (1 - y[i, k])
    for i in C:
        for k in V:
            model.addConstr(s[i, k] - l[i] <= time_window[i, 1] + big_m * (1 - y[i, k]))
    
    # Hard time window for depot
    model.addConstrs(s[i, k] >= time_window[i, 0] for i in [0, customer_quantity + 1] for k in V)
    model.addConstrs(s[i, k] <= time_window[i, 1] for i in [0, customer_quantity + 1] for k in V)
    
    # Non-negativity constraints
    model.addConstrs(s[i, k] >= 0 for i in N for k in V)
    model.addConstrs(e[i] >= 0 for i in N)
    model.addConstrs(l[i] >= 0 for i in N)

    # set timelimit and start solving
    model.Params.Timelimit = timelimit
    model.optimize()

    # obtain the results
    is_feasible = True
    obj = 0
    runtime = model.Runtime
    mip_gap = GRB.INFINITY
    result_arc = np.zeros([vehicle_quantity, node_quantity, node_quantity], dtype=int)
    result_arrival_time = np.zeros([node_quantity, vehicle_quantity])
    result_early_deviation = np.zeros(node_quantity)
    result_late_deviation = np.zeros(node_quantity)

    for k in V:
        for i in N:
            for j in N:
                try:
                    result_arc[k, i, j] = round(x[i, j, k].X)
                except:
                    is_feasible = False
                    break

    for k in V:
        for i in N:
            try:
                result_arrival_time[i, k] = s[i, k].X
            except:
                is_feasible = False
                break
    
    for i in N:
        try:
            result_early_deviation[i] = e[i].X
            result_late_deviation[i] = l[i].X
        except:
            is_feasible = False
            break

    try:
        obj = model.getObjective().getValue()
        mip_gap = model.MIPGap
    except:
        is_feasible = False

    return is_feasible, obj, result_arc, result_arrival_time, result_early_deviation, result_late_deviation, runtime, mip_gap
