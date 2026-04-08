import numpy as np
from gurobipy import GRB, Model, quicksum
import math

def solve_VRPTW_multiobjective(
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
    timelimit: float,
    vehicle_ratings: np.ndarray,
    method: str = 'blended', # 'blended' or 'hierarchical'
    weights: list = [1.0, 1.0] # weights for [distance, lateness] in blended method
):
    """
    Multiobjective VRPTW using Gurobi's multiobjective features.
    Objectives:
    1. Distance Traveled (and vehicle use cost)
    2. Time Lateness
    """
    
    node_quantity = coordinate.shape[0]
    customer_quantity = node_quantity - 2

    N = range(node_quantity)
    C = range(1, customer_quantity + 1)
    V = range(vehicle_quantity)

    # Calculate vehicle usage probability based on rating
    vehicle_usage_probability = np.zeros(vehicle_quantity)
    for k in V:
        x_rating = vehicle_ratings[k]
        vehicle_usage_probability[k] = 0.2 + (0.8 - 0.2) * (1 / (1 + math.exp(-2 * (x_rating - 4))))
    
    base_vehicle_use_cost = 1000.0
    vehicle_use_cost = np.zeros(vehicle_quantity)
    for k in V:
        if vehicle_usage_probability[k] > 0.001:
            vehicle_use_cost[k] = base_vehicle_use_cost / vehicle_usage_probability[k]
        else:
            vehicle_use_cost[k] = big_m

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

    model = Model("VRPTW_Multiobjective")
    x = model.addVars(node_quantity, node_quantity, vehicle_quantity, vtype=GRB.BINARY, name="x")
    s = model.addVars(node_quantity, vehicle_quantity, vtype=GRB.CONTINUOUS, name="s")
    y = model.addVars(node_quantity, vehicle_quantity, vtype=GRB.BINARY, name="y")
    e = model.addVars(node_quantity, vtype=GRB.CONTINUOUS, name="e")
    l = model.addVars(node_quantity, vtype=GRB.CONTINUOUS, name="l")
    
    # Define Objective 1: Distance and Vehicle Cost
    obj1 = quicksum(x[i, j, k] * distance[i, j] * cost_per_distance for i in N for j in N for k in V) + \
           quicksum(vehicle_use_cost[k] * quicksum(x[0, j, k] for j in N) for k in V)
    
    # Define Objective 2: Time Lateness
    obj2 = quicksum(l[i] for i in C)

    model.ModelSense = GRB.MINIMIZE

    if method == 'blended':
        # Blended method: set multiple objectives with weights
        model.setObjectiveN(obj1, index=0, weight=weights[0], name="Distance_Cost")
        model.setObjectiveN(obj2, index=1, weight=weights[1], name="Lateness")
    elif method == 'hierarchical':
        # Hierarchical method: set priorities (higher index/priority is solved first)
        # Priority 1: Distance (Priority=2), Priority 2: Lateness (Priority=1)
        model.setObjectiveN(obj1, index=0, priority=2, name="Distance_Cost")
        model.setObjectiveN(obj2, index=1, priority=1, name="Lateness")
    else:
        raise ValueError("Method must be 'blended' or 'hierarchical'")

    # Constraints (same as ori)
    model.addConstrs((quicksum(x[i, j, k] for j in N for k in V) == 1 for i in C), name="visit_once")
    model.addConstrs((quicksum(x[0, j, k] for j in N) == 1 for k in V), name="start_depot")
    model.addConstrs((quicksum(x[i, customer_quantity + 1, k] for i in N) == 1 for k in V), name="end_depot")
    model.addConstrs((quicksum(x[i, h, k] for i in N) - quicksum(x[h, j, k] for j in N) == 0 for h in C for k in V), name="flow")
    model.addConstrs((quicksum(demand[i] * quicksum(x[i, j, k] for j in N) for i in C) <= vehicle_capacity for k in V), name="capacity")
    model.addConstrs((quicksum(x[i, j, k] for j in N) == y[i, k] for i in C for k in V), name="y_def")
    model.addConstrs((s[i, k] + travel_time[i, j] + service_duration[i] - big_m * (1 - x[i, j, k]) <= s[j, k] for i in N for j in N for k in V), name="time_consist")
    
    for i in C:
        for k in V:
            model.addConstr(s[i, k] + e[i] >= time_window[i, 0] * y[i, k], name=f"early_{i}_{k}")
            model.addConstr(s[i, k] - l[i] <= time_window[i, 1] + big_m * (1 - y[i, k]), name=f"late_{i}_{k}")
    
    model.addConstrs((s[i, k] >= time_window[i, 0] for i in [0, customer_quantity + 1] for k in V), name="depot_time_min")
    model.addConstrs((s[i, k] <= time_window[i, 1] for i in [0, customer_quantity + 1] for k in V), name="depot_time_max")
    
    model.addConstrs((s[i, k] >= 0 for i in N for k in V), name="s_pos")
    model.addConstrs((e[i] >= 0 for i in N), name="e_pos")
    model.addConstrs((l[i] >= 0 for i in N), name="l_pos")

    model.Params.Timelimit = timelimit
    model.optimize()

    # Check if a solution was found
    if model.SolCount == 0:
        print(f"No solution found. Status: {model.Status}")
        return False, 0, None, None, None, None, None, model.Runtime, 0

    is_feasible = True
    # obtain the results
    # For multi-objective models di gurobi, we should use the ObjNVal attribute for each objective
    model.setParam(GRB.Param.ObjNumber, 0)
    obj = model.ObjNVal # Primary objective value (Distance_Cost)
    runtime = model.Runtime
    mip_gap = getattr(model, "MIPGap", 0)
    
    result_arc = np.zeros([vehicle_quantity, node_quantity, node_quantity], dtype=int)
    result_arrival_time = np.zeros([node_quantity, vehicle_quantity])
    result_early_deviation = np.zeros(node_quantity)
    result_late_deviation = np.zeros(node_quantity)
    result_late_deviation_per_vehicle = np.zeros(vehicle_quantity)

    for k in V:
        for i in N:
            for j in N:
                result_arc[k, i, j] = round(x[i, j, k].X)
            result_arrival_time[i, k] = s[i, k].X
    
    for i in N:
        result_early_deviation[i] = e[i].X
        result_late_deviation[i] = l[i].X

    for k in V:
        for i in C:
            if y[i, k].X > 0.5:
                result_late_deviation_per_vehicle[k] += l[i].X

    return is_feasible, obj, result_arc, result_arrival_time, result_early_deviation, result_late_deviation, result_late_deviation_per_vehicle, runtime, mip_gap
