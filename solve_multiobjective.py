from core_multiobjective import solve_VRPTW_multiobjective
from fileutil import load_dataset, save_raw_result
from visual import plot_solution
import numpy as np
import os

def run_multiobjective_test(xmlpath, name_prefix, method, weights=[1.0, 1.0]):
    coord, tw, d, service_dur, v_quant, v_cap = load_dataset(xmlpath)
    node_quantity = coord.shape[0]
    customer_quantity = node_quantity - 2
    
    early_penalty_weight = np.zeros(node_quantity)
    late_penalty_weight = np.zeros(node_quantity)
    for i in range(1, customer_quantity + 1):
        early_penalty_weight[i] = 10.0
        late_penalty_weight[i] = 20.0
        
    vehicle_ratings = np.full(v_quant, 3.0)
    
    print(f"\nRunning {method} method for {name_prefix}...")
    if method == 'blended':
        print(f"Weights: Distance={weights[0]}, Lateness={weights[1]}")
    
    is_feasible, obj, arc, time, early_dev, late_dev, late_dev_per_vehicle, runtime, gap = solve_VRPTW_multiobjective(
        coord, tw, d, service_dur, v_quant, v_cap, 
        1.0, 1.0, early_penalty_weight, late_penalty_weight,
        1e6, 60, vehicle_ratings, method=method, weights=weights
    )

    if is_feasible:
        name = f"{name_prefix}_{method}"
        if method == 'blended':
            name += f"_w{weights[0]}_{weights[1]}"
        
        # Ensure result directory exists
        os.makedirs("./result/raw", exist_ok=True)
        os.makedirs("./result/fig", exist_ok=True)
        
        save_raw_result(name, is_feasible, obj, arc, time, coord, tw, d, service_dur, v_quant, v_cap, 1.0, 1.0, runtime, gap)
        plot_solution(name, is_feasible, obj, arc, time, coord, tw, d, service_dur, v_quant, v_cap, 1.0, 1.0, runtime, gap, False)
        
        total_dist = np.sum([arc[k, i, j] * np.hypot(coord[i, 0] - coord[j, 0], coord[i, 1] - coord[j, 1]) for k in range(v_quant) for i in range(node_quantity) for j in range(node_quantity)])
        total_lateness = np.sum(late_dev)
        
        print(f"Results for {name}:")
        print(f"  Total Distance: {total_dist:.2f}")
        print(f"  Total Lateness: {total_lateness:.2f}")
        print(f"  Objective Value: {obj:.2f}")
    else:
        print(f"Optimization failed for {name_prefix} {method}")

if __name__ == "__main__":
    test_file = "./dataset/simple/VRPSTW_6nodes.xml"
    
    # 1. Blended Method - Case 1: Equal weights
    run_multiobjective_test(test_file, "test", "blended", weights=[1.0, 1.0])
    
    # 2. Blended Method - Case 2: Focus on Distance
    run_multiobjective_test(test_file, "test", "blended", weights=[100.0, 1.0])
    
    # 3. Blended Method - Case 3: Focus on Lateness
    run_multiobjective_test(test_file, "test", "blended", weights=[1.0, 100.0])
    
    # 4. Hierarchical Method
    run_multiobjective_test(test_file, "test", "hierarchical")