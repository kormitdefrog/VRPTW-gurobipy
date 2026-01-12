from core_multiobjective import solve_VRPTW_multiobjective
from fileutil import load_dataset, save_raw_result
from visual import plot_solution
import numpy as np
import os

def run_multiobjective_test(xmlpath, name_prefix, method, weights=[1.0, 1.0], tlimit=60):
    coord, tw, d, service_dur, v_quant, v_cap = load_dataset(xmlpath)
    node_quantity = coord.shape[0]
    customer_quantity = node_quantity - 2
    
    # Use very small penalty weights in the model constraints to allow lateness to be an objective
    early_penalty_weight = np.zeros(node_quantity)
    late_penalty_weight = np.zeros(node_quantity)
    
    vehicle_ratings = np.full(v_quant, 3.0)
    
    print(f"\n--- Running {method} method for {name_prefix} ---")
    if method == 'blended':
        print(f"Weights: Distance={weights[0]}, Lateness={weights[1]}")
    
    is_feasible, obj, arc, time, early_dev, late_dev, late_dev_per_vehicle, runtime, gap = solve_VRPTW_multiobjective(
        coord, tw, d, service_dur, v_quant, v_cap, 
        1.0, 1.0, early_penalty_weight, late_penalty_weight,
        1e6, tlimit, vehicle_ratings, method=method, weights=weights
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
        
        print(f"Results:")
        print(f"  Total Distance: {total_dist:.2f}")
        print(f"  Total Lateness: {total_lateness:.2f}")
        print(f"  Primary Obj: {obj:.2f}")
        return total_dist, total_lateness
    else:
        print(f"Optimization failed.")
        return None, None

if __name__ == "__main__":
    # Using a slightly larger Solomon instance to see trade-offs
    # RC101_025 has 25 customers, more likely to have trade-offs than 6 nodes
    test_file = "./dataset/solomon-1987-rc1/RC101_025.xml"
    
    results = []
    
    # 1. Blended - Focus on Distance
    d, l = run_multiobjective_test(test_file, "RC101", "blended", weights=[1.0, 0.001], tlimit=30)
    results.append(("Blended (Dist Focus)", d, l))
    
    # 2. Blended - Balanced
    d, l = run_multiobjective_test(test_file, "RC101", "blended", weights=[1.0, 1.0], tlimit=30)
    results.append(("Blended (Balanced)", d, l))
    
    # 3. Blended - Focus on Lateness
    d, l = run_multiobjective_test(test_file, "RC101", "blended", weights=[0.001, 1.0], tlimit=30)
    results.append(("Blended (Late Focus)", d, l))
    
    # 4. Hierarchical - Distance first
    print("\n--- Hierarchical: Distance Priority ---")
    d, l = run_multiobjective_test(test_file, "RC101", "hierarchical", tlimit=30)
    results.append(("Hierarchical (Dist First)", d, l))

    print("\n" + "="*30)
    print("SUMMARY OF TRADE-OFFS")
    print(f"{'Method':<25} | {'Distance':<10} | {'Lateness':<10}")
    print("-" * 50)
    for name, d, l in results:
        if d is not None:
            print(f"{name:<25} | {d:<10.2f} | {l:<10.2f}")
