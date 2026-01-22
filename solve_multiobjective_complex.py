from core_multiobjective import solve_VRPTW_multiobjective
from fileutil import load_dataset, save_raw_result
from visual import plot_solution
import numpy as np
import matplotlib.pyplot as plt
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
    
    for i in range (1,10):
        w_dist = i / 10.0
        w_late = 1.0 - w_dist

        label = f"Blended ({w_dist:.1f}/{w_late:.1f})"

        d, l = run_multiobjective_test(test_file, "RC101", "blended", weights=[w_dist, w_late], tlimit=20)

        results.append((label, d, l))
        print(f"Finished: {label} -> Dist: {d}, Late: {l}")

    # # 1. Blended - Focus on Distance
    # d, l = run_multiobjective_test(test_file, "RC101", "blended", weights=[0.7, 0.3], tlimit=30)
    # results.append(("Blended (Dist Focus)", d, l))
    
    # # 2. Blended - Balanced
    # d, l = run_multiobjective_test(test_file, "RC101", "blended", weights=[0.5, 0.5], tlimit=30)
    # results.append(("Blended (Balanced)", d, l))
    
    # # 3. Blended - Focus on Lateness
    # d, l = run_multiobjective_test(test_file, "RC101", "blended", weights=[0.3, 0.7], tlimit=30)
    # results.append(("Blended (Late Focus)", d, l))
    
    # 4. Hierarchical - Distance first
    print("\n--- Hierarchical: Distance Priority ---")
    d, l = run_multiobjective_test(test_file, "RC101", "hierarchical", tlimit=60)
    results.append(("Hierarchical (Dist First)", d, l))

    print("\n" + "="*30)
    print("SUMMARY OF TRADE-OFFS")
    print(f"{'Method':<25} | {'Distance':<10} | {'Lateness':<10} | {'Total Cost': <10}")
    print("-" * 50)
    for name, d, l in results:
        if d is not None:
            print(f"{name:<25} | {d:<10.2f} | {l:<10.2f} | {d + l:.2f}")
    
    valid_results = [r for r in results if r[1] is not None]

    if valid_results:
        blended_res = [r for r in valid_results if "Blended" in r[0]]
        hier_res = [r for r in valid_results if "Hierarchical" in r[0]]

        plt.figure(figsize=(10,6))

        # A. Plot Blended Points (The Spectrum)
        b_dists = [r[1] for r in blended_res]
        b_lates = [r[2] for r in blended_res]
        plt.scatter(b_dists, b_lates, color='blue', s=100, label='Blended Weights')

        # Connect Blended points to show the frontier curve
        # Sort by distance so the line is smooth
        sorted_pairs = sorted(zip(b_dists, b_lates))
        plt.plot([p[0] for p in sorted_pairs], [p[1] for p in sorted_pairs], 
                 linestyle='--', color='blue', alpha=0.3)

        # Annotate Blended points with their weights
        for r in blended_res:
            label_text = r[0].replace("Blended (", "").replace(")", "")
            plt.annotate(label_text, (r[1], r[2]), 
                         xytext=(5, 5), textcoords='offset points', fontsize=8)

        # B. Plot Hierarchical Point
        if hier_res:
            h_dists = [r[1] for r in hier_res]
            h_lates = [r[2] for r in hier_res]
            plt.scatter(h_dists, h_lates, color='red', marker='*', s=200, label='Hierarchical')

        # C. Labels and Formatting
        plt.title('Pareto Frontier: Distance vs Lateness')
        plt.xlabel('Distance (Minimize)')
        plt.ylabel('Lateness (Minimize)')
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()

        plt.savefig("pareto_frontier_results.png", dpi=300)
        print("\nPlot saved to 'pareto_frontier_results.png'")