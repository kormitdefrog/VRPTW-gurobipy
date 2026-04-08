from core_multiobjective import solve_VRPTW_multiobjective
from fileutil import load_dataset, save_raw_result
from visual import plot_solution
import numpy as np
import matplotlib.pyplot as plt
import os

tlimit = 900  # Time limit for optimization in seconds
def run_multiobjective_test(xmlpath, name_prefix, method, weights=[1.0, 1.0], tlimit=tlimit):
    coord, tw, d, service_dur, v_quant, v_cap = load_dataset(xmlpath)
    node_quantity = coord.shape[0]
    customer_quantity = node_quantity - 2
    
    # Use very small penalty weights in the model constraints to allow lateness to be an objective
    early_penalty_weight = np.zeros(node_quantity)
    late_penalty_weight = np.zeros(node_quantity)
    
    # Initialize vehicle ratings with random integers ranging from 1 to 5
    vehicle_ratings = np.random.randint(1, 6, size=v_quant).astype(float)
    print(f"Initial Vehicle Ratings: {vehicle_ratings}")
    
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
        # plot_solution is now called inside the rating update block to include rating_history
        total_dist = np.sum([arc[k, i, j] * np.hypot(coord[i, 0] - coord[j, 0], coord[i, 1] - coord[j, 1]) for k in range(v_quant) for i in range(node_quantity) for j in range(node_quantity)])
        total_lateness = np.sum(late_dev)
        
        print(f"Results:")
        print(f"  Total Distance: {total_dist:.2f}")
        print(f"  Total Lateness: {total_lateness:.2f}")
        print(f"  Primary Obj: {obj:.2f}")

        # Update vehicle ratings after each node based on lateness
        # The rating is the cumulative average of all ratings (initial + one per node)
        new_vehicle_ratings = vehicle_ratings.copy()
        rating_history = [] # List of (vehicle_id, time, rating)
        
        for k in range(v_quant):
            # Initial rating at time 0
            rating_history.append((k, 0.0, vehicle_ratings[k]))
            
            # Find nodes visited by vehicle k (excluding depot start/end)
            visited_nodes = []
            for i in range(1, node_quantity - 1):
                if np.any(arc[k, i, :]) or np.any(arc[k, :, i]):
                    visited_nodes.append(i)
            
            # Sort visited nodes by arrival time to process them in order
            visited_nodes.sort(key=lambda n: time[n, k])
            
            # Track sum and count for cumulative average
            rating_sum = vehicle_ratings[k]
            rating_count = 1
            
            for node in visited_nodes:
                is_late = late_dev[node] > 1e-4
                if is_late:
                    # Late: random rating 1-3
                    new_val = np.random.randint(1, 4)
                else:
                    # Not late: random rating 4-5
                    new_val = np.random.randint(4, 6)
                
                rating_sum += new_val
                rating_count += 1
                current_avg = rating_sum / rating_count
                new_vehicle_ratings[k] = current_avg
                
                # Record rating after service completion at this node
                # time[node, k] is arrival time, service_dur[node] is service duration
                rating_history.append((k, time[node, k] + service_dur[node], current_avg))
        
        print(f"Updated Vehicle Ratings: {np.round(new_vehicle_ratings, 2)}")
        
        # Pass rating_history to plot_solution
        # Note to self cad, We need to modify plot_solution signature in visual.py first or use a workaround
        # For now, let's assume we modify visual.py to accept it HAHAHAH
        plot_solution(name, is_feasible, obj, arc, time, coord, tw, d, service_dur, v_quant, v_cap, 1.0, 1.0, runtime, gap, False, rating_history=rating_history)
        
        return total_dist, total_lateness
    else:
        print(f"Optimization failed.")
        return None, None

if __name__ == "__main__":
    test_file, name_prefix = "./dataset/solomon-1987-c1/C101_025.xml","C101_025"
    
    results = []
    
    for i in range (1,10):
        w_dist = i / 10.0
        w_late = 1.0 - w_dist

        label = f"Blended ({w_dist:.1f}/{w_late:.1f})"

        d, l = run_multiobjective_test(test_file, name_prefix, "blended", weights=[w_dist, w_late], tlimit=tlimit)

        results.append((label, d, l))
        print(f"Finished: {label} -> Dist: {d}, Late: {l}")

    # Hierarchical - Distance first
    print(f"\n--- Hierarchical: Distance Priority for {name_prefix}---")
    d, l = run_multiobjective_test(test_file, name_prefix, "hierarchical", tlimit=tlimit)
    results.append(("Hierarchical (Dist First)", d, l))

    print("\n" + "="*30)
    print(f"SUMMARY OF TRADE-OFFS for {name_prefix}")
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
        plt.title(f'Pareto Frontier: Distance vs Lateness ({name_prefix})')
        plt.xlabel('Distance (Minimize)')
        plt.ylabel('Lateness (Minimize)')
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()

        plot_filename = f"pareto_frontier_{name_prefix}.png"
        plt.savefig(plot_filename, dpi=300)
        print(f"\nPlot saved to '{plot_filename}'")
        plt.close()