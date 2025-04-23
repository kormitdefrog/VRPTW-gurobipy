import numpy as np
from gurobipy import *
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection

import fileutil
import core
from visual import plot_solution

def solve_and_save(xmlpath: str, cpd: float, tpd: float, early_weight: float, late_weight: float, big_m: float, tlimit: float, name: str):
    """
    solve VRPTW with soft time window and save the result
    """
    coord, tw, d, service_dur, v_quant, v_cap, premium_customers = fileutil.load_dataset(xmlpath)
    node_quantity = coord.shape[0]
    customer_quantity = node_quantity - 2
    
    # Initialize penalty weights
    early_penalty_weight = np.zeros(node_quantity)
    late_penalty_weight = np.zeros(node_quantity)
    
    # Set penalty weights only for premium customers
    for i in range(1, customer_quantity + 1):
        # No early penalties for any customers as per requirements
        early_penalty_weight[i] = 0
        
        # Late penalties only for premium customers
        if premium_customers[i]:
            late_penalty_weight[i] = late_weight
        else:
            late_penalty_weight[i] = 0
    
    is_feasible, obj, arc, time, early_dev, late_dev, runtime, gap = core.solve_VRPTW(
        coord, tw, d, service_dur, v_quant, v_cap, 
        cpd, tpd, early_penalty_weight, late_penalty_weight,
        big_m, tlimit, premium_customers
    )
    
    fileutil.save_raw_result(name, is_feasible, obj, arc, time, coord, tw, d, service_dur, v_quant, v_cap, cpd, tpd, runtime, gap, premium_customers)
    plot_solution_premium(name, is_feasible, obj, arc, time, coord, tw, d, service_dur, v_quant, v_cap, cpd, tpd, runtime, gap, premium_customers, False)
    
    # Print time window violations for debugging/analysis
    if is_feasible:
        print(f"\nTime window violations for {name}:")
        violations_found = False
        for i in range(1, customer_quantity + 1):
            if early_dev[i] > 0 or late_dev[i] > 0:
                violations_found = True
                print(f"Customer {i} (Premium: {premium_customers[i]}):")
                print(f"  Time window: [{tw[i, 0]}, {tw[i, 1]}]")
                
                # Find which vehicle serves this customer
                vehicle = -1
                for k in range(v_quant):
                    if any(arc[k, i, j] == 1 for j in range(node_quantity)):
                        vehicle = k
                        break
                
                if vehicle >= 0:
                    arrival = time[i, vehicle]
                    print(f"  Arrival time: {arrival}")
                    print(f"  Early deviation: {early_dev[i]}")
                    print(f"  Late deviation: {late_dev[i]}")
                    
                    # Calculate penalty contribution
                    early_penalty = early_penalty_weight[i] * early_dev[i]
                    late_penalty = late_penalty_weight[i] * late_dev[i]
                    print(f"  Early penalty: {early_penalty}")
                    print(f"  Late penalty: {late_penalty}")
                    print(f"  Penalty applied: {premium_customers[i]}")
        
        if not violations_found:
            print("  No time window violations found.")

def plot_solution_premium(
    title: str,
    is_feasible: bool,
    objective_value: float,
    arc: np.ndarray,
    arrival_time: np.ndarray,
    coordinate: np.ndarray,
    time_window: np.ndarray,
    demand: np.ndarray,
    service_duration: np.ndarray,
    vehicle_quantity: int,
    vehicle_capacity: float,
    cost_per_distance: float,
    time_per_distance: float,
    solver_runtime: float,
    mip_gap: float,
    premium_customers: np.ndarray,
    show_plot: bool
):
    """
    Modified version of plot_solution that highlights premium customers
    """
    node_quantity = coordinate.shape[0]
    customer_quantity = node_quantity - 2
    N = range(node_quantity)
    C = range(1, customer_quantity + 1)
    V = range(vehicle_quantity)
    
    # Calculate chronological information
    chrono_info = []
    for k in V:
        chrono_info.append(np.zeros([1, 5]))
        current_node = 0
        current_time = 0
        current_cargo = 0
        first_iter = True
        while True:
            next_node = -1
            for j in N:
                if arc[k, current_node, j] == 1:
                    next_node = j
                    break
            if next_node == -1:
                break
            
            travel_time = time_per_distance * np.hypot(coordinate[current_node, 0] - coordinate[next_node, 0], coordinate[current_node, 1] - coordinate[next_node, 1])
            current_time += travel_time + service_duration[current_node]
            if next_node != customer_quantity + 1:
                current_cargo += demand[next_node]
            
            if first_iter:
                first_iter = False
                chrono_info[k] = np.array([[current_time, next_node, current_cargo, coordinate[next_node, 0], coordinate[next_node, 1]]])
            else:
                chrono_info[k] = np.vstack((chrono_info[k], np.array([[current_time, next_node, current_cargo, coordinate[next_node, 0], coordinate[next_node, 1]]])))
            
            current_node = next_node
    
    # Create pretty print output
    fileutil.pretty_print(title, customer_quantity, is_feasible, objective_value, vehicle_quantity, vehicle_capacity, cost_per_distance, time_per_distance, solver_runtime, chrono_info, mip_gap, premium_customers)
    
    # Create visualization
    if is_feasible:
        plt.figure(figsize=(10, 10))
        
        # Plot depot
        plt.scatter(coordinate[0, 0], coordinate[0, 1], c='black', s=100, marker='s')
        plt.annotate("Depot", (coordinate[0, 0], coordinate[0, 1]), xytext=(5, 5), textcoords='offset points')
        
        # Plot customers with different colors for premium and non-premium
        for i in C:
            if premium_customers[i]:
                plt.scatter(coordinate[i, 0], coordinate[i, 1], c='red', s=100)
                plt.annotate(f"{i} (P)", (coordinate[i, 0], coordinate[i, 1]), xytext=(5, 5), textcoords='offset points')
            else:
                plt.scatter(coordinate[i, 0], coordinate[i, 1], c='blue', s=100)
                plt.annotate(f"{i}", (coordinate[i, 0], coordinate[i, 1]), xytext=(5, 5), textcoords='offset points')
        
        # Plot routes
        colors = ['green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        for k in V:
            if chrono_info[k].shape[0] > 1:  # If vehicle k is used
                route_x = [coordinate[0, 0]]  # Start at depot
                route_y = [coordinate[0, 1]]
                
                for i in range(chrono_info[k].shape[0]):
                    node = int(chrono_info[k][i, 1])
                    route_x.append(coordinate[node, 0])
                    route_y.append(coordinate[node, 1])
                
                route_x.append(coordinate[0, 0])  # Return to depot
                route_y.append(coordinate[0, 1])
                
                plt.plot(route_x, route_y, c=colors[k % len(colors)], linewidth=2, label=f"Vehicle {k}")
        
        # Add legend for premium and non-premium customers
        premium_patch = mpatches.Patch(color='red', label='Premium Customer')
        non_premium_patch = mpatches.Patch(color='blue', label='Non-Premium Customer')
        depot_patch = mpatches.Patch(color='black', label='Depot')
        
        # Add vehicle routes to legend
        handles = [premium_patch, non_premium_patch, depot_patch]
        for k in V:
            if chrono_info[k].shape[0] > 1:  # If vehicle k is used
                handles.append(mpatches.Patch(color=colors[k % len(colors)], label=f"Vehicle {k}"))
        
        plt.legend(handles=handles, loc='upper right')
        
        plt.title(f"{title} - Premium Customer Model\nObjective: {objective_value:.2f}, Runtime: {solver_runtime:.2f}s")
        plt.grid(True)
        
        # Save the plot
        if not os.path.exists("./result"):
            os.mkdir("result")
        plt.savefig(f"./result/plot-{title}-premium.png")
        
        if show_plot:
            plt.show()
        plt.close()

def solve_simple_test():
    solve_and_save("./dataset/simple/VRPSTW_6nodes.xml", 1, 0.5, 10, 20, 1e6, 3600, "VRPSTW6-TPD0.5")
    solve_and_save("./dataset/simple/VRPSTW_6nodes.xml", 1, 1.0, 10, 20, 1e6, 3600, "VRPSTW6-TPD1.0")

def solve_solomon(setname: str, cpd: float, tpd: float, early_weight: float, late_weight: float, big_m: float, tlimit: float, start: int, end: int):
    I = range(start, end + 1)
    for i in I:
        print("======================================================")
        print(setname, i)
        print("======================================================")
        if i < 10:
            xmlpath_25 = "./dataset/solomon-1987-" + setname + "/" + setname.upper() + "0" + str(i) + "_025.xml"
            xmlpath_50 = "./dataset/solomon-1987-" + setname + "/" + setname.upper() + "0" + str(i) + "_050.xml"
            xmlpath_100 = "./dataset/solomon-1987-" + setname + "/" + setname.upper() + "0" + str(i) + "_100.xml"
            solve_and_save(xmlpath_25, cpd, tpd, early_weight, late_weight, big_m, tlimit, setname.upper() + "0" + str(i) + "_025")
            solve_and_save(xmlpath_50, cpd, tpd, early_weight, late_weight, big_m, tlimit, setname.upper() + "0" + str(i) + "_050")
            solve_and_save(xmlpath_100, cpd, tpd, early_weight, late_weight, big_m, tlimit, setname.upper() + "0" + str(i) + "_100")
        else:
            xmlpath_25 = "./dataset/solomon-1987-" + setname + "/" + setname.upper() + str(i) + "_025.xml"
            xmlpath_50 = "./dataset/solomon-1987-" + setname + "/" + setname.upper() + str(i) + "_050.xml"
            xmlpath_100 = "./dataset/solomon-1987-" + setname + "/" + setname.upper() + str(i) + "_100.xml"
            solve_and_save(xmlpath_25, cpd, tpd, early_weight, late_weight, big_m, tlimit, setname.upper() + str(i) + "_025")
            solve_and_save(xmlpath_50, cpd, tpd, early_weight, late_weight, big_m, tlimit, setname.upper() + str(i) + "_050")
            solve_and_save(xmlpath_100, cpd, tpd, early_weight, late_weight, big_m, tlimit, setname.upper() + str(i) + "_100")

if __name__ == "__main__":
    solve_simple_test()
    solve_solomon("c1", 1, 1, 10, 20, 1e6, 3600, 1, 9)
    solve_solomon("c2", 1, 1, 10, 20, 1e6, 3600, 1, 8)
    solve_solomon("r1", 1, 1, 10, 20, 1e6, 3600, 1, 12)
    solve_solomon("r2", 1, 1, 10, 20, 1e6, 3600, 1, 11)
    solve_solomon("rc1", 1, 1, 10, 20, 1e6, 3600, 1, 3)
    solve_solomon("rc2", 1, 1, 10, 20, 1e6, 3600, 1, 8)
