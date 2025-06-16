from core import solve_VRPTW
from fileutil import load_dataset, save_raw_result
from visual import plot_solution
import numpy as np


def solve_and_save(xmlpath: str, cpd: float, tpd: float, early_weight: float, late_weight: float, big_m: float, tlimit: float, name: str, vehicle_ratings: np.ndarray):

    coord, tw, d, service_dur, v_quant, v_cap = load_dataset(xmlpath)

    # Create penalty weights
    node_quantity = coord.shape[0]
    customer_quantity = node_quantity - 2
    
    # Set early and late penalty weights
    early_penalty_weight = np.zeros(node_quantity)
    late_penalty_weight = np.zeros(node_quantity)
    
    # Set penalty weights only for customers (not for depot)
    for i in range(1, customer_quantity + 1):
        early_penalty_weight[i] = early_weight
        late_penalty_weight[i] = late_weight

    is_feasible, obj, arc, time, early_dev, late_dev, late_dev_per_vehicle, runtime, gap = solve_VRPTW(
        coord, tw, d, service_dur, v_quant, v_cap, 
        cpd, tpd, early_penalty_weight, late_penalty_weight,
        big_m, tlimit, vehicle_ratings
    )

    save_raw_result(name, is_feasible, obj, arc, time, coord, tw, d, service_dur, v_quant, v_cap, cpd, tpd, runtime, gap)
    plot_solution(name, is_feasible, obj, arc, time, coord, tw, d, service_dur, v_quant, v_cap, cpd, tpd, runtime, gap, False)

    # Print time window violations for debugging/analysis
    if is_feasible:
        print(f"\nTime window violations for {name}:")
        violations_found = False
        for i in range(1, customer_quantity + 1):
            if early_dev[i] > 0 or late_dev[i] > 0:
                violations_found = True
                print(f"Customer {i}:")
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
        
        if not violations_found:
            print("  No time window violations found.")

        # Apply lateness penalty to vehicle ratings
        print(f"\nApplying lateness penalties for {name}:")
        updated_vehicle_ratings = np.copy(vehicle_ratings)
        for k in range(v_quant):
            if late_dev_per_vehicle[k] > 0:
                updated_vehicle_ratings[k] = max(1.0, updated_vehicle_ratings[k] - 0.1) # Ensure rating doesn\'t go below 1
                print(f"  Vehicle {k} was late. Old rating: {vehicle_ratings[k]:.1f}, New rating: {updated_vehicle_ratings[k]:.1f}")
        return updated_vehicle_ratings
    return vehicle_ratings # Return original ratings if not feasible

def solve_solomon(setname: str, cpd: float, tpd: float, early_weight: float, late_weight: float, big_m: float, tlimit: float, start: int, end: int):

    I = range(start, end + 1)
    for i in I:
        print("======================================================")
        print(setname, i)
        print("======================================================")
        
        # Dynamically determine vehicle quantity and initialize ratings
        # Load a dummy dataset to get v_quant, then re-load for actual solve_and_save
        # This is a workaround since load_dataset is called inside solve_and_save
        # A better approach would be to refactor load_dataset to return v_quant separately
        # For now, let\'s load the dataset once to get v_quant
        if i < 10:
            xmlpath_25_dummy = "./dataset/solomon-1987-" + setname + "/" + setname.upper() + "0" + str(i) + "_025.xml"
        else:
            xmlpath_25_dummy = "./dataset/solomon-1987-" + setname + "/" + setname.upper() + str(i) + "_025.xml"
        
        _, _, _, _, v_quant_dummy, _ = load_dataset(xmlpath_25_dummy)
        initial_vehicle_ratings = np.full(v_quant_dummy, 3.0) # Initialize all vehicles to rating 3.0

        if i < 10:
            xmlpath_25 = "./dataset/solomon-1987-" + setname + "/" + setname.upper() + "0" + str(i) + "_025.xml"
            xmlpath_50 = "./dataset/solomon-1987-" + setname + "/" + setname.upper() + "0" + str(i) + "_050.xml"
            xmlpath_100 = "./dataset/solomon-1987-" + setname + "/" + setname.upper() + "0" + str(i) + "_100.xml"
            
            # Pass and update vehicle ratings
            initial_vehicle_ratings = solve_and_save(xmlpath_25, cpd, tpd, early_weight, late_weight, big_m, tlimit, setname.upper() + "0" + str(i) + "_025", initial_vehicle_ratings)
            initial_vehicle_ratings = solve_and_save(xmlpath_50, cpd, tpd, early_weight, late_weight, big_m, tlimit, setname.upper() + "0" + str(i) + "_050", initial_vehicle_ratings)
            initial_vehicle_ratings = solve_and_save(xmlpath_100, cpd, tpd, early_weight, late_weight, big_m, tlimit, setname.upper() + "0" + str(i) + "_100", initial_vehicle_ratings)
        else:
            xmlpath_25 = "./dataset/solomon-1987-" + setname + "/" + setname.upper() + str(i) + "_025.xml"
            xmlpath_50 = "./dataset/solomon-1987-" + setname + "/" + setname.upper() + str(i) + "_050.xml"
            xmlpath_100 = "./dataset/solomon-1987-" + setname + "/" + setname.upper() + str(i) + "_100.xml"
            
            # Pass and update vehicle ratings
            initial_vehicle_ratings = solve_and_save(xmlpath_25, cpd, tpd, early_weight, late_weight, big_m, tlimit, setname.upper() + str(i) + "_025", initial_vehicle_ratings)
            initial_vehicle_ratings = solve_and_save(xmlpath_50, cpd, tpd, early_weight, late_weight, big_m, tlimit, setname.upper() + str(i) + "_050", initial_vehicle_ratings)
            initial_vehicle_ratings = solve_and_save(xmlpath_100, cpd, tpd, early_weight, late_weight, big_m, tlimit, setname.upper() + str(i) + "_100", initial_vehicle_ratings)


def solve_simple_test():

    # Dynamically determine vehicle quantity and initialize ratings
    _, _, _, _, v_quant_dummy, _ = load_dataset("./dataset/simple/TW10.xml") # Use one of the simple datasets to get v_quant
    initial_vehicle_ratings = np.full(v_quant_dummy, 3.0) # Initialize all vehicles to rating 3.0

    initial_vehicle_ratings = solve_and_save("./dataset/simple/TW10.xml", 1, 0.5, 10, 20, 1e6, 3600, "TW10-TPD0.5", initial_vehicle_ratings)
    initial_vehicle_ratings = solve_and_save("./dataset/simple/TW10.xml", 1, 1.0, 10, 20, 1e6, 3600, "TW10-TPD1.0", initial_vehicle_ratings)
    initial_vehicle_ratings = solve_and_save("./dataset/simple/TW60.xml", 1, 0.5, 10, 20, 1e6, 3600, "TW60-TPD0.5", initial_vehicle_ratings)
    initial_vehicle_ratings = solve_and_save("./dataset/simple/TW60.xml", 1, 1.0, 10, 20, 1e6, 3600, "TW60-TPD1.0", initial_vehicle_ratings)
    initial_vehicle_ratings = solve_and_save("./dataset/simple/VRPSTW_6nodes.xml", 1, 0.5, 10, 20, 1e6, 3600, "VRPSTW6-TPD0.5", initial_vehicle_ratings)
    initial_vehicle_ratings = solve_and_save("./dataset/simple/VRPSTW_6nodes.xml", 1, 1.0, 10, 20, 1e6, 3600, "VRPSTW6-TPD1.0", initial_vehicle_ratings)


if __name__ == "__main__":

    solve_simple_test()
    # solve_solomon("c1", 1, 1, 10, 20, 1e6, 3600, 1, 9)
    # solve_solomon("c2", 1, 1, 10, 20, 1e6, 3600, 1, 8)
    # solve_solomon("r1", 1, 1, 10, 20, 1e6, 3600, 1, 12)
    # solve_solomon("r2", 1, 1, 10, 20, 1e6, 3600, 1, 11)
    # solve_solomon("rc1", 1, 1, 10, 20, 1e6, 3600, 1, 3)
    # solve_solomon("rc2", 1, 1, 10, 20, 1e6, 3600, 1, 8)


