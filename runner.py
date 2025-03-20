import multiprocessing
import subprocess
import pandas as pd

def run_experiment(run_id):
    print(f"Starting Run {run_id+1}...")
    result = subprocess.run(["python", "rand.py"], capture_output=True, text=True)
    
    output_lines = result.stdout.split("\n")
    idi_ratio = None
    for line in output_lines:
        if "IDI Ratio:" in line:
            idi_ratio = float(line.split(":")[1].strip())
            break

    print(f"Run {run_id+1} Completed - IDI Ratio: {idi_ratio}")
    return {"Run": run_id + 1, "IDI Ratio": idi_ratio}

if __name__ == "__main__":
    num_runs = 10

    with multiprocessing.Pool(processes=num_runs) as pool:
        results = pool.map(run_experiment, range(num_runs))

    # Convert results to DataFrame
    df = pd.DataFrame(results)

    # Calculate the average IDI ratio
    average_idi = df["IDI Ratio"].mean()
    
    # Append the average as a new row
    avg_row = pd.DataFrame([{"Run": "Average", "IDI Ratio": average_idi}])
    df = pd.concat([df, avg_row], ignore_index=True)

    # Save results to CSV
    df.to_csv("results.csv", index=False)
    print(f"All runs completed. Results saved to results.csv")
    print(f"Average IDI Ratio: {average_idi:.4f}")
