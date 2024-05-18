import csv
import pandas as pd
import matplotlib.pyplot as plt
import os

def process_and_plot_csv_data_with_legend_ltr(file_paths, columns, eval_name, output_dir, model_rp, generation_rp):
    # Initialize a DataFrame to store the data
    data = pd.DataFrame(columns=columns)

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Read the last line of each file and append to the DataFrame
    for i, file_path in enumerate(file_paths):
        with open(file_path, 'r') as file:
            last_line = list(csv.reader(file))[-1]
            if i == 0:
                # Set the first file's last line as "RD"
                data.loc["base"] = [float(val) for val in last_line]
            else:
                row_name = f"GEN{i-1}"
                data.loc[row_name] = [float(val) for val in last_line]

    # Save the DataFrame to a CSV file in the output directory
    output_csv_path = os.path.join(output_dir, f'summary_{eval_name}.csv')
    data.to_csv(output_csv_path, index=True)
    print(data)
    
    # Plot each column and save to the specified output directory
    plt.figure(figsize=(12,6))
    for column in columns:
        plt.plot(data.index, data[column], marker='o', label=column)

    plt.title(f'{eval_name}, Model RP: {model_rp}, Batch RP: {generation_rp}')
    plt.xlabel('Generation')
    plt.ylabel('Values')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.legend(loc='lower right')
    plt.ylim(0,1.01)

    output_image_path = os.path.join(output_dir, f'{eval_name}_{model_rp}_{generation_rp}.jpeg')
    plt.savefig(output_image_path)
    plt.show()


def process_and_plot_csv_data_with_legend(file_paths, columns, eval_name, output_dir, model_rp, generation_rp):
    # Initialize a DataFrame to store the data
    data = pd.DataFrame(columns=columns)

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Read the last line of each file and append to the DataFrame
    for i, file_path in enumerate(file_paths):
        with open(file_path, 'r') as file:
            last_line = list(csv.reader(file))[-1]
            if i == 0:
                # Set the first file's last line as "RD"
                data.loc["RD"] = [float(val) for val in last_line]
            elif i == 1:
                # Set the second file's last line as "base"
                data.loc["base"] = [float(val) for val in last_line]
            else:
                row_name = f"GEN{i-2}"
                data.loc[row_name] = [float(val) for val in last_line]

    # Save the DataFrame to a CSV file in the output directory
    output_csv_path = os.path.join(output_dir, f'summary_{eval_name}.csv')
    data.to_csv(output_csv_path, index=True)
    print(data)
    
    # Plot each column and save to the specified output directory
    plt.figure(figsize=(12,6))
    for column in columns:
        plt.plot(data.index, data[column], marker='o', label=column)

    plt.title(f'{eval_name}, Model RP: {model_rp}, Batch RP: {generation_rp}')
    plt.xlabel('Generation')
    plt.ylabel('Values')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.legend(loc='lower right')
    plt.ylim(0,1.01)

    output_image_path = os.path.join(output_dir, f'{eval_name}_{model_rp}_{generation_rp}.jpeg')
    plt.savefig(output_image_path)
    plt.show()