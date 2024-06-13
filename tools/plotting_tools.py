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
        plt.plot(data.index, data[column], marker='', label=column)
        for i in range(len(data) - 1):
            start_point = data.index[i]
            end_point = data.index[i + 1]
            start_value = data.loc[start_point, column]
            end_value = data.loc[end_point, column]
    
           # Add triangles
            if end_value > start_value:
                plt.scatter(end_point, end_value, marker='^', color='green', s=100)
            else:
                plt.scatter(end_point, end_value, marker='v', color='red', s=100)
        
    plt.title(f'{eval_name}, Model RP: {model_rp}, Inf RP: {generation_rp}')
    plt.xlabel('Generation')
    plt.ylabel('Values')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.legend(loc='lower left')
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
        plt.plot(data.index, data[column], marker='', label=column)
        for i in range(len(data) - 1):
            start_point = data.index[i]
            end_point = data.index[i + 1]
            start_value = data.loc[start_point, column]
            end_value = data.loc[end_point, column]
    
           # Add triangles
            if end_value > start_value:
                plt.scatter(end_point, end_value, marker='^', color='green', s=100)
            else:
                plt.scatter(end_point, end_value, marker='v', color='red', s=100)
            
        


    plt.title(f'{eval_name}, Model RP: {model_rp}, Inf RP: {generation_rp}')
    plt.xlabel('Generation')
    plt.ylabel('Values')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.legend(loc='lower left')
    plt.ylim(0,1.01)

    output_image_path = os.path.join(output_dir, f'{eval_name}_{model_rp}_{generation_rp}.jpeg')
    plt.savefig(output_image_path)
    plt.show()


def process_and_plot_csv_per_metric(score_lineages, column, eval_name, output_dir):
    # Initialize a DataFrame to store the data
    output_data = pd.DataFrame()
    #print(data)
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    def return_variable_name(var):
        variable_name = [name for name, value in locals().items() if value is var][0]
        return variable_name

    # Read the last line of each file and append to the DataFrame
    for lineage in score_lineages:
        print(locals())
        print(return_variable_name(score_lineages [score_lineages.index(lineage)]))
        lineage_rps = '_'.join(return_variable_name(lineage).split('_')[2:])
        print(lineage_rps)
        model_rp = "Inf_RP " + str(float(lineage_rps.split("_")[0]) /10)
        inference_rp = "Inf_RP " + str(float(lineage_rps.split("_")[1]) /10)

        for i, file_path in enumerate(lineage):
            value = pd.read_csv(file_path).loc[:, column].iloc[-1]
            
            if i == 0:
                # Set the first file's last line as "RD"
                output_data.loc["RD", inference_rp] = value
            elif i == 1:
                # Set the first file's last line as "Base"
                output_data.loc["Base", inference_rp] = value
            else:
                row_name = f"GEN{i-2}"
                output_data.loc[row_name, inference_rp] = value

    # Save the DataFrame to a CSV file in the output directory
    output_csv_path = os.path.join(output_dir, f'{column}_{eval_name}.csv')
    output_data.to_csv(output_csv_path, index=True)
    print(output_data)

    # Plot each column and save to the specified output directory
    plt.figure(figsize=(12,6))
    for column in output_data.columns:
        plt.plot(output_data.index, output_data[column], marker='', label=column)
        for i in range(len(output_data) - 1):
            start_point = output_data.index[i]
            end_point = output_data.index[i + 1]
            start_value = output_data.loc[start_point, column]
            end_value = output_data.loc[end_point, column]

            # Add triangles
            if end_value > start_value:
                plt.scatter(end_point, end_value, marker='^', color='green', s=100)
            else:
                plt.scatter(end_point, end_value, marker='v', color='red', s=100)
        
    plt.title(f'{eval_name}, Lineage RP: {model_rp}')
    plt.xlabel('Generation')
    plt.ylabel('Values')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.legend(loc='lower left')
    plt.ylim(0,1.01)

    output_image_path = os.path.join(output_dir, f'{eval_name}_{model_rp}_{inference_rp}.jpeg')
    plt.savefig(output_image_path)
    plt.show()