import os
import random
import shutil
import pandas as pd
import numpy as np
import math
import pyexcel as pe
import pyexcel_ods3


source_dir = r"C:\Users\msaiz\Documents\1_PhD_Diary\Comparison_PPSP_Algorithms\RND_Schedules"
# choose the id of the folder and store as variable
id_folder = 1
# create the destination folder and name it "New_Files" + id_folder
dest_dir = r"C:\Users\msaiz\Documents\1_PhD_Diary\Comparison_PPSP_Algorithms\RND_Schedules\New_Files" + str(id_folder)

if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

# Specify the number of files to create
num_files = 20

for file_num in range(1, num_files+1):
    # Generate random data for the "MODE", "LOW", "HIGH", and "PREDECESSORS" columns
    # initialize the dataframe with the columns "DESCR" and "CODE" etc.
    # df = pd.DataFrame(columns=["DESCR", "CODE", "PREDECESSORS", "LOW", "HIGH", "MODE"])
    num_tasks = random.randint(8, 25)
    df = pd.DataFrame({"DESCR": ["Task " + str(i+1) for i in range(num_tasks)], "CODE": [chr(ord("A") + i) for i in range(num_tasks)]})
    for i in range(len(df)):
        # Randomize the "PREDECESSORS" column with a logic that later tasks tend to have earlier tasks as potential predecessors
        if i == 0:
            df.at[i, "PREDECESSORS"] = ""
        else:
            potential_predecessors = df.loc[:i-1, "CODE"].astype(str).tolist()
            if len(potential_predecessors) > 0:
                # Randomly select 0, 1, 2 or 3 predecessors
                predecessors = random.sample(potential_predecessors, random.randint(0, min(3, len(potential_predecessors))))
                if len(predecessors) > 1:
                    predecessors.sort()  # sort the list in ascending order
                df.at[i, "PREDECESSORS"] = "".join(predecessors)
            else:
                df.at[i, "PREDECESSORS"] = ""

        # Randomize the "MODE" column
        mode = math.floor(random.uniform(3, 12))

        # Randomize the "LOW" column with a value that is 10-20% lower than "MODE"
        low = math.floor(mode * random.uniform(0.8, 0.9))
        df.at[i, "LOW"] = low if not pd.isna(low) else ""

        # Randomize the "HIGH" column with a value that is 10-40% higher than "MODE"
        high = math.floor(mode * random.uniform(1.1, 1.4))
        df.at[i, "HIGH"] = high if not pd.isna(high) else ""
        # I place this at the end so that it follows the order I want
        df.at[i, "MODE"] = mode

    # convert all the numeric data inside df into integer type
    df = df.astype({"MODE": int, "LOW": int, "HIGH": int})

    # Save the new file with a different name replacing the "_wb" with "_wb" + id_folder
    if file_num > 9:
        new_filename = "data_wb" + str(file_num) + ".csv"
    else:
        new_filename = "data_wb0" + str(file_num) + ".csv"
    # record at a .csv
    df.to_csv(os.path.join(dest_dir, new_filename), index=False)
        
# create a new dataframe and store the number of the file and the sum of the values in the "MODE" column
# proj_overview = pd.DataFrame(columns=["File", "Total_days_Sum"])
# iterate over the files in the destination folder
# for filename in os.listdir(dest_dir):
    # Load the CSV file into a pyexcel sheet
#     sheet = pe.get_sheet(file_name=os.path.join(dest_dir, filename))
    # add the sum of the values in the "MODE" column to the dataframe
#     proj_overview = pd.concat([proj_overview, pd.DataFrame({"File": filename, "Total_days_Sum":[df["MODE"].sum()]})], ignore_index=True)

# create a new dataframe and store the number of the file and the sum of the values in the "MODE" column
proj_overview = pd.DataFrame(columns=["File", "Total_days_Sum"])
for filename in os.listdir(dest_dir):
    if filename.startswith("data_wb") and filename.endswith(".csv"):
        df = pd.read_csv(os.path.join(dest_dir, filename))
        proj_overview = pd.concat([proj_overview, pd.DataFrame({"File": [filename], "Total_days_Sum": [df["MODE"].sum()]})], ignore_index=True)

print (proj_overview)



# Load the expected cash flows file located at sourcedir
cf_filename = "expected_cash_flows.txt"
with open(os.path.join(source_dir, cf_filename), "r") as f:
    cash_flows = f.readlines()


# Normalize the Total_days_Sum column to have values between 0 and 1
proj_overview['Normalized_days'] = proj_overview['Total_days_Sum'] / proj_overview['Total_days_Sum'].max()

# Generate random cash flows for 4 years with a weak correlation to Total_days_Sum
cash_flows = []
for index, row in proj_overview.iterrows():
    base_cash_flow = round(row['Normalized_days'],2) * 800000  # Arbitrary scaling factor
    year_1 = int(base_cash_flow * (1 + round(np.random.uniform(0, 0.12),2)))
    year_2 = int(base_cash_flow * (1 + round(np.random.uniform(0.1, 0.23),2)))
    year_3 = int(base_cash_flow * (1 + round(np.random.uniform(0.2, 0.35),2)))
    year_4 = int(base_cash_flow * (1 + round(np.random.uniform(0.3, 0.48),2)))  
    cash_flows.append([year_1, year_2, year_3, year_4])

# Create a DataFrame with the generated cash flows and save it as a txt file
cash_flows_df = pd.DataFrame(cash_flows, columns=['Year_1', 'Year_2', 'Year_3', 'Year_4'])
print(cash_flows_df)
#store the cash flows in a txt file located in the same folder as expected_cash_flows.txt use spaces as separator
cash_flows_df.to_csv(os.path.join(source_dir, "cash_flows.txt"), sep=" ", index=False)

# convert all .csv files in the destination folder to .ods
for filename in os.listdir(dest_dir):
    if filename.endswith(".csv"):
        # Load the CSV file into a pyexcel sheet
        sheet = pe.get_sheet(file_name=os.path.join(dest_dir, filename))

        # Save the sheet as an ODS file
        output_filename = os.path.splitext(filename)[0] + ".ods"
        pyexcel_ods3.save_data(os.path.join(dest_dir, output_filename), {"Sheet 1": sheet.to_array()})


# Generate data for the risk register
data = []
z=0
# read the values from the proj_overview dataframe
total_days_sum_values = proj_overview["Total_days_Sum"].values
# for each value in the total_days_sum_values array, generate values, store the result in the riskreg_df dataframe
for d in total_days_sum_values:
    # initialize empty dataframe riskreg_df
    data = []
    z += 1
    for i in range(10):
        ml_impact = np.random.randint(0.3*d, 0.7*d)
        opt_impact = np.random.randint(0.2*ml_impact, 0.8*ml_impact)
        pess_impact = np.random.randint(1.2*ml_impact, 1.9*ml_impact)
        #base_bdgt = 0
        if i == 0:
            base_bdgt = np.random.randint(350*d, 450*d) 
        base_bdgt = 0
        if i == 0:
            base_bdgt = np.random.randint(7.50*d, 10.50*d) 
        row = [round(random.uniform(0.05, 0.45), 2), opt_impact, pess_impact, ml_impact, base_bdgt]
        # round all values to integers except for the first column
        row = [round(x) if i > 0 else x for i, x in enumerate(row)]
        data.append(row)
        riskreg_df = pd.DataFrame(data, columns=["Probability", "Opt_impact", "Pess_impact", "ML_impact", "Base_Bdgt"])
    # write the dafaframe in an ods file with name riskreg_1.ods, riskreg_2.ods, etc. where 1, 2, etc. 
    # correspond to the number of iteration of the first loop
    if z > 9:
        riskreg_df.to_excel(os.path.join(dest_dir, "riskreg_" + str(z) + ".ods"), index=False)
    else:
        riskreg_df.to_excel(os.path.join(dest_dir, "riskreg_0" + str(z) + ".ods"), index=False)
