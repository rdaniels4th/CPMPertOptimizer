# %%
#### Requirments ####
import pandas as pd
import numpy as np
from collections import defaultdict

#### variable definitions ####
file_path = 'data2.txt'

# %%
#### Data Acquisition and Precrocessing ####

def load_tasks_from_txt(file_path):
    tasks = {}
    with open(file_path, 'r') as file:
        next(file)  # Skip the header line
        for line in file:
            task_id, duration, dependencies = line.strip().split('\t')
            tasks[task_id] = create_task(task_id, duration, dependencies)
    return tasks

def create_task(task_id, duration, dependencies):
    if dependencies == '-':
        dependencies = [-1]
    else:
        dependencies = dependencies.replace('"', '').split(',')
        dependencies = [dep.strip() for dep in dependencies]
    return {
        'name': task_id,
        'duration': int(duration),
        'dependencies': dependencies,
        'ES': 0,
        'EF': 0,
        'LS': 0,
        'LF': 0,
        'Float': 0,
        'isCritical': False
    }

# %%
# Load tasks from text file
tasks = load_tasks_from_txt(file_path)

# %%
# Convert tasks from dict to a DataFrame with a unique numerical identifier index
df = pd.DataFrame.from_dict(tasks, orient='index')

# Add unique identifier column
df['ID'] = range(1, len(df) + 1)
df.set_index('ID', inplace=True)

# %%
# Make a copy of the original DataFrame
matrix_trans_df = df.copy()
matrix_df = df.copy()

# %%
# Create a dictionary that maps task names to their respective IDs
name_to_id = matrix_trans_df.reset_index().set_index('name')['ID'].to_dict()

# Define a function that takes a list of task names and returns a list of corresponding IDs
def names_to_ids(names):
    return [-1 if name == -1 else name_to_id[name] for name in names]

# %%
# Apply the function to the 'dependencies' column in matrix_trans_df
# This creates a new column 'dependencies_ID' where task names in 'dependencies' are replaced by their IDs
matrix_trans_df['dependencies_ID'] = matrix_trans_df['dependencies'].apply(names_to_ids)

# Copy the 'dependencies_ID' column from matrix_trans_df to matrix_df
matrix_df['dependencies_ID'] = matrix_trans_df['dependencies_ID']

# %%
matrix_df

# %%
# Make a copy of the original DataFrame
matrix_trans_df = df.copy()
matrix_df = df.copy()

# Create a dictionary that maps task names to their respective IDs
name_to_id = matrix_trans_df.reset_index().set_index('name')['ID'].to_dict()

# Define a function that takes a list of task names and returns a list of corresponding IDs
def names_to_ids(names):
    return [-1 if name == -1 else name_to_id[name] for name in names]

# Apply the function to the 'dependencies' column in matrix_trans_df
# This creates a new column 'dependencies_ID' where task names in 'dependencies' are replaced by their IDs
matrix_trans_df['dependencies_ID'] = matrix_trans_df['dependencies'].apply(names_to_ids)

# Copy the 'dependencies_ID' column from matrix_trans_df to matrix_df
matrix_df['dependencies_ID'] = matrix_trans_df['dependencies_ID']

matrix_df

# %% [markdown]
# Solve CPM/PERT

# %%
# Function to create a graph from the DataFrame
def create_graph(df):
    graph = defaultdict(list)
    for idx, row in df.iterrows():
        for dep in row['dependencies_ID']:
            if dep != -1:
                graph[dep].append(idx)
    return graph

# Function to perform forward pass to calculate ES and EF
def forward_pass(df):
    for node in df.index:
        if -1 in df.loc[node, 'dependencies_ID']:
            df.loc[node, 'ES'] = 0
            df.loc[node, 'EF'] = df.loc[node, 'duration']
        else:
            df.loc[node, 'ES'] = max(df.loc[dep, 'EF'] for dep in df.loc[node, 'dependencies_ID'])
            df.loc[node, 'EF'] = df.loc[node, 'ES'] + df.loc[node, 'duration']

# Function to perform backward pass to calculate LS and LF
def backward_pass(df, graph):
    for node in reversed(df.index):
        if graph[node]:
            df.loc[node, 'LF'] = min(df.loc[dep, 'LS'] for dep in graph[node])
        else:
            df.loc[node, 'LF'] = df.loc[node, 'EF']
        df.loc[node, 'LS'] = df.loc[node, 'LF'] - df.loc[node, 'duration']

# Function to calculate Float and isCritical
def calculate_float_and_critical(df):
    df['Float'] = df['LF'] - df['EF']
    df['isCritical'] = df['Float'] == 0
    return df

# Function to perform DFS on the graph
def dfs(graph, node, path, visited, all_paths):
    visited[node] = True
    path.append(node)

    # If the current node has no dependencies, we've found a path
    if not graph[node]:
        all_paths.append(list(path))
    else:
        # Recur for all dependencies of this node
        for i in graph[node]:
            if visited[i] == False:
                dfs(graph, i, path, visited, all_paths)

    # Remove current node from path[] and mark it as unvisited
    path.pop()
    visited[node] = False

# Function to find all paths
def find_all_paths(df, graph):
    # Initialize visited dictionary and paths list
    visited = {i: False for i in df.index}
    all_paths = []

    # Call the dfs function for each node that has a dependency of -1
    for node in df.index:
        if -1 in df.loc[node, 'dependencies_ID']:
            dfs(graph, node, [], visited, all_paths)

    return all_paths

# Function to calculate total duration of each path
def calculate_total_duration(df, all_paths):
    paths_and_durations = []
    for path in all_paths:
        total_duration = sum(df.loc[node, 'duration'] for node in path)
        paths_and_durations.append((path, total_duration))
    return paths_and_durations

# %%
#  make a copy of the matrix_df
final_imputed_df = matrix_df.copy()

# Create a graph
graph = create_graph(final_imputed_df)

# Perform forward pass
forward_pass(final_imputed_df)

# Perform backward pass
backward_pass(final_imputed_df, graph)

# Calculate Float and isCritical
final_imputed_df = calculate_float_and_critical(final_imputed_df)

# Find all paths
all_paths = find_all_paths(final_imputed_df, graph)

# Calculate total duration of each path
paths_and_durations = calculate_total_duration(final_imputed_df, all_paths)
print(paths_and_durations)

# %%
# Create a new DataFrame from the list of tuples
paths_df = pd.DataFrame(paths_and_durations, columns=['Path', 'Total Duration'])
# Print the new DataFrame
print(paths_df)

# %%
final_full_table = final_imputed_df[['name', 'duration', 'dependencies', 'ES', 'EF', 'LS', 'LF', 'Float', 'isCritical']]
print(final_full_table.head(10))

# %%
# Clear the file before writing
with open('output.txt', 'w') as file:
    file.write('')

# Write to output txt file
final_full_table.to_csv('output.txt', sep='\t', index=False)

