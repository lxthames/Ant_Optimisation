# Import required libraries
import pandas as pd
import numpy as np
from haversine import haversine, Unit
import random
import sys
import streamlit as st
import plotly.graph_objects as go
import plotly.graph_objs as go
from scipy.spatial import distance
from tsp_solver.greedy import solve_tsp  # Example: Using a library for LKH-like heuristic
import time
# Suppress warnings
import warnings
warnings.filterwarnings('ignore')


class AntColonyOptimizer:
    def __init__(self):
        self.best_distances = []
        self.current_distances = []
        self.y = []  # for tracking values to plot
        self.z = []  # for tracking values to plot

    def calculate_distance(self, path, distance_matrix):
        total_distance = 0
        current_node = path[0]

        for next_node in path[1:]:
            total_distance += distance_matrix[current_node][next_node]
            current_node = next_node

        return total_distance

    def initialize_pheromone(self, distance_matrix):
        total_distance = sum(distance_matrix[i][j] for i in range(len(distance_matrix)) for j in range(len(distance_matrix)) if i != j)
        return 1 / total_distance if total_distance > 0 else 0

    def fit(self, distance_matrix, num_ants=10, max_iterations=100, beta=2, zeta=0.1, rho=0.2, q0=0.7, depot=-1, verbose=False):
        pheromone_matrix = [[self.initialize_pheromone(distance_matrix) for _ in range(len(distance_matrix))] for _ in range(len(distance_matrix))]

        # Ensure depot nodes are valid
        if depot == -1:
            depots = random.sample(range(len(distance_matrix)), num_ants)
        else:
            depots = [depot] * num_ants

        best_paths = []
        best_distance = sys.float_info.max

        for iteration in range(max_iterations):
            best_current_distance = sys.float_info.max
            best_current_path = []

            for ant_index in range(num_ants):
                path, distance = self.construct_path(depots[ant_index], distance_matrix, pheromone_matrix, beta, q0)

                if distance < best_current_distance:
                    best_current_distance = distance
                    best_current_path = path

            if best_current_distance < best_distance:
                best_distance = best_current_distance
                best_paths = best_current_path

            self.update_pheromone(pheromone_matrix, best_current_path, best_current_distance, zeta, rho, self.initialize_pheromone(distance_matrix))
            self.best_distances.append(best_distance)
            self.current_distances.append(best_current_distance)

            # Tracking data for plotting
            self.y.append(best_current_distance)
            self.z.append(best_distance)

            if verbose:
                st.write(f"Iteration {iteration + 1}: Best distance = {best_distance}")

        return best_paths, best_distance

    def construct_path(self, start_node, distance_matrix, pheromone_matrix, beta, q0):
        path = [start_node]
        visited = {start_node}

        while len(path) < len(distance_matrix):
            possible_nodes = [node for node in range(len(distance_matrix)) if node not in visited]
            if not possible_nodes:
                break  # No more unvisited nodes
            next_node = self.select_next_node(start_node, possible_nodes, distance_matrix, pheromone_matrix, beta, q0)

            path.append(next_node)
            visited.add(next_node)
            start_node = next_node

        distance = self.calculate_distance(path, distance_matrix)
        return path, distance

    def select_next_node(self, current_node, possible_nodes, distance_matrix, pheromone_matrix, beta, q0):
        if random.random() <= q0:
            # Heuristic approach
            transitions = [
                (node, (1 / pheromone_matrix[current_node][node]) * (distance_matrix[current_node][node] ** beta))
                for node in possible_nodes
            ]
            next_node = min(transitions, key=lambda x: x[1])[0]
        else:
            # Random selection
            next_node = random.choice(possible_nodes)

        return next_node

    def update_pheromone(self, pheromone_matrix, best_path, best_distance, zeta, rho, initial_pheromone):
        for i in range(len(best_path) - 1):
            from_node = best_path[i]
            to_node = best_path[i + 1]
            pheromone_matrix[from_node][to_node] = (1 - zeta) * pheromone_matrix[from_node][to_node] + initial_pheromone / best_distance
    def Plot(self, max_iterations):
        # Convert data to numpy arrays
        y = np.array(self.y)
        z = np.array(self.z)
        __w = np.average(self.z)
        w = np.array([__w] * max_iterations)
        x = np.linspace(0, max_iterations, max_iterations)

        # Create 3D Plot with Plotly
        fig = go.Figure()

        # Plot y, z vs x
        fig.add_trace(go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='lines',
            name='Path',
            line=dict(color='blue', width=4)
        ))

        # Plot average w
        fig.add_trace(go.Scatter3d(
            x=x,
            y=w,
            z=z,
            mode='lines',
            name='Average Path',
            line=dict(color='red', width=2, dash='dash')
        ))

        # Add layout settings
        fig.update_layout(
            title="Optimization Path over Iterations",
            scene=dict(
                xaxis_title="Iterations",
                yaxis_title="Current Distance (y)",
                zaxis_title="Best Distance (z)"
            ),
            width=700,
            height=700,
        )

        # Display plot in Streamlit
        st.plotly_chart(fig)


def lin_kernighan_heuristic(distance_matrix):
    """
    A simplified implementation or library wrapper for the Lin-Kernighan heuristic.
    Using an external library like `tsp_solver` for demonstration purposes.

    Parameters:
        distance_matrix (list): A 2D list representing the distance matrix.

    Returns:
        list: The best path found.
        float: The total distance of the best path.
    """
    # Convert to a symmetric distance matrix if not already symmetric
    symmetric_matrix = (np.array(distance_matrix) + np.array(distance_matrix).T) / 2.0

    # Use an external library to find the approximate solution
    path = solve_tsp(symmetric_matrix)
    total_distance = sum(symmetric_matrix[path[i]][path[i + 1]] for i in range(len(path) - 1))
    total_distance += symmetric_matrix[path[-1]][path[0]]  # Complete the cycle
    return path, total_distance

def load_data(file_path):
    """
    Load and preprocess the data from a CSV file.

    Parameters:
        file_path (UploadedFile): The uploaded file containing latitude and longitude data.

    Returns:
        list: Distance matrix.
        DataFrame: Locations DataFrame containing latitudes and longitudes.
        str: Error message if any issues are encountered; otherwise, None.
    """
    try:
        # Load the CSV file
        df = pd.read_csv(file_path)

        # Ensure lat and lng columns exist in the dataset
        if 'lat' not in df.columns or 'lng' not in df.columns:
            return None, None, "No columns found for latitude ('lat') or longitude ('lng')."

        # Select only the relevant columns and drop rows with NaN values in those columns
        locations = df[['lat', 'lng']].dropna()

        # Validate latitude and longitude values
        invalid_rows = locations[(locations['lat'] < -90) | (locations['lat'] > 90) | (locations['lng'] < -180) | (locations['lng'] > 180)]
        if not invalid_rows.empty:
            return None, None, f"Invalid latitude/longitude values found in rows:\n{invalid_rows}"

        # Initialize distance matrix
        distance_matrix = []

        # Calculate the distance matrix using the Haversine formula
        for i in range(len(locations)):
            from_node = (locations.iloc[i]['lat'], locations.iloc[i]['lng'])  # Latitude first
            distances = []
            for j in range(len(locations)):
                to_node = (locations.iloc[j]['lat'], locations.iloc[j]['lng'])  # Latitude first
                distance = haversine(from_node, to_node, unit=Unit.METERS) / 1000  # Convert to kilometers
                distances.append(distance)
            distance_matrix.append(distances)

        return distance_matrix, locations, None

    except Exception as e:
        return None, None, f"An error occurred while processing the file: {str(e)}"


# Streamlit UI
st.title("TSP Solver with Ant Colony Optimization (ACO) and Lin-Kernighan Heuristic (LKH)")

# File upload
uploaded_file = st.file_uploader("Upload CSV file with latitude and longitude. Please ensure the data has column name lat for latitude and lng for longitude.", type='csv')

if uploaded_file is not None:
    distance_matrix, locations, message = load_data(uploaded_file)

    if message:
        st.error(message)
    else:
        st.write("Number of Locations:", len(locations))
        st.write("Locations DataFrame:", locations)

        # Plot the locations using Plotly
        fig_locations = go.Figure()
        fig_locations.add_trace(go.Scatter(
            x=locations['lng'],
            y=locations['lat'],
            mode='markers+text',
            marker=dict(size=10, color='blue'),
            text=locations.index,
            textposition='top center'
        ))
        fig_locations.update_layout(
            title="Locations",
            xaxis_title="Longitude",
            yaxis_title="Latitude"
        )
        st.plotly_chart(fig_locations)

        # Method selection
        method = st.selectbox("Select Optimization Method", options=["Ant Colony Optimization (ACO)", "Lin-Kernighan Heuristic (LKH)"])
        # Parameters input
        # Parameters input
        if method == "Ant Colony Optimization (ACO)":
             num_ants = st.number_input(
                "Number of Ants:",
                min_value=1,
                max_value=100,
                value=5,
                help="The number of ants in the colony. A higher number increases exploration but also computational time."
            )
             max_iterations = st.number_input(
                "Max Iterations:",
                min_value=1,
                max_value=500,
                value=100,
                help="The maximum number of iterations the optimization will run for."
            )
             beta = st.number_input(
                "Heuristic constant (beta):",
                min_value=0.0,
                value=4.0,
                help="Controls the influence of the heuristic information (distance) in the decision-making process."
            )
             zeta = st.number_input(
                "Local pheromone decay (zeta):",
                min_value=0.0,
                value=0.4,
                help="The rate at which local pheromone evaporates, reducing its influence over time."
            )
             rho = st.number_input(
                "Global pheromone decay (rho):",
                min_value=0.0,
                value=0.2,
                help="The rate at which global pheromone evaporates. A higher value emphasizes recent paths."
            )
             q0 = st.number_input(
                "Randomization factor (q0):",
                min_value=0.0,
                value=0.7,
                help="Probability of choosing the next node based on exploitation (best choice) rather than exploration."
            )

        elif method == "Lin-Kernighan Heuristic (LKH)":
            st.write("No additional parameters are required for LKH.")

        # Run the selected method
        if st.button("Run Optimization"):
            if method == "Ant Colony Optimization (ACO)":
                with st.spinner("Running optimization..."):
                   progress = st.progress(0)
                   for i in range(100):  # Simulate progress
                     progress.progress(i + 1)
                     time.sleep(0.05)
                optimizer = AntColonyOptimizer()
                try:
                    best_paths, best_distance = optimizer.fit(
                        distance_matrix,
                        num_ants=num_ants,
                        max_iterations=max_iterations,
                        beta=beta,
                        zeta=zeta,
                        rho=rho,
                        q0=q0,
                        verbose=False
                    )

                    st.success("ACO Optimization Completed Successfully!")
                    st.write("Best Path (ACO):", best_paths)
                    st.write(f"Best Distance (ACO): {best_distance:.2f} km")

                    # Create an animated plot for ACO Best Path
                    frames = []
                    for i in range(1, len(best_paths)+1):
                        frame = go.Frame(
                            data=[
                                go.Scatter(
                                    x=locations['lng'].iloc[best_paths[:i]],
                                    y=locations['lat'].iloc[best_paths[:i]],
                                    mode='lines+markers',
                                    line=dict(color='red', width=2),
                                    marker=dict(size=10, color='blue'),
                                )
                            ],
                            name=f'frame_{i}'
                        )
                        frames.append(frame)

                    fig_best_path_aco = go.Figure(
                        data=[
                            go.Scatter(
                                x=locations['lng'].iloc[best_paths[:1]],  # Start with the first point
                                y=locations['lat'].iloc[best_paths[:1]],  # Start with the first point
                                mode='lines+markers',
                                line=dict(color='red', width=2),
                                marker=dict(size=10, color='blue'),
                            )
                        ],
                        layout=go.Layout(
                            title="Best Path from ACO (Animated)",
                            xaxis_title="Longitude",
                            yaxis_title="Latitude",
                            updatemenus=[dict(
                                type='buttons',
                                x=0.1,
                                y=-0.1,
                                buttons=[dict(
                                    label="Play",
                                    method="animate",
                                    args=[None, dict(frame=dict(duration=200, redraw=True), fromcurrent=True)],
                                )],
                            font=dict(color='black'),
                            bgcolor="lightblue"
                            )],
                            sliders=[dict(
                                steps=[dict(
                                    args=[['frame_{}'.format(i)], dict(mode='immediate', frame=dict(duration=0, redraw=True))],
                                    label=str(i),
                                    method='animate',
                                ) for i in range(1, len(best_paths)+1)]
                            )]
                        )
                    )
                    fig_best_path_aco.frames = frames
                    st.plotly_chart(fig_best_path_aco)

                    distance_details = []
                    for i in range(len(best_paths) - 1):
                        from_point = best_paths[i]
                        to_point = best_paths[i + 1]
                        distance = distance_matrix[from_point][to_point]
                        distance_details.append((from_point, to_point, distance))

                    # Create a DataFrame to display the distances
                    distance_df = pd.DataFrame(distance_details, columns=["From", "To", "Distance (km)"])
                    st.write("Distances between Points in the Best Path:")
                    st.dataframe(distance_df)
                    # Save results to a CSV
                    output_file = 'ACO-Results.csv'
                    results_df = pd.DataFrame({
                        "Best Path": [best_paths],
                        "Best Distance": [best_distance]
                    })
                    results_df.to_csv(output_file, index=False)
                    optimizer.Plot(max_iterations)

                except Exception as e:
                    st.error(f"An error occurred during ACO: {e}")

            elif method == "Lin-Kernighan Heuristic (LKH)":
                with st.spinner("Running optimization..."):
                   progress = st.progress(0)
                   for i in range(100):  # Simulate progress
                     progress.progress(i + 1)
                     time.sleep(0.05)
                try:
                    best_paths, best_distance = lin_kernighan_heuristic(distance_matrix)

                    st.success("LKH Optimization Completed Successfully!")
                    st.write("Best Path (LKH):", best_paths)
                    st.write(f"Best Distance (LKH): {best_distance:.2f} km")

                    # Create an animated plot for LKH Best Path
                    frames = []
                    for i in range(1, len(best_paths)+1):
                        frame = go.Frame(
                            data=[
                                go.Scatter(
                                    x=locations['lng'].iloc[best_paths[:i]],
                                    y=locations['lat'].iloc[best_paths[:i]],
                                    mode='lines+markers',
                                    line=dict(color='blue', width=2),
                                    marker=dict(size=10, color='red'),
                                )
                            ],
                            name=f'frame_{i}'
                        )
                        frames.append(frame)

                    fig_best_path_lkh = go.Figure(
                        data=[
                            go.Scatter(
                                x=locations['lng'].iloc[best_paths[:1]],  # Start with the first point
                                y=locations['lat'].iloc[best_paths[:1]],  # Start with the first point
                                mode='lines+markers',
                                line=dict(color='blue', width=2),
                                marker=dict(size=10, color='red'),
                            )
                        ],
                        layout=go.Layout(
                            title="Best Path from LKH (Animated)",
                            xaxis_title="Longitude",
                            yaxis_title="Latitude",
                            updatemenus=[dict(
                                type='buttons',
                                x=0.1,
                                y=-0.1,
                                buttons=[dict(
                                    label="Play",
                                    method="animate",
                                    args=[None, dict(frame=dict(duration=200, redraw=True), fromcurrent=True)],
                                )],
                                font=dict(color='black'),
                                bgcolor="lightblue"
                            )],
                            sliders=[dict(
                                steps=[dict(
                                    args=[['frame_{}'.format(i)], dict(mode='immediate', frame=dict(duration=0, redraw=True))],
                                    label=str(i),
                                    method='animate',
                                ) for i in range(1, len(best_paths)+1)]
                            )]
                        )
                    )
                    fig_best_path_lkh.frames = frames
                    st.plotly_chart(fig_best_path_lkh)
                    # Display results
                    #st.write("Best Path:", best_paths)
                    #st.write("Best Distance (Total): {:.2f} km".format(best_distance))

                    # Detailed distances between each point in the best path
                    distance_details = []
                    for i in range(len(best_paths) - 1):
                        from_point = best_paths[i]
                        to_point = best_paths[i + 1]
                        distance = distance_matrix[from_point][to_point]
                        distance_details.append((from_point, to_point, distance))

                    # Create a DataFrame to display the distances
                    distance_df = pd.DataFrame(distance_details, columns=["From", "To", "Distance (km)"])
                    st.write("Distances between Points in the Best Path:")
                    st.dataframe(distance_df)
                    # Save results to a CSV
                    output_file = 'LK-Results.csv'
                    results_df = pd.DataFrame({
                        "Best Path": [best_paths],
                        "Best Distance": [best_distance]
                    })
                    results_df.to_csv(output_file, index=False)
                    #st.write(f"Results saved to {output_file}.")

                except Exception as e:
                    st.error(f"An error occurred during LKH: {e}")
