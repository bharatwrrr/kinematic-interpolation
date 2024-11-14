import numpy as np
import pandas as pd
import pyproj
import matplotlib.pyplot as plt

# Constants
EPS = 1e-4
EARTH_RADIUS = 6371000  # in meters

# Define projection for lat/lon to x/y conversion
PROJECTION = pyproj.Proj(proj='utm', zone=15, ellps='WGS84')

def latlon_to_xy(lat, lon):
    """Converts latitude and longitude to Cartesian x, y coordinates."""
    x, y = PROJECTION(lon, lat)
    return x, y

def xy_to_latlon(x, y):
    """Converts Cartesian x, y coordinates back to latitude and longitude."""
    lon, lat = PROJECTION(x, y, inverse=True)
    return lat, lon

def calculate_velocity_components(speed, heading):
    """Calculates velocity components (vx, vy) from speed and heading."""
    radians = np.deg2rad(heading)
    vx = speed * np.sin(radians)
    vy = speed * np.cos(radians)
    return vx, vy

def prepare_data(df):
    """Prepares the dataframe by calculating velocity components and organizing columns."""
    df['vx'], df['vy'] = calculate_velocity_components(df['speed'], df['heading'])
    return df[['x', 'y', 't', 'vx', 'vy', 'speed', 'heading']]

def kinematic_position(t, x1, v1, b, c):
    """Computes interpolated position using kinematic equations."""
    return x1 + v1 * t + (t**2) * b / 2 + (t**3) * c / 6

def kinematic_speed(t, v1, b, c):
    """Computes interpolated speed using kinematic equations."""
    return v1 + t * b + (t**2) * c / 2

def kinematic_acceleration(t, b, c):
    """Computes acceleration at a given time using kinematic equations."""
    return b + t * c

def kinematic_interpolation(segment, t_s):
    """Performs kinematic interpolation for a single segment."""
    x1, x2 = segment[0, :2], segment[1, :2]
    t1, t2 = segment[0, 2], segment[1, 2]
    v1, v2 = segment[0, 3:5], segment[1, 3:5]

    t = t2 - t1
    
    # Solve for coefficients in x-direction
    ax = np.array([[t**2 / 2, t**3 / 6], [t, t**2 / 2]])
    bx = [x2[0] - x1[0] - v1[0] * t, v2[0] - v1[0]]
    coef_x = np.linalg.solve(ax, bx)

    # Solve for coefficients in y-direction
    by = [x2[1] - x1[1] - v1[1] * t, v2[1] - v1[1]]
    coef_y = np.linalg.solve(ax, by)

    # Compute interpolated positions, velocities, and headings
    x = kinematic_position(t_s, x1[0], v1[0], coef_x[0], coef_x[1])
    y = kinematic_position(t_s, x1[1], v1[1], coef_y[0], coef_y[1])

    vx = kinematic_speed(t_s, v1[0], coef_x[0], coef_x[1])
    vy = kinematic_speed(t_s, v1[1], coef_y[0], coef_y[1])
    v = np.sqrt(vx**2 + vy**2)

    ax = kinematic_acceleration(t_s, coef_x[0], coef_x[1])
    ay = kinematic_acceleration(t_s, coef_y[0], coef_y[1])
    a = np.sqrt(ax**2 + ay**2)

    heading = np.arctan2(vx, vy + EPS)
    heading = np.where(heading < 0, heading + 2 * np.pi, heading)

    return pd.DataFrame({
        'x': x, 'y': y, 't': t_s + t1, 'speed': v,
        'acceleration': a, 'heading': np.rad2deg(heading)
    })

def timestamp_to_seconds(timestamps):
    """Converts timestamps to seconds relative to the first timestamp."""
    timestamps = pd.to_datetime(timestamps)
    start_time = timestamps[0]
    journey_time = [(ts - start_time).total_seconds() for ts in timestamps]
    return journey_time, start_time

def interpolate_full_trajectory(df, start_time):
    """Performs interpolation for the full trajectory."""
    interpolated_points = pd.DataFrame()
    
    for i in range(len(df) - 1):
        segment = df.iloc[i:i+2][['x', 'y', 't', 'vx', 'vy', 'speed', 'heading']].to_numpy()
        t_start = df.iloc[i]['t']
        t_end = df.iloc[i+1]['t']
        
        times = np.arange(t_start, t_end, 1) if t_end - t_start > 1 else np.array([t_start, t_end])
        t_s = times - t_start

        interpolated_segment = kinematic_interpolation(segment, t_s)
        interpolated_points = pd.concat([interpolated_points, interpolated_segment], ignore_index=True)

    return interpolated_points

def visualize_results(original_df, interpolated_df):
    """Visualizes the original and interpolated trajectory, speed, and headings."""
    plt.figure(figsize=(10, 6))
    plt.plot(original_df['x'], original_df['y'], 'bo-', label='Original Points')
    plt.scatter(interpolated_df['x'], interpolated_df['y'], c='red', label='Interpolated Points')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Kinematic Interpolation of Trajectory')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(original_df['t'], original_df['speed'], 'bo-', label='Original Speed')
    plt.plot(interpolated_df['t'], interpolated_df['speed'], 'ro-', label='Interpolated Speed')
    plt.xlabel('Time (s)')
    plt.ylabel('Speed')
    plt.title('Speed Interpolation')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(original_df['t'], original_df['heading'], 'bo-', label='Original Heading')
    plt.plot(interpolated_df['t'], interpolated_df['heading'], 'ro-', label='Interpolated Heading')
    plt.xlabel('Time (s)')
    plt.ylabel('Heading')
    plt.title('Heading Interpolation')
    plt.legend()
    plt.grid(True)
    plt.show()
