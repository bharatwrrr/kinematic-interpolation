import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyproj import Transformer
from abc import ABC, abstractmethod


# Constants
EPS = 1e-4
EARTH_RADIUS = 6371000  # in meters

# Define the UTM transformer for converting lat/lon to x/y coordinates
_TO_UTM = Transformer.from_crs("EPSG:4326", "EPSG:32615", always_xy=True)
_FROM_UTM = Transformer.from_crs("EPSG:32615", "EPSG:4326", always_xy=True)

def latlon_to_xy(lat, lon):
    """Converts latitude and longitude to Cartesian x, y coordinates."""
    x, y = _TO_UTM.transform(lon, lat)
    return x, y

def xy_to_latlon(x, y):
    """Converts Cartesian x, y coordinates back to latitude and longitude."""
    lon, lat = _FROM_UTM.transform(x, y)
    return lat, lon

def calculate_velocity_components(speed, heading):
    """Calculates velocity components (vx, vy) from speed and heading."""
    radians = np.deg2rad(heading)
    vx = speed * np.sin(radians)
    vy = speed * np.cos(radians)
    return vx, vy

def prepare_data(df):
    """Prepares the dataframe by calculating velocity components and organizing columns."""

    if 'speed' not in df.columns or 'heading' not in df.columns:
        raise ValueError("DataFrame must contain 'speed' and 'heading' columns.")
    df_ = df.copy()
    df_['vx'], df_['vy'] = calculate_velocity_components(df_['speed'], df_['heading'])
    return df_[['t', 'x', 'y', 'vx', 'vy', 'speed', 'heading']]

def kinematic_position(t, x1, v1, b, c):
    """Computes interpolated position using kinematic equations."""
    return x1 + v1 * t + (t**2) * b / 2 + (t**3) * c / 6

def kinematic_speed(t, v1, b, c):
    """Computes interpolated speed using kinematic equations."""
    return v1 + t * b + (t**2) * c / 2

def kinematic_acceleration(t, b, c):
    """Computes acceleration at a given time using kinematic equations."""
    return b + t * c

def solve_bc_from_deltas(dx, dv, t):
    """Solve 2x2 system for b,c with closed form. dx and dv are scalars or arrays."""
    # Matrix:
    # [t^2/2   t^3/6] [b] = [dx - v1*t]
    # [t       t^2/2] [c]   [dv]
    A = (t**2) / 2.0
    B = (t**3) / 6.0
    C = t
    D = (t**2) / 2.0
    det = (t**4) / 12.0  # derived analytically
    # b = (D*bx - B*by) / det
    # c = (-C*bx + A*by) / det
    bx = dx
    by = dv
    b = (D * bx - B * by) / det
    c = (-C * bx + A * by) / det
    return b, c

# Defining an abstract class for interpolation strategies
class InterpolationStrategy(ABC):
    @abstractmethod
    def interpolate(self, t_s, x1, v1, x2, v2, t):
        pass

class LinearAccelerationInterpolation(InterpolationStrategy):
    def interpolate(self, t_s, x1, v1, x2, v2, t):
        # Solve for coefficients in x-direction
        # ax = np.array([[t**2 / 2, t**3 / 6], [t, t**2 / 2]])
        # bx = [x2[0] - x1[0] - v1[0] * t, v2[0] - v1[0]]
        # coef_x = np.linalg.solve(ax, bx)

        # # Solve for coefficients in y-direction
        # by = [x2[1] - x1[1] - v1[1] * t, v2[1] - v1[1]]
        # coef_y = np.linalg.solve(ax, by)

        dx_x = x2[0] - x1[0] - v1[0] * t
        dv_x = v2[0] - v1[0]
        dx_y = x2[1] - x1[1] - v1[1] * t
        dv_y = v2[1] - v1[1]

        bx_x, cx_x = solve_bc_from_deltas(dx_x, dv_x, t)
        bx_y, cx_y = solve_bc_from_deltas(dx_y, dv_y, t)

        # Compute interpolated positions, velocities, and headings
        x = kinematic_position(t_s, x1[0], v1[0], bx_x, cx_x)
        y = kinematic_position(t_s, x1[1], v1[1], bx_y, cx_y)

        vx = kinematic_speed(t_s, v1[0], bx_x, cx_x)
        vy = kinematic_speed(t_s, v1[1], bx_y, cx_y)
        # v = np.sqrt(vx**2 + vy**2)
        v = np.hypot(vx, vy)

        ax = kinematic_acceleration(t_s, bx_x, cx_x)
        ay = kinematic_acceleration(t_s, bx_y, cx_y)
        # a = np.sqrt(ax**2 + ay**2)
        a = np.hypot(ax, ay)

        heading = np.arctan2(vx, vy + EPS)
        heading = np.where(heading < 0, heading + 2 * np.pi, heading)
        heading_deg = np.rad2deg(heading)

        out = np.column_stack((x, y, t_s, v, a, heading_deg))
        return pd.DataFrame(out, columns=['x','y','t','speed','acceleration','heading'])


# TODO: Implement the ConstantAccelerationInterpolation class
class ConstantAccelerationInterpolation(InterpolationStrategy):
    def interpolate(self, t_s, x1, v1, x2, v2, t):
        # Implementation for constant acceleration interpolation
        pass


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


def interpolate_trajectory(df, strategy: InterpolationStrategy, num_interpolations=100):
    """Performs interpolation for the full trajectory using the specified strategy."""
    parts = []
    for i in range(len(df) - 1):
        segment = df.iloc[i:i+2][['x', 'y', 't', 'vx', 'vy', 'speed', 'heading']].to_numpy()
        t_start = df.iloc[i]['t']
        t_end = df.iloc[i+1]['t']
        
        # Generate multiple intermediate time steps between t_start and t_end
        times = np.linspace(t_start, t_end, num=num_interpolations + 2) # because we want to include both endpoints
        print(times)
        t_s = times - t_start

        interpolated_segment = strategy.interpolate(t_s, segment[0, :2], segment[0, 3:5], segment[1, :2], segment[1, 3:5], t_end - t_start)
        parts.append(interpolated_segment)
    return pd.concat(parts, ignore_index=True)

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
