from os import times
import numpy as np
import pandas as pd
import pyproj
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod


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
    return df[['t', 'x', 'y', 'vx', 'vy', 'speed', 'heading']]

def kinematic_position(t, x1, v1, b, c):
    """Computes interpolated position using kinematic equations."""
    return x1 + v1 * t + (t**2) * b / 2 + (t**3) * c / 6

def kinematic_speed(t, v1, b, c):
    """Computes interpolated speed using kinematic equations."""
    return v1 + t * b + (t**2) * c / 2

def kinematic_acceleration(t, b, c):
    """Computes acceleration at a given time using kinematic equations."""
    return b + t * c

# Defining an abstract class for interpolation strategies
class InterpolationStrategy(ABC):
    @abstractmethod
    def interpolate(self, t_s, x1, v1, x2, v2, t):
        pass

# class LinearAccelerationInterpolation(InterpolationStrategy):
#     def interpolate(self, t_s, x1, v1, x2, v2, t):
#         # Solve for coefficients in x-direction
#         ax = np.array([[t**2 / 2, t**3 / 6], [t, t**2 / 2]])
#         bx = [x2[0] - x1[0] - v1[0] * t, v2[0] - v1[0]]
#         coef_x = np.linalg.solve(ax, bx)

#         # Solve for coefficients in y-direction
#         by = [x2[1] - x1[1] - v1[1] * t, v2[1] - v1[1]]
#         coef_y = np.linalg.solve(ax, by)

#         # Compute interpolated positions, velocities, and headings
#         x = kinematic_position(t_s, x1[0], v1[0], coef_x[0], coef_x[1])
#         y = kinematic_position(t_s, x1[1], v1[1], coef_y[0], coef_y[1])

#         vx = kinematic_speed(t_s, v1[0], coef_x[0], coef_x[1])
#         vy = kinematic_speed(t_s, v1[1], coef_y[0], coef_y[1])
#         v = np.sqrt(vx**2 + vy**2)

#         ax = kinematic_acceleration(t_s, coef_x[0], coef_x[1])
#         ay = kinematic_acceleration(t_s, coef_y[0], coef_y[1])
#         a = np.sqrt(ax**2 + ay**2)

#         heading = np.arctan2(vx, vy + EPS)
#         heading = np.where(heading < 0, heading + 2 * np.pi, heading)

#         return pd.DataFrame({
#             'x': x, 'y': y, 't': t_s, 'speed': v,
#             'acceleration': a, 'heading': np.rad2deg(heading)
#         })

class LinearAccelerationInterpolation(InterpolationStrategy):
    # def interpolate(self, x1, v1, x2, v2, t1, t2, step):
    #     t = t2 - t1
    #     times = np.linspace(t1, t2, step + 2)  # Exclude start and end times
    #     time_from_start = times - t1
    #     print(f"Interpolating from t={t1} to t={t2} with step={step}")
    #     # Solve for coefficients in x-direction
    #     ax = np.array([[t**2 / 2, t**3 / 6], [t, t**2 / 2]])
    #     bx = [x2[0] - x1[0] - v1[0] * t, v2[0] - v1[0]]
    #     coef_x = np.linalg.solve(ax, bx)

    #     # Solve for coefficients in y-direction
    #     by = [x2[1] - x1[1] - v1[1] * t, v2[1] - v1[1]]
    #     coef_y = np.linalg.solve(ax, by)

    #     # Compute interpolated positions, velocities, and headings
    #     x = kinematic_position(time_from_start, x1[0], v1[0], coef_x[0], coef_x[1])
    #     y = kinematic_position(time_from_start, x1[1], v1[1], coef_y[0], coef_y[1])

    #     vx = kinematic_speed(time_from_start, v1[0], coef_x[0], coef_x[1])
    #     vy = kinematic_speed(time_from_start, v1[1], coef_y[0], coef_y[1])
    #     v = np.sqrt(vx**2 + vy**2)

    #     ax = kinematic_acceleration(time_from_start, coef_x[0], coef_x[1])
    #     ay = kinematic_acceleration(time_from_start, coef_y[0], coef_y[1])
    #     a = np.sqrt(ax**2 + ay**2)

    #     heading = np.arctan2(vx, vy + EPS)
    #     heading = np.where(heading < 0, heading + 2 * np.pi, heading)
    #     # print(times)
    #     return pd.DataFrame({
    #         'x': x, 'y': y, 't': times, 'vx': vx, 'vy': vy,
    #         'speed': v, 'ax': ax, 'ay': ay,
    #         'acceleration': a, 'heading': np.rad2deg(heading)
    #     })

    def interpolate(self, x1, v1, a1, a2, t1, t2, step):
    
        """
        Interpolate segment using given endpoint acceleration components.
        a1, a2: 2-element arrays (ax, ay) at t1 and t2 respectively.
        """
        T = float(t2 - t1)
        if T <= 0:
            raise ValueError("t2 must be > t1")

        times = np.linspace(t1, t2, step + 2)
        t_rel = times[1:-1] - t1
        print(f"Interpolating from t={t1} to t={t2} with step={step}")

        # For each axis, acceleration is linear: a(t) = b + m * t_rel
        # endpoint values give b = a1, m = (a2 - a1) / T
        a1 = np.asarray(a1, dtype=float)
        a2 = np.asarray(a2, dtype=float)
        m = (a2 - a1) / T
        b = a1.copy()

        # integrate acceleration -> velocity: v(t) = v1 + b*t + 0.5*m*t^2
        vx = v1[0] + b[0] * t_rel + 0.5 * m[0] * t_rel**2
        vy = v1[1] + b[1] * t_rel + 0.5 * m[1] * t_rel**2

        # integrate velocity -> position: x(t) = x1 + v1*t + 0.5*b*t^2 + (1/6)*m*t^3
        x = x1[0] + v1[0] * t_rel + 0.5 * b[0] * t_rel**2 + (1.0 / 6.0) * m[0] * t_rel**3
        y = x1[1] + v1[1] * t_rel + 0.5 * b[1] * t_rel**2 + (1.0 / 6.0) * m[1] * t_rel**3

        # acceleration components (linear)
        ax_comp = b[0] + m[0] * t_rel
        ay_comp = b[1] + m[1] * t_rel

        heading = np.rad2deg(np.unwrap(np.arctan2(vy, vx + EPS)))
        return pd.DataFrame({
            'x': x, 'y': y, 't': times[1:-1], 'vx': vx, 'vy': vy,
            'ax': ax_comp, 'ay': ay_comp,
            'heading': heading
        })

# TODO: Implement the ConstantAccelerationInterpolation class
class ConstantAccelerationInterpolation(InterpolationStrategy):
    def interpolate(self, t_s, x1, v1, x2, v2, t):
        # Implementation for constant acceleration interpolation
        pass


def timestamp_to_seconds(timestamps):
    """Converts timestamps to seconds relative to the first timestamp."""
    timestamps = pd.to_datetime(timestamps)
    start_time = timestamps[0]
    journey_time = [(ts - start_time).total_seconds() for ts in timestamps]
    return journey_time, start_time

# def interpolate_trajectory(df, strategy: InterpolationStrategy, num_interpolations=100):
#     """Performs interpolation for the full trajectory using the specified strategy."""
#     interpolated_points = pd.DataFrame()
    
#     for i in range(len(df) - 1):
#         segment = df.iloc[i:i+2][['x', 'y', 't', 'vx', 'vy', 'speed', 'heading']].to_numpy()
#         t_start = df.iloc[i]['t']
#         t_end = df.iloc[i+1]['t']
        
#         # Generate multiple intermediate time steps between t_start and t_end
#         times = np.linspace(t_start, t_end, num=num_interpolations+2)
#         t_s = times - t_start
#         print(times)
#         interpolated_segment = strategy.interpolate(t_s, segment[0, :2], segment[0, 3:5], segment[1, :2], segment[1, 3:5], t_end - t_start)
#         interpolated_points = pd.concat([interpolated_points, interpolated_segment], ignore_index=True)

#     return interpolated_points

def interpolate_trajectory(df, strategy: InterpolationStrategy, num_interpolations=100):
    parts = []
    vx = df['vx'].astype(np.float64)
    vy = df['vy'].astype(np.float64)
    t  = df['t'].astype(np.float64)

    # forward diff for all rows except the last
    dt_f = t.shift(-1) - t
    ax_f = (vx.shift(-1) - vx) / dt_f
    ay_f = (vy.shift(-1) - vy) / dt_f

    # last row: use backward difference
    if len(df) >= 2:
        dt_last = t.iloc[-1] - t.iloc[-2]
        if dt_last != 0.0:
            ax_f.iloc[-1] = (vx.iloc[-1] - vx.iloc[-2]) / dt_last
            ay_f.iloc[-1] = (vy.iloc[-1] - vy.iloc[-2]) / dt_last
        else:
            ax_f.iloc[-1] = 0.0
            ay_f.iloc[-1] = 0.0

    # guard against inf/nan and store
    df['ax'] = ax_f.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float64)
    df['ay'] = ay_f.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float64)
    for i in range(len(df) - 1):
        segment = df.iloc[i:i+2][['x','y','vx','vy','ax','ay']].to_numpy()
        t_start = df.iloc[i]['t']
        t_end = df.iloc[i+1]['t']

        start_pt = df.iloc[i]
        end_pt = df.iloc[i+1]
        seg_res = strategy.interpolate(segment[0,:2], segment[0,2:4], segment[0,4:6], segment[1,4:6], t_start, t_end, num_interpolations)
        # If seg_res.t is relative, add t_start here or ensure strategy returns absolute t
        parts.append(start_pt.to_frame().T)  # include start point
        parts.append(seg_res)
    
    parts.append(end_pt.to_frame().T)    # include end point
    return pd.concat(parts, ignore_index=True)

# def interpolate_trajectory(df, strategy: InterpolationStrategy, num_interpolations=100):
#     """Performs interpolation for the full trajectory using the specified strategy."""
#     interpolated_points = pd.DataFrame()
    
#     for i in range(len(df) - 1):
#         segment = df.iloc[i:i+2][['x', 'y', 't', 'vx', 'vy', 'speed', 'heading']].to_numpy()
#         t_start = df.iloc[i]['t']
#         t_end = df.iloc[i+1]['t']
        
#         # Generate multiple intermediate time steps between t_start and t_end
#         times = np.linspace(t_start, t_end, num=num_interpolations+2)
#         # print(times)
#         time_from_start= times - t_start

#         # interpolated_segment = strategy.interpolate(time_from_start, segment[0, :2], segment[0, 3:5], segment[1, :2], segment[1, 3:5], t_end - t_start)
#         interpolated_segment = strategy.interpolate(segment[0, :2], segment[0, 3:5], segment[1, :2], segment[1, 3:5], t_start, t_end, num_interpolations)
#         # print(interpolated_segment)
#         interpolated_points = pd.concat([interpolated_points, interpolated_segment], ignore_index=True)

#     return interpolated_points

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
