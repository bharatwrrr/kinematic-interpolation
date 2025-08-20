import unittest
import pandas as pd
import numpy as np
import os
import sys

MODULE_DIR_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if MODULE_DIR_PATH not in sys.path:
    sys.path.insert(0, MODULE_DIR_PATH)

from kinematic_interpolation.kinematic_interpolation import (latlon_to_xy, xy_to_latlon, prepare_data, 
                         timestamp_to_seconds, interpolate_trajectory, LinearAccelerationInterpolation)

class TestKinematicInterpolation(unittest.TestCase):

    def setUp(self):
        """Set up toy example data for testing."""
        self.sample_time = pd.date_range("2023-01-01 12:00:00", periods=5, freq="5S")
        self.sample_lat = [37.7749, 37.7750, 37.7751, 37.7752, 37.7753]
        self.sample_lon = [-122.4194, -122.4195, -122.4196, -122.4197, -122.4198]
        self.sample_speed = [10, 15, 20, 25, 30]  # in m/s
        self.sample_heading = [0, 45, 90, 135, 180]  # in degrees

    def test_latlon_to_xy_and_back(self):
        """Test conversion from lat/lon to x/y and back."""
        x, y = latlon_to_xy(self.sample_lat[0], self.sample_lon[0])
        lat, lon = xy_to_latlon(x, y)

        self.assertAlmostEqual(self.sample_lat[0], lat, places=5)
        self.assertAlmostEqual(self.sample_lon[0], lon, places=5)

    def test_timestamp_to_seconds(self):
        """Test conversion of timestamps to seconds relative to start."""
        t_seconds, start_time = timestamp_to_seconds(self.sample_time)

        self.assertEqual(t_seconds[0], 0)
        self.assertEqual(t_seconds[1], 5)  # 5 seconds between timestamps

    def test_prepare_data(self):
        """Test preparation of dataframe with velocity components."""
        x_coords, y_coords = zip(*[latlon_to_xy(lat, lon) for lat, lon in zip(self.sample_lat, self.sample_lon)])
        t_seconds, _ = timestamp_to_seconds(self.sample_time)

        df = pd.DataFrame({
            'x': x_coords,
            'y': y_coords,
            't': t_seconds,
            'speed': self.sample_speed,
            'heading': self.sample_heading
        })

        df_prepared = prepare_data(df)

        self.assertIn('vx', df_prepared.columns)
        self.assertIn('vy', df_prepared.columns)
        self.assertAlmostEqual(df_prepared['vx'].iloc[0], 0, places=5)  # Speed in x should be 0 for heading=0

    def test_interpolate_trajectory(self):
        """Test full trajectory interpolation."""
        x_coords, y_coords = zip(*[latlon_to_xy(lat, lon) for lat, lon in zip(self.sample_lat, self.sample_lon)])
        t_seconds, start_time = timestamp_to_seconds(self.sample_time)

        df = pd.DataFrame({
            'x': x_coords,
            'y': y_coords,
            't': t_seconds,
            'speed': self.sample_speed,
            'heading': self.sample_heading
        })

        df_prepared = prepare_data(df)
        interpolated_points = interpolate_trajectory(df_prepared, 
                                                     strategy = LinearAccelerationInterpolation(),
                                                     num_interpolations=10)

        # Check that interpolated points are generated
        self.assertGreater(len(interpolated_points), len(df))

        # Check that interpolated points contain lat/lon
        interpolated_latlon = [xy_to_latlon(row['x'], row['y']) for _, row in interpolated_points.iterrows()]
        interpolated_points['lat'], interpolated_points['lon'] = zip(*interpolated_latlon)

        self.assertIn('lat', interpolated_points.columns)
        self.assertIn('lon', interpolated_points.columns)

if __name__ == '__main__':
    unittest.main()
