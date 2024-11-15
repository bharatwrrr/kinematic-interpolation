# Kinematic Interpolation

This repository provides a code for performing kinematic interpolation of vehicle trajectories (or anything that moves). 
The interpolation considers real-world attributes like latitude, longitude, velocity, and heading angle, allowing for smooth trajectory estimation between sampled points.

**NOTE: This is not a published Python package yet. I am still working on that part.**

## Features
1. **Lat/Lon to Cartesian Conversion**: Easily convert geographic coordinates to Cartesian coordinates for accurate calculations.
2. **Velocity and Heading Integration**: Use velocity and heading angle to estimate intermediate positions and speeds.
3. **Customizable Interpolation**: Perform kinematic interpolation with options for speed, acceleration, and heading tracking.
4. **Visualization Tools**: Built-in visualizations for trajectories, speed, and heading over time.

## Installation

### Clone the Repository
```bash
$ git clone https://github.com/bharatwrrr/kinematic-interpolation.git
$ cd kinematic-interpolation
```

### Install the Package
```bash
$ pip install -e .
```

### Install Dependencies
```bash
$ pip install -r requirements.txt
```

## Usage

### Import the Package

After installation, you can import the package and use the functions:

```python
from kinematic_interpolation.kinematic_interpolation import (
    latlon_to_xy, xy_to_latlon, prepare_data, timestamp_to_seconds, interpolate_full_trajectory
)
```

### Example Workflow

1. Load a dataset with latitude, longitude, speed, and heading.
2. Convert the data to Cartesian coordinates.
3. Perform kinematic interpolation.

See the [demo_notebook.ipynb](notebooks/demo_notebook.ipynb) for a complete example.

## Running Tests

Run the test suite to ensure everything works as expected:

```bash
$ pytest tests/
```

## Contributing

### Steps to Contribute

1. Fork the repository:
   ```bash
   $ git fork https://github.com/bharatwrrr/kinematic-interpolation.git
   ```
2. Create a feature branch:
   ```bash
   $ git checkout -b feature-name
   ```
3. Commit your changes:
   ```bash
   $ git commit -m "Add a new feature"
   ```
4. Push to the branch:
   ```bash
   $ git push origin feature-name
   ```
5. Open a pull request.

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

This repository was created to demonstrate the use of kinematic interpolation for vehicle trajectories. Data examples and methodologies are inspired by real-world applications in transportation science.

---

For questions or suggestions, feel free to contact the maintainer or open an issue on the GitHub repository.

