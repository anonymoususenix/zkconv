# zkConv

## Overview

zkConv is a state-of-the-art zero-knowledge proof system designed for convolutional neural networks (CNNs). It leverages cutting-edge cryptographic techniques to ensure both efficiency and scalability, making it one of the fastest zero-knowledge proof systems for CNNs available today.

## Project Structure

The project is organized into the following directories:

- **`Cargo.lock` and `Cargo.toml`**: Rust package manager files to track dependencies and project metadata.
- **`benches`**: Contains benchmarking scripts to evaluate the performance of the proof system.
- **`crates`**: Includes modular libraries and components that form the building blocks of zkConv.
- **`dat`**: Stores real computational traces from the VGG16 model, providing authentic data for experiments and testing.
- **`src`**: The main source code of zkConv, including core logic and implementations.
- **`tests`**: Unit and integration tests to ensure correctness and reliability of the system.

## Installation and Usage

To use zkConv in your project, follow these steps:

1. Ensure you have the Rust toolchain installed, preferably the nightly version (e.g., `rustc 1.86.0-nightly`).
2. Clone the repository:
   ```sh
   git clone https://github.com/anonymoususenix/zkconv.git
   ```
2. Navigate to the project root

3. Build the project:
   ```sh
   cargo build --release
   ```
4. Run examples or benchmarks as needed. All experimental data can be generated using `cargo bench`. This will run the benchmarking suite and output performance metrics for the proof system:
   ```sh
   cargo bench
   ```
**Note**: If you encounter an error like `failed to parse lock file`, please ensure your Cargo version is at least `1.84.0`. You can check your Cargo version with:
   ```sh
   cargo --version
   ```
   If your version is lower, update it by running:
   ```sh
   rustup update
   ```

## Contributing

We welcome contributions to zkConv! If you have ideas for improvements, please feel free to open an issue or submit a pull request. Follow the standard Rust community guidelines and ensure your code passes the tests and benchmarks.

## License

zkConv is licensed under the [MIT License](LICENSE). Feel free to use, modify, and distribute the code as per the license terms.

---

For more information or questions, please contact the project maintainers.

The author of this repository is identified by the SHA256 hash: `2bd2b50a15fd3ba6f076e9321b98a4edac9c8f257560ce4ea8c8cef2f10d80cf` for credibility purposes.
