use ark_ff::UniformRand;
use ark_std::{
    fs::File,
    io::{self, BufRead, BufReader},
    test_rng,
};
use rand_chacha::rand_core::le;
use std::path::Path;
use std::{fs, vec};
use zkconv::{commit, F};

use zkconv::commit::commit::{
    commit_all, generate_pp, open_all, open_w, prepare_srs, preprocess_w, verify_all, verify_w,
};

// Helper functions to parse data
fn parse_dimensions(dim_str: &str) -> Vec<usize> {
    dim_str
        .trim_matches(&['(', ')'][..])
        .split(',')
        .map(|v| v.trim().parse::<usize>().expect("Invalid dimension"))
        .collect()
}

fn parse_line(line: &str, prefix: &str) -> (Vec<usize>, Vec<F>) {
    if !line.starts_with(prefix) {
        panic!("Expected line to start with '{}'", prefix);
    }
    let parts: Vec<&str> = line.splitn(2, '[').collect();
    if parts.len() != 2 {
        panic!(
            "Invalid format: expected dimensions and data for '{}'",
            prefix
        );
    }

    let dim_part = parts[0]
        .split(':')
        .nth(1)
        .expect("Missing dimensions")
        .trim();
    let dims = parse_dimensions(dim_part);

    let data_part = parts[1].trim_end_matches(']').trim();
    let values = data_part
        .split(',')
        .map(|v| {
            v.trim()
                .parse::<i64>()
                .map(F::from)
                .expect("Invalid data value")
        })
        .collect();

    (dims, values)
}

fn parse_line_with_shapes(line: &str, prefix: &str) -> (Vec<usize>, Vec<F>) {
    if !line.starts_with(prefix) {
        panic!("Expected line to start with '{}'", prefix);
    }

    let parts: Vec<&str> = line.split("padded shape:").collect();
    if parts.len() != 2 {
        panic!("Invalid format: expected padded shape in '{}'", line);
    }

    let useful_shape = parts[0]
        .split("useful shape:")
        .nth(1)
        .expect("Missing useful shape")
        .trim();
    let padded_shape = parts[1]
        .split('[')
        .nth(0)
        .expect("Missing padded shape")
        .trim();

    let useful_dims = parse_dimensions(useful_shape);
    let padded_dims = parse_dimensions(padded_shape);

    if useful_dims.len() != padded_dims.len() {
        panic!("Mismatch between useful and padded dimensions");
    }

    let data_part = parts[1]
        .split('[')
        .nth(1)
        .expect("Missing data section")
        .trim_end_matches(']')
        .trim();
    let values = data_part
        .split(',')
        .map(|v| {
            v.trim()
                .parse::<i64>()
                .map(F::from)
                .expect("Invalid data value")
        })
        .collect();

    (padded_dims, values)
}

fn read_and_prepare_data_conv<P: AsRef<std::path::Path>>(
    file_path: P,
) -> io::Result<(
    Vec<F>,
    Vec<F>,
    Vec<F>,
    Vec<F>,
    Vec<F>,
    Vec<F>,
    usize,
    usize,
    usize,
    usize,
)> {
    let file = File::open(file_path)?;
    let reader = BufReader::new(file);

    let mut lines = reader.lines();

    let plain_x_line = lines
        .next()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Missing plain x line"))??;
    let (plain_x_dims, plain_x_values) = parse_line(&plain_x_line, "plain x:");

    let rot_pad_x_line = lines
        .next()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Missing rot pad x line"))??;
    let (rot_pad_x_dims, rot_pad_x_values) = parse_line(&rot_pad_x_line, "rot pad x:");

    let weight_w_line = lines
        .next()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Missing weight W line"))??;
    let (weight_w_dims, weight_w_values) = parse_line(&weight_w_line, "weight W:");

    let conv_y_line = lines.next().ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidData, "Missing conv direct Y line")
    })??;
    let (conv_y_dims, conv_y_values) = parse_line(&conv_y_line, "conv direct Y:");

    let rot_y_line = lines
        .next()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Missing rot y line"))??;
    let (rot_y_dims, rot_y_values) = parse_line_with_shapes(&rot_y_line, "rot y:");

    let p_line = lines
        .next()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Missing P line"))??;
    let (p_dims, p_values) = parse_line_with_shapes(&p_line, "P:");

    let padding = 1;
    let kernel_size = (weight_w_dims[2] as f64).sqrt() as usize;
    let input_channels = plain_x_dims[0];
    let output_channels = conv_y_dims[0];

    println!(
        "padding: {}, kernel_size: {}, input_channels: {}, output_channels: {}",
        padding, kernel_size, input_channels, output_channels
    );
    //print all dim
    println!("plain_x_dims: {:?}", plain_x_dims);
    println!("rot_pad_x_dims: {:?}", rot_pad_x_dims);
    println!("weight_w_dims: {:?}", weight_w_dims);
    println!("conv_y_dims: {:?}", conv_y_dims);
    println!("rot_y_dims: {:?}", rot_y_dims);
    println!("p_dims: {:?}", p_dims);

    println!("plain_x_values.len(): {}", plain_x_values.len());
    println!("rot_pad_x_values.len(): {}", rot_pad_x_values.len());
    println!("rot_y_values.len(): {}", rot_y_values.len());
    println!("conv_y_values.len(): {}", conv_y_values.len());
    println!("p_values.len(): {}", p_values.len());

    Ok((
        plain_x_values,
        rot_pad_x_values,
        weight_w_values,
        conv_y_values,
        rot_y_values,
        p_values,
        padding,
        kernel_size,
        input_channels,
        output_channels,
    ))
}

fn read_and_prepare_data_maxpool<P: AsRef<Path>>(
    file_path: P,
) -> io::Result<(Vec<F>, Vec<F>, usize, usize, usize, usize)> {
    let file = File::open(file_path)?; // Open the specified file
    let reader = BufReader::new(file); // Create a buffered reader for efficient line-by-line reading

    let mut lines = reader.lines();

    // Parse dimensions of maxpool input
    let maxpool_in_header = lines
        .next()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Missing input header"))??;
    let maxpool_in_dim: Vec<usize> = maxpool_in_header
        .split_whitespace() // Remove extra spaces and split into tokens
        .skip(3) // Skip the first two tokens (e.g., "max pool in:")
        .map(|v| {
            v.parse::<usize>().map_err(|e| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("Invalid dimension value: {:?}", e),
                )
            })
        })
        .collect::<Result<_, _>>()?;
    let maxpool_in_channel = maxpool_in_dim[0]; // Number of input channels
    let maxpool_in_data = maxpool_in_dim[1] * maxpool_in_dim[2]; // Total input data size (height * width)

    // Read maxpool input data
    let maxpool_in_values: Vec<F> = {
        // Read the first line containing all the input data
        let line = lines.next().unwrap().unwrap(); // Get the single line and unwrap Result
        line.split_whitespace() // Split the line into individual tokens
            .map(|v| {
                F::from(v.parse::<u32>().expect("Invalid data value")) // Parse each token
            })
            .collect::<Vec<F>>() // Collect the parsed tokens into a vector
    };

    // Parse dimensions of maxpool output
    let maxpool_out_header = lines
        .next()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Missing output header"))??;
    let maxpool_out_dim: Vec<usize> = maxpool_out_header
        .split_whitespace() // Remove extra spaces and split into tokens
        .skip(3) // Skip the first three tokens (e.g., "max pooling output:")
        .map(|v| {
            v.parse::<usize>().map_err(|e| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("Invalid dimension value: {:?}", e),
                )
            })
        })
        .collect::<Result<_, _>>()?;
    let maxpool_out_channel = maxpool_out_dim[0]; // Number of output channels
    let maxpool_out_data = maxpool_out_dim[1] * maxpool_out_dim[2]; // Total output data size (height * width)

    // Read maxpool output data
    let maxpool_out_values: Vec<F> = {
        // Read the second line containing all the output data
        let line = lines.next().unwrap().unwrap(); // Get the single line and unwrap Result
        line.split_whitespace() // Split the line into individual tokens
            .map(|v| {
                F::from(v.parse::<u32>().expect("Invalid data value")) // Parse each token
            })
            .collect::<Vec<F>>() // Collect the parsed tokens into a vector
    };

    // Return parsed data and dimensions
    Ok((
        maxpool_in_values,
        maxpool_out_values,
        maxpool_in_channel,
        maxpool_in_data,
        maxpool_out_channel,
        maxpool_out_data,
    ))
}

fn read_and_prepare_test_files(dir_path: &str) -> Vec<std::path::PathBuf> {
    fs::read_dir(dir_path)
        .expect("Unable to read directory")
        .filter_map(Result::ok)
        .filter(|entry| {
            entry
                .file_name()
                .to_string_lossy()
                .starts_with("conv_layer_")
        })
        .map(|entry| entry.path())
        .collect()
}

#[test]
fn test_commit_and_verify_conv_files() {
    let dir_path = "./dat/dat";
    let conv_files = read_and_prepare_test_files(dir_path);

    let mut max_vector_for_all_files = Vec::new();
    for file_path in &conv_files {
        let file_name = file_path.file_name().unwrap().to_string_lossy().to_string();
        println!("Testing file: {}", file_name);

        let (
            plain_x_values,
            rot_pad_x_values,
            weight_w_values,
            conv_y_values,
            rot_y_values,
            p_values,
            _,
            _,
            _,
            _,
        ) = read_and_prepare_data_conv(file_path)
            .expect(&format!("Failed to read file: {}", file_name));

        // choose the maximum vector of (plain_x_values, rot_pad_x_values, weight_w_values, conv_y_values, rot_y_values, p_values)
        let vectors = vec![
            ("plain_x_values", plain_x_values),
            ("rot_pad_x_values", rot_pad_x_values),
            ("weight_w_values", weight_w_values),
            ("conv_y_values", conv_y_values),
            ("rot_y_values", rot_y_values),
            ("p_values", p_values),
        ];

        // Find the vector with the maximum length
        let (name, max_vector) = vectors
            .into_iter()
            .max_by_key(|(_, vector)| vector.len())
            .expect("Vectors cannot be empty");

        max_vector_for_all_files.push(max_vector);
    }

    let max_vector = max_vector_for_all_files
        .iter()
        .max_by_key(|vector| vector.len())
        .expect("Vectors cannot be empty");
    println!("max_vector.len(): {}", max_vector.len());

    let mut rng = test_rng();
    // Generate SRS and parameters
    let srs = prepare_srs(&mut rng, &max_vector);
    let (prover_param, verifier_param) = generate_pp(&srs);

    for file_path in &conv_files {
        let file_name = file_path.file_name().unwrap().to_string_lossy().to_string();
        println!("Testing file: {}", file_name);

        let (
            plain_x_values,
            rot_pad_x_values,
            weight_w_values,
            conv_y_values,
            rot_y_values,
            p_values,
            _,
            _,
            _,
            _,
        ) = read_and_prepare_data_conv(file_path)
            .expect(&format!("Failed to read file: {}", file_name));
        // Commit to weights
        let weight_commit = preprocess_w(&prover_param, &weight_w_values);

        // Generate a random evaluation point
        let eval_point: Vec<F> = (0..weight_w_values.len().ilog2())
            .map(|_| F::rand(&mut rng))
            .collect();

        // Open the commitment
        let (proof, value) = open_w(&prover_param, &weight_w_values, &eval_point);

        // Verify the commitment
        assert!(verify_w(
            &verifier_param,
            &weight_commit,
            &eval_point,
            &proof,
            value
        ));

        println!("File {} passed the commit and verify tests.", file_name);

        // commit and open all other vectors
        let other_vectors: Vec<Vec<F>> = vec![
            plain_x_values.clone(),
            rot_pad_x_values.clone(),
            conv_y_values.clone(),
            rot_y_values.clone(),
            p_values.clone(),
        ];

        // Commit to all other vectors
        let other_commits = commit_all(&prover_param, &other_vectors);
        let other_points: Vec<Vec<F>> = other_vectors
            .iter()
            .map(|vector| {
                (0..vector.len().ilog2())
                    .map(|_| F::rand(&mut rng))
                    .collect()
            })
            .collect();

        let other_results = open_all(&prover_param, &other_vectors, &other_points);

        let other_proofs: Vec<_> = other_results
            .iter()
            .map(|(proof, _)| proof.clone())
            .collect();
        let other_values: Vec<_> = other_results.iter().map(|(_, value)| *value).collect();

        assert!(verify_all(
            &verifier_param,
            &other_commits,
            &other_points,
            &other_proofs,
            &other_values
        ));
    }
}

#[test]
fn test_commit_and_verify_maxpool_files() {
    let dir_path = "./dat/dat";
    let maxpool_files = fs::read_dir(dir_path)
        .expect("Unable to read directory")
        .filter_map(Result::ok)
        .filter(|entry| {
            entry
                .file_name()
                .to_string_lossy()
                .starts_with("maxpool_layer_")
        })
        .map(|entry| entry.path())
        .collect::<Vec<_>>();
    let mut max_vector_for_all_files = Vec::new();

    for file_path in &maxpool_files {
        let file_name = file_path.file_name().unwrap().to_string_lossy().to_string();
        println!("Testing file: {}", file_name);

        let (maxpool_in_values, _, _, _, _, _) = read_and_prepare_data_maxpool(file_path)
            .expect(&format!("Failed to read file: {}", file_name));
        max_vector_for_all_files.push(maxpool_in_values.clone());
    }

    let max_vector = max_vector_for_all_files
        .iter()
        .max_by_key(|vector| vector.len())
        .expect("Vectors cannot be empty");

    let mut rng = test_rng();

    // Generate SRS and parameters
    let srs = prepare_srs(&mut rng, &max_vector);
    let (prover_param, verifier_param) = generate_pp(&srs);

    for file_path in &maxpool_files {
        let file_name = file_path.file_name().unwrap().to_string_lossy().to_string();
        println!("Testing file: {}", file_name);

        let (maxpool_in_values, maxpool_out_values, _, _, _, _) =
            read_and_prepare_data_maxpool(file_path)
                .expect(&format!("Failed to read file: {}", file_name));

        // Commit to maxpool input
        let input_commit = preprocess_w(&prover_param, &maxpool_in_values);

        // Generate a random evaluation point
        let eval_point: Vec<F> = (0..maxpool_in_values.len().ilog2())
            .map(|_| F::rand(&mut rng))
            .collect();

        // Open the commitment
        let (proof, value) = open_w(&prover_param, &maxpool_in_values, &eval_point);

        // Verify the commitment
        assert!(verify_w(
            &verifier_param,
            &input_commit,
            &eval_point,
            &proof,
            value
        ));

        println!("File {} passed the commit and verify tests.", file_name);

        // commit and open all other vectors
        let other_vectors: Vec<Vec<F>> = vec![maxpool_out_values.clone()];

        // Commit to all other vectors
        let other_commits = commit_all(&prover_param, &other_vectors);
        let other_points: Vec<Vec<F>> = other_vectors
            .iter()
            .map(|vector| {
                (0..vector.len().ilog2())
                    .map(|_| F::rand(&mut rng))
                    .collect()
            })
            .collect();

        let other_results = open_all(&prover_param, &other_vectors, &other_points);

        let other_proofs: Vec<_> = other_results
            .iter()
            .map(|(proof, _)| proof.clone())
            .collect();
        let other_values: Vec<_> = other_results.iter().map(|(_, value)| *value).collect();

        assert!(verify_all(
            &verifier_param,
            &other_commits,
            &other_points,
            &other_proofs,
            &other_values
        ));

        println!("File {} passed the commit and verify tests.", file_name);
    }
}
