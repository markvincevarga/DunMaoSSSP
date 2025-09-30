#![cfg(all(
    feature = "bincode",
    feature = "log",
    feature = "env_logger",
    feature = "serde",
    feature = "serde_json"
))]

use fast_sssp::Graph;
use geo::{Distance, Haversine, Point};
use num_format::{Locale, ToFormattedString};
use osmpbf::{Element, ElementReader};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::env;
use std::io::{self, Write};
use std::path::Path;
use std::time::Instant;

const CACHE_PATH: &str = "./data/cache";

#[derive(Serialize, Deserialize, Debug)]
pub struct GraphMetricsOutput {
    pub graph_info: GraphInfo,
    pub degrees: DegreeMetrics,
    pub diameter: DiameterMetrics,
    pub average_path_length: PathLengthMetrics,
    pub clustering: ClusteringMetrics,
    pub execution: ExecutionInfo,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct GraphInfo {
    pub file_path: String,
    pub vertices: usize,
    pub edges: usize,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct DegreeMetrics {
    pub min: usize,
    pub max: usize,
    pub median: f64,
    pub average: f64,
    pub distribution: HashMap<usize, usize>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct DiameterMetrics {
    pub value: Option<usize>,
    pub is_estimated: bool,
    pub sample_size: Option<usize>,
    pub method: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct PathLengthMetrics {
    pub value: Option<f64>,
    pub skipped: bool,
    pub reason: Option<String>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ClusteringMetrics {
    pub min: f64,
    pub max: f64,
    pub median: f64,
    pub average: f64,
    pub nodes_calculated: usize,
    pub total_nodes: usize,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ExecutionInfo {
    pub mode: String,
    pub total_time_seconds: f64,
}

fn read_osm_pbf_map(pbf_path: &Path, json_output: bool) -> Graph {
    let filename = pbf_path.file_name().expect("Path should have a filename");
    let cache_path = Path::new(CACHE_PATH).join(Path::new(filename).with_extension("cache"));

    // Try to load from cache first
    if cache_path.exists() {
        if !json_output {
            println!("Loading cached graph: {}", cache_path.display());
        }
        match Graph::from_file(&cache_path) {
            Ok(graph) => {
                if !json_output {
                    println!("Successfully loaded from cache");
                }
                return graph;
            }
            Err(e) => {
                if !json_output {
                    println!("Cache file corrupted or incompatible, reprocessing: {}", e);
                }
            }
        }
    } else {
        if !json_output {
            println!("No cache found, processing OSM PBF file...");
        }
    }

    // Process OSM PBF file
    let graph = {
        if !json_output {
            println!("Loading OSM PBF file: {}", pbf_path.display());
        }
        let start = Instant::now();

        let reader = ElementReader::from_path(pbf_path).expect("OSM PBF file should exist");

        struct OsmNode {
            idx: usize,
            lat: f64,
            lon: f64,
        }

        struct OsmEdge {
            from: i64,
            to: i64,
        }

        let mut nodes = HashMap::new();
        let mut edges = Vec::new();
        let mut idx = 0;

        reader
            .for_each(|element| {
                match element {
                    Element::Node(node) => {
                        nodes.insert(
                            node.id(),
                            OsmNode {
                                idx,
                                lat: node.lat(),
                                lon: node.lon(),
                            },
                        );
                        idx += 1;
                    }
                    Element::DenseNode(node) => {
                        nodes.insert(
                            node.id(),
                            OsmNode {
                                idx,
                                lat: node.lat(),
                                lon: node.lon(),
                            },
                        );
                        idx += 1;
                    }
                    Element::Way(way) => {
                        let ways_iter_clone = way.refs().clone();
                        for (a, b) in way.refs().zip(ways_iter_clone.skip(1)) {
                            edges.push(OsmEdge { from: a, to: b });
                        }
                    }
                    _ => {}
                };
            })
            .expect("file must include correct osm data");

        let mut graph = Graph::new(nodes.len());

        if !json_output {
            println!(
                "Building graph with {} nodes and {} edges...",
                nodes.len().to_formatted_string(&Locale::en),
                edges.len().to_formatted_string(&Locale::en)
            );
        }

        for (i, edge) in edges.iter().enumerate() {
            if let (Some(node_a), Some(node_b)) = (nodes.get(&edge.from), nodes.get(&edge.to)) {
                let distance = Haversine.distance(
                    Point::new(node_a.lon, node_a.lat),
                    Point::new(node_b.lon, node_b.lat),
                );
                graph.add_edge(node_a.idx, node_b.idx, distance);
                graph.add_edge(node_b.idx, node_a.idx, distance);
            }

            if i % 100000 == 0 && !json_output {
                println!("Processed {} edges...", i.to_formatted_string(&Locale::en));
            }
        }

        // Save to cache
        std::fs::create_dir_all(CACHE_PATH).expect("Failed to create cache directory");
        match Graph::to_file(&graph, &cache_path) {
            Ok(_) => {
                if !json_output {
                    println!("Graph cached to: {}", cache_path.display());
                }
            }
            Err(e) => {
                if !json_output {
                    println!("Warning: Failed to cache graph: {}", e);
                }
            }
        }

        let elapsed = start.elapsed();
        if !json_output {
            println!(
                "Graph loaded and processed in {:.2}s",
                elapsed.as_secs_f64()
            );
        }

        graph
    };

    if !json_output {
        println!(
            "Graph ready: {} nodes, {} edges",
            graph.vertices.to_formatted_string(&Locale::en),
            graph.edge_count().to_formatted_string(&Locale::en)
        );
    }

    graph
}

fn collect_degree_metrics(degrees: &[usize]) -> DegreeMetrics {
    if degrees.is_empty() {
        return DegreeMetrics {
            min: 0,
            max: 0,
            median: 0.0,
            average: 0.0,
            distribution: HashMap::new(),
        };
    }

    let min_degree = degrees[0];
    let max_degree = degrees[degrees.len() - 1];
    let median_degree = if degrees.len() % 2 == 0 {
        (degrees[degrees.len() / 2 - 1] + degrees[degrees.len() / 2]) as f64 / 2.0
    } else {
        degrees[degrees.len() / 2] as f64
    };
    let avg_degree = degrees.iter().sum::<usize>() as f64 / degrees.len() as f64;

    // Count occurrences of each degree
    let mut degree_counts = HashMap::new();
    for &degree in degrees {
        *degree_counts.entry(degree).or_insert(0) += 1;
    }

    DegreeMetrics {
        min: min_degree,
        max: max_degree,
        median: median_degree,
        average: avg_degree,
        distribution: degree_counts,
    }
}

fn print_degrees_summary(metrics: &DegreeMetrics) {
    if metrics.distribution.is_empty() {
        println!("No nodes found");
        return;
    }

    println!("Node Degrees (sorted):");
    println!("  Min: {}", metrics.min);
    println!("  Max: {}", metrics.max);
    println!("  Median: {:.2}", metrics.median);
    println!("  Average: {:.2}", metrics.average);

    // Sort by degree and print in requested format
    let mut sorted_degrees: Vec<_> = metrics.distribution.iter().collect();
    sorted_degrees.sort_by_key(|(degree, _)| *degree);

    println!("  Degree distribution:");
    println!("  {{");
    for (degree, count) in sorted_degrees {
        println!("    {}: {}", degree, count);
    }
    println!("  }}");
}

fn collect_clustering_metrics(
    clustering: &[f64],
    nodes_calculated: usize,
    total_nodes: usize,
) -> ClusteringMetrics {
    if clustering.is_empty() {
        return ClusteringMetrics {
            min: 0.0,
            max: 0.0,
            median: 0.0,
            average: 0.0,
            nodes_calculated,
            total_nodes,
        };
    }

    let mut sorted_clustering = clustering[..nodes_calculated].to_vec();
    sorted_clustering.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let min_cc = sorted_clustering[0];
    let max_cc = sorted_clustering[sorted_clustering.len() - 1];
    let median_cc = if sorted_clustering.len() % 2 == 0 {
        (sorted_clustering[sorted_clustering.len() / 2 - 1]
            + sorted_clustering[sorted_clustering.len() / 2])
            / 2.0
    } else {
        sorted_clustering[sorted_clustering.len() / 2]
    };
    let avg_cc = sorted_clustering.iter().sum::<f64>() / sorted_clustering.len() as f64;

    ClusteringMetrics {
        min: min_cc,
        max: max_cc,
        median: median_cc,
        average: avg_cc,
        nodes_calculated,
        total_nodes,
    }
}

fn print_clustering_summary(metrics: &ClusteringMetrics) {
    if metrics.nodes_calculated == 0 {
        println!("No clustering coefficients calculated");
        return;
    }

    println!("Clustering Coefficients:");
    println!("  Min: {:.6}", metrics.min);
    println!("  Max: {:.6}", metrics.max);
    println!("  Median: {:.6}", metrics.median);
    println!("  Average: {:.6}", metrics.average);
}

fn calculate_metrics(graph: &Graph, fast_mode: bool, json_output: bool) -> GraphMetricsOutput {
    let total_start = Instant::now();

    if !json_output {
        println!("\n=== Calculating Graph Metrics ===");
        if fast_mode {
            println!("Running in FAST MODE - using sampling and limits for large graphs");
        }
    }

    // 1. Degrees (sorted)
    if !json_output {
        println!("\n1. Calculating node degrees...");
    }
    let start = Instant::now();
    let degrees = graph.degrees_sorted();
    let degree_time = start.elapsed().as_secs_f64();
    if !json_output {
        println!("   Completed in {:.3}s", degree_time);
    }
    let degree_metrics = collect_degree_metrics(&degrees);
    if !json_output {
        print_degrees_summary(&degree_metrics);
    }

    // 2. Diameter (use sampling for large graphs)
    if !json_output {
        println!("\n2. Calculating diameter...");
    }
    let start = Instant::now();
    let diameter_metrics = if fast_mode && graph.vertices > 10000 {
        if !json_output {
            println!("   Using sampling method for large graph...");
        }
        match estimate_diameter_sample(graph, 100, json_output) {
            Some(diameter_estimate) => {
                if !json_output {
                    println!("   Completed in {:.3}s", start.elapsed().as_secs_f64());
                    println!(
                        "   Diameter (estimated): {} (sampled from {} nodes)",
                        diameter_estimate,
                        100.min(graph.vertices)
                    );
                }
                DiameterMetrics {
                    value: Some(diameter_estimate),
                    is_estimated: true,
                    sample_size: Some(100.min(graph.vertices)),
                    method: "sampling".to_string(),
                }
            }
            None => {
                if !json_output {
                    println!("   Completed in {:.3}s", start.elapsed().as_secs_f64());
                    println!("   Diameter: Unable to estimate");
                }
                DiameterMetrics {
                    value: None,
                    is_estimated: true,
                    sample_size: Some(100.min(graph.vertices)),
                    method: "sampling_failed".to_string(),
                }
            }
        }
    } else {
        match graph.diameter() {
            Some(diameter) => {
                if !json_output {
                    println!("   Completed in {:.3}s", start.elapsed().as_secs_f64());
                    println!("   Diameter: {}", diameter);
                }
                DiameterMetrics {
                    value: Some(diameter),
                    is_estimated: false,
                    sample_size: None,
                    method: "exact".to_string(),
                }
            }
            None => {
                if !json_output {
                    println!("   Completed in {:.3}s", start.elapsed().as_secs_f64());
                    println!("   Diameter: Graph is disconnected or has no edges");
                }
                DiameterMetrics {
                    value: None,
                    is_estimated: false,
                    sample_size: None,
                    method: "exact_disconnected".to_string(),
                }
            }
        }
    };

    // 3. Average path length for strongly connected component from node 0
    if !json_output {
        println!("\n3. Calculating average path length for SCC from node 0...");
    }
    let start = Instant::now();
    let path_length_metrics = if fast_mode && graph.vertices > 50000 {
        if !json_output {
            println!(
                "   Skipped for large graph (>50k nodes) - use --full for complete calculation"
            );
        }
        PathLengthMetrics {
            value: None,
            skipped: true,
            reason: Some("Large graph in fast mode".to_string()),
        }
    } else {
        match graph.average_path_length_scc(0) {
            Some(avg_path) => {
                if !json_output {
                    println!("   Completed in {:.3}s", start.elapsed().as_secs_f64());
                    println!("   Average path length: {:.6}", avg_path);
                }
                PathLengthMetrics {
                    value: Some(avg_path),
                    skipped: false,
                    reason: None,
                }
            }
            None => {
                if !json_output {
                    println!("   Completed in {:.3}s", start.elapsed().as_secs_f64());
                    println!(
                        "   Average path length: Unable to calculate (component too small or disconnected)"
                    );
                }
                PathLengthMetrics {
                    value: None,
                    skipped: false,
                    reason: Some("Component too small or disconnected".to_string()),
                }
            }
        }
    };

    // 4. Clustering coefficient
    if !json_output {
        println!("\n4. Calculating clustering coefficients...");
    }
    let start = Instant::now();
    let max_nodes = if fast_mode { 10000 } else { graph.vertices };
    let clustering = calculate_clustering_efficient(graph, max_nodes, json_output);
    if !json_output {
        println!("   Completed in {:.3}s", start.elapsed().as_secs_f64());
    }

    let nodes_calculated = max_nodes.min(graph.vertices);
    let clustering_metrics =
        collect_clustering_metrics(&clustering, nodes_calculated, graph.vertices);
    if !json_output {
        print_clustering_summary(&clustering_metrics);
    }

    GraphMetricsOutput {
        graph_info: GraphInfo {
            file_path: "".to_string(), // Will be filled in by caller
            vertices: graph.vertices,
            edges: graph.edge_count(),
        },
        degrees: degree_metrics,
        diameter: diameter_metrics,
        average_path_length: path_length_metrics,
        clustering: clustering_metrics,
        execution: ExecutionInfo {
            mode: if fast_mode {
                "FAST".to_string()
            } else {
                "FULL".to_string()
            },
            total_time_seconds: total_start.elapsed().as_secs_f64(),
        },
    }
}

fn print_progress(current: usize, total: usize, prefix: &str) {
    let percent = (current * 100) / total;
    print!("\r{}: {}/{} ({}%)", prefix, current, total, percent);
    io::stdout().flush().unwrap();
}

fn estimate_diameter_sample(graph: &Graph, sample_size: usize, json_output: bool) -> Option<usize> {
    let vertices = graph.vertices;
    if vertices == 0 {
        return None;
    }

    let sample_size = sample_size.min(vertices);
    let step = if vertices <= sample_size {
        1
    } else {
        vertices / sample_size
    };

    let mut max_distance = 0;
    let mut samples_processed = 0;

    for vertex in (0..vertices).step_by(step).take(sample_size) {
        let distances = graph.bfs_distances(vertex);
        if let Some(&max_dist) = distances.values().max() {
            max_distance = max_distance.max(max_dist);
        }

        samples_processed += 1;
        if samples_processed % 10 == 0 && !json_output {
            print_progress(samples_processed, sample_size, "Diameter estimation");
        }
    }

    if !json_output {
        println!(); // New line after progress
    }
    Some(max_distance)
}

fn calculate_clustering_efficient(graph: &Graph, max_nodes: usize, json_output: bool) -> Vec<f64> {
    let nodes_to_process = graph.vertices.min(max_nodes);
    let mut coefficients = vec![0.0; graph.vertices];

    for node in 0..nodes_to_process {
        let neighbors: Vec<usize> = graph.edges[node].iter().map(|e| e.to).collect();
        let k = neighbors.len();

        if k < 2 {
            coefficients[node] = 0.0;
            continue;
        }

        let mut triangle_count = 0;

        for i in 0..neighbors.len() {
            for j in (i + 1)..neighbors.len() {
                let neighbor1 = neighbors[i];
                let neighbor2 = neighbors[j];

                // Check if neighbor1 connects to neighbor2
                if neighbor1 < graph.vertices
                    && graph.edges[neighbor1].iter().any(|e| e.to == neighbor2)
                {
                    triangle_count += 1;
                }
            }
        }

        let possible_triangles = k * (k - 1) / 2;
        coefficients[node] = triangle_count as f64 / possible_triangles as f64;

        if node % 1000 == 0 && node > 0 && !json_output {
            print_progress(node, nodes_to_process, "Clustering coefficient");
        }
    }

    if nodes_to_process < graph.vertices && !json_output {
        println!(
            "\nNote: Clustering coefficient calculated for first {} nodes only",
            nodes_to_process
        );
    } else if !json_output {
        println!(); // New line after progress
    }

    coefficients
}

fn print_help(program_name: &str) {
    println!("Graph Metrics Calculator");
    println!("Calculates graph metrics for OpenStreetMap PBF files");
    println!();
    println!("USAGE:");
    println!("    {} [OPTIONS] <osm.pbf file>", program_name);
    println!("    {} --help", program_name);
    println!();
    println!("ARGUMENTS:");
    println!("    <osm.pbf file>    Path to OpenStreetMap PBF file");
    println!();
    println!("OPTIONS:");
    println!("    --fast            Use fast mode with sampling (default for large graphs)");
    println!("    --full            Calculate all metrics exactly (may be slow)");
    println!("    --json            Output results in JSON format");
    println!();
    println!("METRICS CALCULATED:");
    println!("    • Node degrees (sorted ascending)");
    println!("    • Graph diameter");
    println!("    • Average path length for strongly connected component");
    println!("    • Clustering coefficient for each node");
    println!();
    println!("EXAMPLES:");
    println!("    {} data/monaco.osm.pbf", program_name);
    println!("    {} --fast data/monaco.osm.pbf", program_name);
    println!("    {} --json data/monaco.osm.pbf", program_name);
    println!("    {} --full --json data/monaco.osm.pbf", program_name);
    println!();
    println!("NOTE: Large files may take significant time to process.");
    println!("      Processed graphs are cached in ./data/cache/ directory.");
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();

    let args: Vec<String> = env::args().collect();

    if args.len() >= 2 && (args[1] == "--help" || args[1] == "-h") {
        print_help(&args[0]);
        std::process::exit(0);
    }

    if args.len() < 2 {
        eprintln!("Error: Missing required argument");
        eprintln!();
        eprintln!("Usage: {} [OPTIONS] <osm.pbf file>", args[0]);
        eprintln!("Try '{} --help' for more information.", args[0]);
        std::process::exit(1);
    }

    // Parse arguments
    let mut fast_mode: Option<bool> = None;
    let mut json_output = false;
    let mut pbf_path: Option<&str> = None;

    for arg in &args[1..] {
        match arg.as_str() {
            "--fast" => fast_mode = Some(true),
            "--full" => fast_mode = Some(false),
            "--json" => json_output = true,
            _ => {
                if pbf_path.is_none() {
                    pbf_path = Some(arg);
                } else {
                    eprintln!("Error: Multiple file paths specified");
                    std::process::exit(1);
                }
            }
        }
    }

    let pbf_path = match pbf_path {
        Some(path) => Path::new(path),
        None => {
            eprintln!("Error: No OSM PBF file specified");
            std::process::exit(1);
        }
    };

    // Determine fast_mode if not explicitly set
    let fast_mode = fast_mode.unwrap_or_else(|| {
        if pbf_path.exists() {
            match pbf_path.metadata() {
                Ok(metadata) => metadata.len() > 10_000_000, // > 10MB use fast mode
                Err(_) => true, // Default to fast mode if can't read metadata
            }
        } else {
            true
        }
    });

    if !pbf_path.exists() {
        eprintln!("Error: File {} does not exist", pbf_path.display());
        std::process::exit(1);
    }

    if !pbf_path.extension().map_or(false, |ext| ext == "pbf") && !json_output {
        eprintln!("Warning: File does not have .pbf extension");
    }

    if !json_output {
        println!("Graph Metrics Calculator");
        println!("OSM PBF file: {}", pbf_path.display());
    }

    // Load the graph
    let graph = read_osm_pbf_map(pbf_path, json_output);

    // Calculate metrics
    let mut results = calculate_metrics(&graph, fast_mode, json_output);
    results.graph_info.file_path = pbf_path.display().to_string();

    if json_output {
        // Output JSON
        let json = serde_json::to_string_pretty(&results)?;
        println!("{}", json);
    } else {
        // Regular output (already printed during calculation)
        println!("\n=== Summary ===");
        println!(
            "Mode: {} | Total execution time: {:.2}s",
            results.execution.mode, results.execution.total_time_seconds
        );
    }

    Ok(())
}
