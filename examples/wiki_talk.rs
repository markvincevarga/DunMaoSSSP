use bincode;
use std::fs::File;
use std::path::Path;
use std::time::Instant;

use fast_sssp::{Graph, SSSpSolver};

fn load_graph(path: &Path) -> Result<Graph, Box<dyn std::error::Error>> {
    println!("Loading graph from {}...", path.display());
    let file = File::open(path)?;
    let graph: Graph = bincode::deserialize(file)?;
    Ok(graph)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let data_path = Path::new("data/wiki-talk-graph.bin");

    if !data_path.exists() {
        println!("Wiki-Talk graph not found!");
        println!("Please run: cargo run --bin fetch_data");
        return Ok(());
    }

    println!("Loading Wiki-Talk dataset...");
    let graph = load_graph(data_path)?;

    println!(
        "Graph loaded: {} vertices, {} edges",
        graph.vertices,
        graph.edge_count()
    );

    // Test on a subset of source vertices
    let test_sources = vec![0, 100, 1000, 5000];

    println!("\nBenchmarking on Wiki-Talk dataset:");
    println!(
        "{:<8} {:<15} {:<15} {:<12}",
        "Source", "Dijkstra (ms)", "New Algo (ms)", "Speedup"
    );
    println!("{}", "-".repeat(55));

    for &source in &test_sources {
        if source >= graph.vertices {
            continue;
        }

        // Benchmark Dijkstra
        let start = Instant::now();
        let mut solver1 = SSSpSolver::new(graph.clone());
        let _distances1 = solver1.solve_with_dijkstra(source);
        let dijkstra_time = start.elapsed().as_millis();

        // Benchmark new algorithm
        let start = Instant::now();
        let mut solver2 = SSSpSolver::new(graph.clone());
        let _distances2 = solver2.solve(source);
        let new_algo_time = start.elapsed().as_millis();

        let speedup = if new_algo_time > 0 {
            dijkstra_time as f64 / new_algo_time as f64
        } else {
            0.0
        };

        println!(
            "{:<8} {:<15} {:<15} {:<12.2}x",
            source, dijkstra_time, new_algo_time, speedup
        );
    }

    println!(
        "\nNote: This is a real-world directed graph with {} vertices and {} edges",
        graph.vertices,
        graph.edge_count()
    );
    println!("The new algorithm should show improvements on this scale!");

    Ok(())
}
