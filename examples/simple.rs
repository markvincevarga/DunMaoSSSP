fn main() {
    // Create a larger graph to showcase the new algorithm
    let mut graph = Graph::new(25);

    // Create a more complex graph structure
    for i in 0..24 {
        graph.add_edge(i, i + 1, (i % 5 + 1) as f64);
        if i >= 5 {
            graph.add_edge(i, i - 5, 2.0);
        }
    }

    // Add some cross connections
    graph.add_edge(0, 10, 15.0);
    graph.add_edge(5, 20, 12.0);
    graph.add_edge(12, 3, 8.0);
    graph.add_edge(18, 7, 6.0);

    let mut solver = SSSpSolver::new(graph);
    let distances = solver.solve(0);

    println!("Shortest distances from vertex 0 (using new O(m log^(2/3) n) algorithm):");
    for (i, &dist) in distances.iter().enumerate() {
        if dist == INFINITY {
            println!("  {} -> ∞", i);
        } else {
            println!("  {} -> {:.1}", i, dist);
        }
    }
}
