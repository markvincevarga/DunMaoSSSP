use criterion::{Criterion, criterion_group, criterion_main};
use fast_sssp::DuanMaoSolverV2;
use fast_sssp::graph::Graph;
use petgraph::algo::dijkstra;
use petgraph::graph::DiGraph;
use rand::seq::SliceRandom;
use std::hint::black_box;
use std::path::Path;

#[path = "../tests/graph_loader.rs"]
mod graph_loader;

fn run_fast_sssp_sequential(graph: &Graph, pairs: &[(usize, usize)]) {
    let mut solver = DuanMaoSolverV2::new(graph.clone());
    for (source, goal) in pairs {
        black_box(solver.solve(*source, *goal));
    }
}

fn run_petgraph_dijkstra(graph: &DiGraph<(), f64>, pairs: &[(usize, usize)]) {
    for (source, goal) in pairs {
        let source_node = graph.node_indices().find(|i| i.index() == *source).unwrap();
        let goal_node = graph.node_indices().find(|i| i.index() == *goal).unwrap();
        black_box(dijkstra(&graph, source_node, Some(goal_node), |e| {
            *e.weight()
        }));
    }
}

fn benchmark(c: &mut Criterion) {
    let path = Path::new("tests/test_data/Rome99");
    let fast_sssp_graph = graph_loader::read_dimacs_graph_for_fast_sssp(path);
    let (petgraph_graph, _) = graph_loader::read_dimacs_graph_for_petgraph(path);

    let mut rng = rand::rng();
    let mut nodes: Vec<usize> = (0..fast_sssp_graph.vertices).collect();
    nodes.shuffle(&mut rng);
    let pairs: Vec<(usize, usize)> = nodes
        .chunks(2)
        .map(|chunk| (chunk[0], chunk[1]))
        .take(10)
        .collect();

    let mut group = c.benchmark_group("Rome99 SSSP");

    group.bench_function("fast_sssp_sequential", |b| {
        b.iter(|| run_fast_sssp_sequential(black_box(&fast_sssp_graph), black_box(&pairs)))
    });

    group.bench_function("petgraph_dijkstra", |b| {
        b.iter(|| run_petgraph_dijkstra(black_box(&petgraph_graph), black_box(&pairs)))
    });

    group.finish();
}

criterion_group!(benches, benchmark);
criterion_main!(benches);
