use criterion::{Criterion, criterion_group, criterion_main};
use fast_sssp::DuanMaoSolverV2;
use fast_sssp::algo::dijkstra as own_dijkstra;
use fast_sssp::graph::Graph;
use petgraph::algo::dijkstra;
use petgraph::graph::DiGraph;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
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

fn run_own_dijkstra(graph: &Graph, pairs: &[(usize, usize)]) {
    for (source, goal) in pairs {
        black_box(own_dijkstra(graph, *source, *goal));
    }
}

fn benchmark(c: &mut Criterion) {
    let path = Path::new("data/gotland.osm.pbf");
    let fast_sssp_graph = graph_loader::read_osm_pbf_map(path);
    let (petgraph_graph, _) = graph_loader::convert_to_petgraph(&fast_sssp_graph);
    let own_graph = fast_sssp_graph.clone();

    let mut rng = StdRng::seed_from_u64(42);
    let pairs: Vec<(usize, usize)> = (0..10)
        .map(|_| {
            (
                rng.random_range(0..fast_sssp_graph.vertices),
                rng.random_range(0..fast_sssp_graph.vertices),
            )
        })
        .collect();
    println!("Benchmarking pairs: {:?}", pairs);
    let mut group = c.benchmark_group("Stockholm SSSP");

    group.bench_function("fast_sssp_sequential", |b| {
        b.iter(|| run_fast_sssp_sequential(black_box(&fast_sssp_graph), black_box(&pairs)))
    });

    group.bench_function("petgraph_dijkstra", |b| {
        b.iter(|| run_petgraph_dijkstra(black_box(&petgraph_graph), black_box(&pairs)))
    });

    group.bench_function("own_dijkstra", |b| {
        b.iter(|| run_own_dijkstra(black_box(&own_graph), black_box(&pairs)))
    });

    group.finish();
}

criterion_group!(benches, benchmark);
criterion_main!(benches);
