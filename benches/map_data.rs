use criterion::{Criterion, criterion_group, criterion_main};
use fast_sssp::DuanMaoSolverV2;
use fast_sssp::algo::dijkstra as own_dijkstra;
use fast_sssp::algo::dijkstra_fibonacci as own_dijkstra_fib;
use fast_sssp::graph::Graph;
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

fn run_own_dijkstra_fib(graph: &Graph, pairs: &[(usize, usize)]) {
    for (source, goal) in pairs {
        black_box(own_dijkstra_fib(graph, *source, *goal));
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
    let own_graph = fast_sssp_graph.clone();
    let own_graph_fib = fast_sssp_graph.clone();

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

    group.bench_function("own_dijkstra", |b| {
        b.iter(|| run_own_dijkstra(black_box(&own_graph), black_box(&pairs)))
    });

    group.bench_function("own_dijkstra_fibonacci", |b| {
        b.iter(|| run_own_dijkstra_fib(black_box(&own_graph_fib), black_box(&pairs)))
    });

    group.bench_function("fast_sssp_sequential", |b| {
        b.iter(|| run_fast_sssp_sequential(black_box(&fast_sssp_graph), black_box(&pairs)))
    });

    group.finish();
}

criterion_group!(benches, benchmark);
criterion_main!(benches);
