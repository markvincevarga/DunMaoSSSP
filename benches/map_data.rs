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

fn bench_file(c: &mut Criterion, path: &Path, samples: usize, num_pairs: usize) {
    let name = path
        .file_stem()
        .expect("file should have name")
        .to_string_lossy();

    let fast_sssp_graph = graph_loader::read_osm_pbf_map(path);

    let own_graph = fast_sssp_graph.clone();
    let own_graph_fib = fast_sssp_graph.clone();

    let mut rng = StdRng::seed_from_u64(42);
    let pairs: Vec<(usize, usize)> = (0..num_pairs)
        .map(|_| {
            (
                rng.random_range(0..fast_sssp_graph.vertices),
                rng.random_range(0..fast_sssp_graph.vertices),
            )
        })
        .collect();

    let mut group = c.benchmark_group(name);
    group.sample_size(samples);

    group.bench_function("fast_sssp_sequential", |b| {
        b.iter(|| run_fast_sssp_sequential(black_box(&fast_sssp_graph), black_box(&pairs)))
    });

    group.bench_function("dijkstra", |b| {
        b.iter(|| run_own_dijkstra(black_box(&own_graph), black_box(&pairs)))
    });

    group.bench_function("dijkstra_fibonacci", |b| {
        b.iter(|| run_own_dijkstra_fib(black_box(&own_graph_fib), black_box(&pairs)))
    });

    group.finish();
}

fn benchmark(c: &mut Criterion) {
    [
        // (path, samples, num_pairs)
        ("jan_mayen", 50, 250),  // Size: 97K
        ("gibraltar", 50, 500),  // Size: 417K
        ("monaco", 50, 250),     // Size: 423K
        ("san_marino", 50, 100), // Size: 1.1M
        ("andorra", 25, 75),     // Size: 2.7M
        ("gotland", 25, 75),     // Size: 4.8M
        ("malta", 25, 75),       // Size: 6.9M
        ("reykjavik", 25, 75),   // Size: 8.9M
        ("budapest", 10, 75),    // Size: 25M
        ("luxembourg", 10, 25),  // Size: 37M
        ("haiti", 10, 25),       // Size: 51M
        ("iceland", 10, 20),     // Size: 57M
        ("stockholm", 10, 20),   // Size: 59M
        ("missisippi", 10, 20),  // Size: 79M
        ("peru", 10, 5),         // Size: 208M
        ("sweden", 10, 5),       // Size: 692M
    ]
    .iter()
    .for_each(|(name, samples, num_pairs)| {
        bench_file(
            c,
            Path::new(&format!("data/{}.osm.pbf", name)),
            *samples,
            *num_pairs,
        )
    });
}

criterion_group!(benches, benchmark);
criterion_main!(benches);
