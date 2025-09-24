#![allow(dead_code)]
use fast_sssp::Graph;
use geo::{Distance, Haversine, Point};
use num_format::{Locale, ToFormattedString};
use osmpbf::{Element, ElementReader};
use petgraph::graph::{DiGraph, NodeIndex};

use std::collections::HashMap;

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

const CACHE_PATH: &str = "./data/cache";

#[cfg(feature = "flate2")]
use flate2::read::GzDecoder;
use tqdm::tqdm;

#[allow(dead_code)]
pub fn read_dimacs_graph_for_petgraph(
    path: &Path,
) -> (DiGraph<(), f64>, HashMap<usize, NodeIndex>) {
    let file = File::open(path).unwrap();
    let reader = BufReader::new(file);
    let mut graph = DiGraph::new();
    let mut node_map = HashMap::new();

    for line in reader.lines() {
        let line = line.unwrap();
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.is_empty() {
            continue;
        }

        match parts[0] {
            "c" => continue, // Comment
            "p" => {
                // Problem line: p sp <nodes> <edges>
                let vertices = parts[2].parse::<usize>().unwrap();
                for i in 1..=vertices {
                    let node = graph.add_node(());
                    node_map.insert(i, node);
                }
            }
            "a" => {
                // Arc descriptor: a <from> <to> <weight>
                let from = parts[1].parse::<usize>().unwrap();
                let to = parts[2].parse::<usize>().unwrap();
                let weight = parts[3].parse::<f64>().unwrap();
                let from_node = node_map[&from];
                let to_node = node_map[&to];
                graph.add_edge(from_node, to_node, weight);
            }
            _ => continue,
        }
    }
    (graph, node_map)
}

#[cfg(feature = "flate2")]
pub fn read_wiki_graph_for_petgraph(
    gz_path: &Path,
) -> (DiGraph<(), f64>, HashMap<usize, NodeIndex>) {
    let file = File::open(gz_path).unwrap();
    let gz = GzDecoder::new(file);
    let reader = BufReader::new(gz);

    let mut edges = Vec::new();
    let mut node_map = HashMap::new();
    let mut graph = DiGraph::new();
    let mut max_node = 0;

    for line in reader.lines() {
        let line = line.unwrap();
        if line.starts_with('#') {
            continue;
        }
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() >= 2 {
            let u = parts[0].parse::<usize>().unwrap();
            let v = parts[1].parse::<usize>().unwrap();
            edges.push((u, v));
            max_node = max_node.max(u).max(v);
        }
    }

    for i in 0..=max_node {
        let node = graph.add_node(());
        node_map.insert(i, node);
    }

    for (u, v) in edges {
        let from_node = node_map[&u];
        let to_node = node_map[&v];
        graph.add_edge(from_node, to_node, 1.0);
    }

    (graph, node_map)
}

pub fn read_dimacs_graph_for_fast_sssp(path: &Path) -> Graph {
    let file = File::open(path).unwrap();
    let reader = BufReader::new(file);
    let mut graph = None;

    for line in reader.lines() {
        let line = line.unwrap();
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.is_empty() {
            continue;
        }

        match parts[0] {
            "c" => continue, // Comment
            "p" => {
                // Problem line: p sp <nodes> <edges>
                let vertices = parts[2].parse::<usize>().unwrap();
                graph = Some(Graph::new(vertices));
            }
            "a" => {
                // Arc descriptor: a <from> <to> <weight>
                let from = parts[1].parse::<usize>().unwrap() - 1; // Adjust for 0-based indexing
                let to = parts[2].parse::<usize>().unwrap() - 1; // Adjust for 0-based indexing
                let weight = parts[3].parse::<f64>().unwrap();
                if let Some(g) = &mut graph {
                    g.add_edge(from, to, weight);
                }
            }
            _ => continue,
        }
    }
    graph.unwrap()
}

#[cfg(feature = "flate2")]
pub fn read_wiki_graph_for_fast_sssp(gz_path: &Path) -> Graph {
    let file = File::open(gz_path).unwrap();
    let gz = GzDecoder::new(file);
    let reader = BufReader::new(gz);

    let mut edges = Vec::new();
    let mut max_node = 0;

    for line in reader.lines() {
        let line = line.unwrap();
        if line.starts_with('#') {
            continue;
        }
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() >= 2 {
            let u = parts[0].parse::<usize>().unwrap();
            let v = parts[1].parse::<usize>().unwrap();
            edges.push((u, v));
            max_node = max_node.max(u).max(v);
        }
    }

    let mut graph = Graph::new(max_node + 1);
    for (u, v) in edges {
        graph.add_edge(u, v, 1.0);
    }

    graph
}

pub fn read_osm_pbf_map(pbf_path: &Path) -> Graph {
    let filename = pbf_path.file_name().expect("Path should have a filename");
    let cache_path = Path::new(CACHE_PATH).join(Path::new(filename).with_extension("cache"));
    let graph = Graph::from_file(&cache_path).unwrap_or_else(|_| {
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
                                idx: idx,
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
                                idx: idx,
                                lat: node.lat(),
                                lon: node.lon(),
                            },
                        );
                        idx += 1;
                    }
                    Element::Way(way) => {
                        let ways_iter_clone = way.refs().clone();
                        for (a, b) in way.refs().zip(ways_iter_clone.skip(1)) {
                            edges.push(OsmEdge { from: a, to: b })
                        }
                    }
                    _ => {}
                };
            })
            .expect("file must include correct osm data");

        let mut graph = Graph::new(nodes.len());

        println!(
            "Number of nodes: {:}, Number of edges: {:}",
            nodes.len().to_formatted_string(&Locale::is),
            edges.len().to_formatted_string(&Locale::is)
        );
        tqdm(edges.iter()).for_each(|edge| {
            let node_a = &nodes[&edge.from];
            let node_b = &nodes[&edge.to];
            let distance = Haversine.distance(
                Point::new(node_a.lon, node_a.lat),
                Point::new(node_b.lon, node_b.lat),
            );
            graph.add_edge(node_a.idx, node_b.idx, distance);
            graph.add_edge(node_b.idx, node_a.idx, distance);
        });

        std::fs::create_dir_all(CACHE_PATH).expect("Failed to create cache directory");
        Graph::to_file(&graph, &cache_path).expect("saving to cache file should succeed");

        graph
    });
    println!(
        "Number of edges: {:} number of nodes: {:}",
        graph.edge_count().to_formatted_string(&Locale::is),
        graph.vertices.to_formatted_string(&Locale::is)
    );
    graph
}

pub fn convert_to_petgraph(graph: &Graph) -> (DiGraph<(), f64>, HashMap<usize, NodeIndex>) {
    let mut petgraph_graph = DiGraph::new();
    let mut node_map = HashMap::new();

    for i in 0..graph.vertices {
        let node = petgraph_graph.add_node(());
        node_map.insert(i, node);
    }

    for (from, edges) in graph.edges.iter().enumerate() {
        for edge in edges {
            let from_node = node_map[&from];
            let to_node = node_map[&edge.to];
            petgraph_graph.add_edge(from_node, to_node, edge.weight);
        }
    }

    (petgraph_graph, node_map)
}
