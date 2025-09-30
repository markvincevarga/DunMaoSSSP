#[cfg(feature = "bincode")]
use bincode::{Decode, Encode};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "bincode", derive(Decode, Encode))]
pub struct Edge {
    pub to: usize,
    pub weight: f64,
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "bincode", derive(Decode, Encode))]
pub struct Graph {
    pub vertices: usize,
    pub edges: Vec<Vec<Edge>>,
}

impl Graph {
    pub fn new(vertices: usize) -> Self {
        Graph {
            vertices,
            edges: vec![Vec::new(); vertices],
        }
    }

    pub fn add_edge(&mut self, from: usize, to: usize, weight: f64) {
        self.edges[from].push(Edge { to, weight });
    }

    pub fn edge_count(&self) -> usize {
        self.edges.iter().map(|adj| adj.len()).sum()
    }

    #[cfg(feature = "bincode")]
    pub fn from_file(path: &std::path::Path) -> Result<Graph, Box<dyn std::error::Error>> {
        let file = std::fs::File::open(path)?;
        let config = bincode::config::legacy();
        let reader = std::io::BufReader::new(file);
        let graph: Graph = bincode::decode_from_reader(reader, config)?;
        Ok(graph)
    }

    #[cfg(feature = "bincode")]
    pub fn to_file(
        graph: &Graph,
        path: &std::path::Path,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let file = std::fs::File::create(path)?;
        let config = bincode::config::legacy();
        let mut writer = std::io::BufWriter::new(file);

        bincode::serde::encode_into_std_write(graph, &mut writer, config)?;

        Ok(())
    }

    #[cfg(feature = "petgraph")]
    pub fn to_petgraph(&self) -> petgraph::Graph<usize, f64> {
        let mut pg_graph = petgraph::Graph::new();
        let nodes: Vec<_> = (0..self.vertices).map(|i| pg_graph.add_node(i)).collect();

        for (from_idx, from_node) in self.edges.iter().enumerate() {
            for edge in from_node {
                pg_graph.add_edge(nodes[from_idx], nodes[edge.to], edge.weight);
            }
        }
        pg_graph
    }

    pub fn degrees_sorted(&self) -> Vec<usize> {
        let mut degrees: Vec<usize> = self.edges.iter().map(|adj| adj.len()).collect();
        degrees.sort_unstable();
        degrees
    }

    pub fn bfs_distances(&self, start: usize) -> HashMap<usize, usize> {
        let mut distances = HashMap::new();
        let mut queue = VecDeque::new();

        distances.insert(start, 0);
        queue.push_back(start);

        while let Some(node) = queue.pop_front() {
            let current_dist = distances[&node];

            for edge in &self.edges[node] {
                if !distances.contains_key(&edge.to) {
                    distances.insert(edge.to, current_dist + 1);
                    queue.push_back(edge.to);
                }
            }
        }

        distances
    }

    pub fn strongly_connected_component(&self, start: usize) -> HashSet<usize> {
        let distances = self.bfs_distances(start);
        distances.keys().cloned().collect()
    }

    pub fn diameter(&self) -> Option<usize> {
        let mut max_distance = 0;

        for vertex in 0..self.vertices {
            let distances = self.bfs_distances(vertex);
            if let Some(&max_dist) = distances.values().max() {
                max_distance = max_distance.max(max_dist);
            }
        }

        if max_distance == 0 && self.vertices > 1 {
            None
        } else {
            Some(max_distance)
        }
    }

    pub fn average_path_length_scc(&self, start: usize) -> Option<f64> {
        let scc = self.strongly_connected_component(start);
        let scc_size = scc.len();

        if scc_size <= 1 {
            return None;
        }

        let mut total_distance = 0usize;
        let mut path_count = 0usize;

        for &node in &scc {
            let distances = self.bfs_distances(node);

            for &other_node in &scc {
                if node != other_node {
                    if let Some(&dist) = distances.get(&other_node) {
                        total_distance += dist;
                        path_count += 1;
                    }
                }
            }
        }

        if path_count == 0 {
            None
        } else {
            Some(total_distance as f64 / path_count as f64)
        }
    }

    pub fn clustering_coefficient(&self) -> Vec<f64> {
        let mut coefficients = Vec::with_capacity(self.vertices);

        for node in 0..self.vertices {
            let neighbors: HashSet<usize> = self.edges[node].iter().map(|e| e.to).collect();
            let k = neighbors.len();

            if k < 2 {
                coefficients.push(0.0);
                continue;
            }

            let mut triangle_count = 0;

            for &neighbor1 in &neighbors {
                for &neighbor2 in &neighbors {
                    if neighbor1 < neighbor2 {
                        if self.edges[neighbor1].iter().any(|e| e.to == neighbor2) {
                            triangle_count += 1;
                        }
                    }
                }
            }

            let possible_triangles = k * (k - 1) / 2;
            let coefficient = triangle_count as f64 / possible_triangles as f64;
            coefficients.push(coefficient);
        }

        coefficients
    }
}
