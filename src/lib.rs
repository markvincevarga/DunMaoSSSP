#![allow(deprecated)]
pub mod algo;
pub use algo as pathfinding;
pub use algo::dijkstra as find_shortest_path;
pub mod graph;
pub mod sequential_v2;
pub mod utils;

pub use graph::{Edge, Graph};

pub use sequential_v2::DuanMaoSolverV2;
