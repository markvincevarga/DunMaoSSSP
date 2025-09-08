use crate::graph::Graph;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

#[derive(Copy, Clone, PartialEq)]
struct State {
    cost: f64,
    position: usize,
}

impl Eq for State {}

impl Ord for State {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .cost
            .partial_cmp(&self.cost)
            .unwrap_or(Ordering::Equal)
            .then_with(|| self.position.cmp(&other.position))
    }
}

impl PartialOrd for State {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub fn dijkstra(graph: &Graph, start: usize, goal: usize) -> Option<f64> {
    if start >= graph.vertices || goal >= graph.vertices {
        return None;
    }

    let mut dist = vec![f64::INFINITY; graph.vertices];
    let mut heap = BinaryHeap::new();

    dist[start] = 0.0;
    heap.push(State {
        cost: 0.0,
        position: start,
    });

    while let Some(State { cost, position }) = heap.pop() {
        if position == goal {
            return Some(cost);
        }

        if cost > dist[position] {
            continue;
        }

        for edge in &graph.edges[position] {
            let next = State {
                cost: cost + edge.weight,
                position: edge.to,
            };

            if next.cost < dist[next.position] {
                heap.push(next);
                dist[next.position] = next.cost;
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dijkstra_simple_path() {
        let mut graph = Graph::new(4);
        graph.add_edge(0, 1, 1.0);
        graph.add_edge(1, 2, 2.0);
        graph.add_edge(2, 3, 1.0);
        graph.add_edge(0, 3, 5.0);

        let result = dijkstra(&graph, 0, 3);
        assert!(result.is_some());

        let cost = result.unwrap();
        assert_eq!(cost, 4.0);
    }

    #[test]
    fn test_dijkstra_no_path() {
        let mut graph = Graph::new(3);
        graph.add_edge(0, 1, 1.0);

        let result = dijkstra(&graph, 0, 2);
        assert!(result.is_none());
    }

    #[test]
    fn test_dijkstra_same_start_goal() {
        let graph = Graph::new(3);

        let result = dijkstra(&graph, 0, 0);
        assert!(result.is_some());

        let cost = result.unwrap();
        assert_eq!(cost, 0.0);
    }

    #[test]
    fn test_dijkstra_invalid_nodes() {
        let graph = Graph::new(3);

        assert!(dijkstra(&graph, 0, 5).is_none());
        assert!(dijkstra(&graph, 5, 0).is_none());
    }
}
