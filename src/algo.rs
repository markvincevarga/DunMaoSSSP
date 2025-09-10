use crate::graph::Graph;
use fibonacci_heap::FibonacciHeap;
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

#[derive(Copy, Clone, PartialEq, PartialOrd)]
struct OrderedFloat(f64);

impl Eq for OrderedFloat {}

impl Ord for OrderedFloat {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.partial_cmp(&other.0).unwrap_or(Ordering::Equal)
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

pub fn dijkstra_fibonacci(graph: &Graph, start: usize, goal: usize) -> Option<f64> {
    if start >= graph.vertices || goal >= graph.vertices {
        return None;
    }

    let mut dist = vec![f64::INFINITY; graph.vertices];
    let mut heap = FibonacciHeap::new();
    let mut node_ptrs = vec![None; graph.vertices];
    let mut positions = vec![None; graph.vertices];

    dist[start] = 0.0;
    let start_node = heap.insert(OrderedFloat(0.0)).unwrap();
    node_ptrs[start] = Some(start_node.clone());
    positions[start] = Some(start);

    while let Some(ordered_cost) = heap.extract_min() {
        let cost = ordered_cost.0;

        let position = positions.iter().position(|&pos| {
            if let Some(p) = pos {
                dist[p] == cost && node_ptrs[p].is_some()
            } else {
                false
            }
        });

        if let Some(pos_idx) = position {
            if let Some(actual_pos) = positions[pos_idx] {
                node_ptrs[actual_pos] = None;
                positions[pos_idx] = None;

                if actual_pos == goal {
                    return Some(cost);
                }

                if cost > dist[actual_pos] {
                    continue;
                }

                for edge in &graph.edges[actual_pos] {
                    let new_cost = cost + edge.weight;
                    let to = edge.to;

                    if new_cost < dist[to] {
                        dist[to] = new_cost;

                        if let Some(ref node) = node_ptrs[to] {
                            heap.decrease_key(node, OrderedFloat(new_cost)).ok();
                        } else {
                            let new_node = heap.insert(OrderedFloat(new_cost)).unwrap();
                            node_ptrs[to] = Some(new_node);
                            if let Some(empty_pos) = positions.iter().position(|&x| x.is_none()) {
                                positions[empty_pos] = Some(to);
                            } else {
                                positions.push(Some(to));
                            }
                        }
                    }
                }
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

    #[test]
    fn test_dijkstra_fibonacci_simple_path() {
        let mut graph = Graph::new(4);
        graph.add_edge(0, 1, 1.0);
        graph.add_edge(1, 2, 2.0);
        graph.add_edge(2, 3, 1.0);
        graph.add_edge(0, 3, 5.0);

        let result = dijkstra_fibonacci(&graph, 0, 3);
        assert!(result.is_some());

        let cost = result.unwrap();
        assert_eq!(cost, 4.0);
    }

    #[test]
    fn test_dijkstra_fibonacci_no_path() {
        let mut graph = Graph::new(3);
        graph.add_edge(0, 1, 1.0);

        let result = dijkstra_fibonacci(&graph, 0, 2);
        assert!(result.is_none());
    }

    #[test]
    fn test_dijkstra_fibonacci_same_start_goal() {
        let graph = Graph::new(3);

        let result = dijkstra_fibonacci(&graph, 0, 0);
        assert!(result.is_some());

        let cost = result.unwrap();
        assert_eq!(cost, 0.0);
    }

    #[test]
    fn test_dijkstra_fibonacci_invalid_nodes() {
        let graph = Graph::new(3);

        assert!(dijkstra_fibonacci(&graph, 0, 5).is_none());
        assert!(dijkstra_fibonacci(&graph, 5, 0).is_none());
    }

    #[test]
    fn test_dijkstra_fibonacci_vs_binary_heap() {
        let mut graph = Graph::new(5);
        graph.add_edge(0, 1, 2.0);
        graph.add_edge(0, 2, 4.0);
        graph.add_edge(1, 2, 1.0);
        graph.add_edge(1, 3, 7.0);
        graph.add_edge(2, 4, 3.0);
        graph.add_edge(3, 4, 2.0);

        let binary_result = dijkstra(&graph, 0, 4);
        let fibonacci_result = dijkstra_fibonacci(&graph, 0, 4);

        assert!(binary_result.is_some());
        assert!(fibonacci_result.is_some());
        assert_eq!(binary_result.unwrap(), fibonacci_result.unwrap());
        assert_eq!(binary_result.unwrap(), 6.0);
    }
}
