use serde_json;
use std::fs::File;
use std::io::BufReader;
use std::time::Duration;

mod models {
    use rand::seq::SliceRandom;
    use std::collections::HashMap;

    #[derive(serde::Deserialize, Debug)]
    pub struct Depot {
        pub return_time: f64,
        pub x_coord: f64,
        pub y_coord: f64,
    }

    #[derive(serde::Deserialize, Debug)]
    pub struct Patient {
        pub x_coord: f64,
        pub y_coord: f64,
        pub demand: f64,
        pub start_time: f64,
        pub end_time: f64,
        pub care_time: f64,
    }

    #[derive(serde::Deserialize, Debug)]
    pub struct ProblemInstance {
        pub instance_name: String,
        pub nbr_nurses: usize,
        pub capacity_nurse: f64,
        pub benchmark: f64,
        pub depot: Depot,
        pub patients: HashMap<String, Patient>,
        pub travel_times: Vec<Vec<f64>>,
    }

    // Solution representation with routes and cached flat permutation
    #[derive(Debug, Clone)]
    pub struct Solution {
        pub routes: Vec<Vec<usize>>,
        pub flat: Vec<usize>,
    }

    /// Simulerer én etappe for en rute.
    /// Returnerer (ny finish tid, reisetid for etappen, oppdatert demand) dersom pasienten kan
    /// legges til uten å bryte constraints.
    pub fn simulate_leg(
        current_duration: f64,
        last_node: usize,
        current_demand: f64,
        patient_id: usize,
        instance: &ProblemInstance,
    ) -> Option<(f64, f64, f64)> {
        let patient = instance.patients.get(&patient_id.to_string())?;
        let travel_time = instance.travel_times[last_node][patient_id];
        let arrival_time = current_duration + travel_time;
        let start_service_time = arrival_time.max(patient.start_time);
        let finish_time = start_service_time + patient.care_time;
        let new_demand = current_demand + patient.demand;
        // Ta med retur til depot (indeks 0)
        let return_travel_time = instance.travel_times[patient_id][0];
        let potential_total_duration = finish_time + return_travel_time;
        if new_demand <= instance.capacity_nurse
            && finish_time <= patient.end_time
            && potential_total_duration <= instance.depot.return_time
        {
            Some((finish_time, travel_time, new_demand))
        } else {
            None
        }
    }

    /// Simulerer en hel rute (uten depotoppføring) og returnerer (total reisetid, total varighet)
    /// om gyldig.
    pub fn simulate_route(route: &[usize], instance: &ProblemInstance) -> Option<(f64, f64)> {
        let depot_index = 0;
        let mut total_travel_time = 0.0;
        let mut total_duration = 0.0;
        let mut current_demand = 0.0;
        let mut current_node = depot_index;
        for &patient_id in route {
            let (new_duration, travel, new_demand) = simulate_leg(
                total_duration,
                current_node,
                current_demand,
                patient_id,
                instance,
            )?;
            total_travel_time += travel;
            total_duration = new_duration;
            current_demand = new_demand;
            current_node = patient_id;
        }
        total_travel_time += instance.travel_times[current_node][depot_index];
        total_duration += instance.travel_times[current_node][depot_index];
        if total_duration > instance.depot.return_time {
            None
        } else {
            Some((total_travel_time, total_duration))
        }
    }

    /// Deler en flat permutasjon inn i ruter slik at alle constraints overholdes.
    /// Hver pasient plasseres nøyaktig én gang.
    pub fn split_permutation_into_routes(
        permutation: &[usize],
        instance: &ProblemInstance,
    ) -> Vec<Vec<usize>> {
        let depot_index = 0;
        let mut routes = Vec::new();
        let mut current_route = Vec::new();
        let mut current_duration = 0.0;
        let mut current_demand = 0.0;
        let mut last_node = depot_index;
        for &patient_id in permutation {
            if let Some((finish_time, _travel, new_demand)) = simulate_leg(
                current_duration,
                last_node,
                current_demand,
                patient_id,
                instance,
            ) {
                current_route.push(patient_id);
                current_duration = finish_time;
                current_demand = new_demand;
                last_node = patient_id;
            } else {
                routes.push(current_route.clone());
                // Start en ny rute med denne pasienten fra depot
                let travel_time = instance.travel_times[depot_index][patient_id];
                let patient = instance.patients.get(&patient_id.to_string()).unwrap();
                let arrival_time = travel_time;
                let start_service_time = arrival_time.max(patient.start_time);
                let finish_time = start_service_time + patient.care_time;
                current_route = vec![patient_id];
                current_duration = finish_time;
                current_demand = patient.demand;
                last_node = patient_id;
            }
        }
        if !current_route.is_empty() {
            routes.push(current_route);
        }
        routes
    }

    /// Quick validity check for a route
    pub fn is_route_valid(route: &[usize], instance: &ProblemInstance) -> bool {
        simulate_route(route, instance).is_some()
    }

    /// Konstruerer en gyldig løsning basert på en tilfeldig permutasjon.
    pub fn generate_valid_solution(instance: &ProblemInstance) -> Solution {
        let mut perm: Vec<usize> = (1..=instance.patients.len()).collect();
        perm.shuffle(&mut rand::thread_rng());
        let routes = split_permutation_into_routes(&perm, instance);
        Solution { routes, flat: perm }
    }

    /// OPTIMIZATION: Time-window based solution generator
    /// Generates a solution by ordering patients by earliest start time
    pub fn generate_time_window_solution(instance: &ProblemInstance) -> Solution {
        let mut patients: Vec<(usize, f64)> = 
            (1..=instance.patients.len())
            .map(|id| {
                let patient = instance.patients.get(&id.to_string()).unwrap();
                (id, patient.start_time)
            })
            .collect();
        
        // Sort by earliest start time
        patients.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        
        let perm: Vec<usize> = patients.iter().map(|(id, _)| *id).collect();
        let routes = split_permutation_into_routes(&perm, instance);
        Solution { routes, flat: perm }
    }

    /// OPTIMIZATION: Nearest neighbor heuristic solution generator
    pub fn generate_nearest_neighbor_solution(instance: &ProblemInstance) -> Solution {
        let mut unvisited: Vec<usize> = (1..=instance.patients.len()).collect();
        let mut perm = Vec::with_capacity(unvisited.len());
        
        // Start from depot (node 0)
        let mut current_node = 0;
        
        // Build solution using nearest neighbor heuristic
        while !unvisited.is_empty() {
            // Find nearest unvisited patient
            let next_index = unvisited
                .iter()
                .enumerate()
                .min_by(|(_, &a), (_, &b)| {
                    instance.travel_times[current_node][a]
                        .partial_cmp(&instance.travel_times[current_node][b])
                        .unwrap()
                })
                .map(|(index, _)| index)
                .unwrap();
            
            let next_node = unvisited.remove(next_index);
            perm.push(next_node);
            current_node = next_node;
        }
        
        let routes = split_permutation_into_routes(&perm, instance);
        Solution { routes, flat: perm }
    }

    /// IMPROVEMENT: Generate solution based on time windows and clusters
    pub fn generate_cluster_solution(instance: &ProblemInstance) -> Solution {
        // Create clusters based on geographic proximity
        let num_clusters = (instance.patients.len() as f64 / 8.0).ceil() as usize;
        let mut centroids: Vec<(f64, f64)> = Vec::with_capacity(num_clusters);
        let mut rng = rand::thread_rng();
        
        // Initialize centroids with random patients
        let mut patient_ids: Vec<usize> = (1..=instance.patients.len()).collect();
        patient_ids.shuffle(&mut rng);
        
        for &id in patient_ids.iter().take(num_clusters) {
            let patient = instance.patients.get(&id.to_string()).unwrap();
            centroids.push((patient.x_coord, patient.y_coord));
        }
        
        // Assign patients to clusters
        let mut clusters: Vec<Vec<usize>> = vec![Vec::new(); num_clusters];
        
        for id in 1..=instance.patients.len() {
            let patient = instance.patients.get(&id.to_string()).unwrap();
            let coords = (patient.x_coord, patient.y_coord);
            
            // Find closest centroid
            let mut min_dist = f64::MAX;
            let mut closest_cluster = 0;
            
            for (i, &centroid) in centroids.iter().enumerate() {
                let dist = ((coords.0 - centroid.0).powi(2) + (coords.1 - centroid.1).powi(2)).sqrt();
                if dist < min_dist {
                    min_dist = dist;
                    closest_cluster = i;
                }
            }
            
            clusters[closest_cluster].push(id);
        }
        
        // Order patients within each cluster by time window
        for cluster in &mut clusters {
            cluster.sort_by(|&a, &b| {
                let patient_a = instance.patients.get(&a.to_string()).unwrap();
                let patient_b = instance.patients.get(&b.to_string()).unwrap();
                patient_a.start_time.partial_cmp(&patient_b.start_time).unwrap()
            });
        }
        
        // Create permutation by selecting from clusters in a round-robin fashion
        let mut perm = Vec::with_capacity(instance.patients.len());
        
        // Instead of BinaryHeap, use a Vec that we'll sort by start time
        let mut queue: Vec<(f64, usize, usize)> = Vec::new();
        
        // Initialize with first patient from each non-empty cluster
        for (i, cluster) in clusters.iter().enumerate() {
            if !cluster.is_empty() {
                let patient = instance.patients.get(&cluster[0].to_string()).unwrap();
                queue.push((patient.start_time, i, 0));
            }
        }
        
        // Process patients in order of increasing start time
        while !queue.is_empty() {
            // Sort by start time (first element of tuple)
            queue.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            
            // Take the patient with earliest start time
            let (_, cluster_idx, pos) = queue.remove(0);
            let patient_id = clusters[cluster_idx][pos];
            perm.push(patient_id);
            
            // Add next patient from this cluster if available
            let next_pos = pos + 1;
            if next_pos < clusters[cluster_idx].len() {
                let next_patient = instance.patients.get(&clusters[cluster_idx][next_pos].to_string()).unwrap();
                queue.push((next_patient.start_time, cluster_idx, next_pos));
            }
        }
        
        let routes = split_permutation_into_routes(&perm, instance);
        Solution { routes, flat: perm }
    }

    /// Validerer at løsningen overholder alle constraints.
    pub fn validate_solution(solution: &Solution, instance: &ProblemInstance) -> bool {
        let mut visited = std::collections::HashSet::new();
        for route in &solution.routes {
            if simulate_route(route, instance).is_none() {
                return false;
            }
            for &p in route {
                if !visited.insert(p) {
                    return false;
                }
            }
        }
        visited.len() == instance.patients.len()
    }
    
    /// OPTIMIZATION: Updates the flat representation from routes
    pub fn update_flat_from_routes(solution: &mut Solution) {
        solution.flat.clear();
        for route in &solution.routes {
            solution.flat.extend(route);
        }
    }
}

mod ga {
    use crate::models::{split_permutation_into_routes, ProblemInstance, Solution};
    use rand::seq::SliceRandom;
    use rand::Rng;
    use rayon::prelude::*;
    use std::collections::HashMap;

    /// Returnerer referansen til den cachede, flate representasjonen.
    pub fn flatten_solution(solution: &Solution) -> &Vec<usize> {
        &solution.flat
    }

    /// Order-1 Crossover (OX1) operatør på flate permutasjoner.
    pub fn order_one_crossover(parent1: &[usize], parent2: &[usize]) -> (Vec<usize>, Vec<usize>) {
        let len = parent1.len();
        let mut rng = rand::thread_rng();
        let i = rng.gen_range(0..len);
        let j = rng.gen_range(0..len);
        let (start, end) = if i < j { (i, j) } else { (j, i) };

        let mut child1: Vec<Option<usize>> = vec![None; len];
        let mut child2: Vec<Option<usize>> = vec![None; len];

        for idx in start..=end {
            child1[idx] = Some(parent1[idx]);
            child2[idx] = Some(parent2[idx]);
        }

        let mut pos_child1 = (end + 1) % len;
        let mut pos_parent2 = (end + 1) % len;
        while child1.iter().any(|gene| gene.is_none()) {
            let gene = parent2[pos_parent2];
            if !child1.contains(&Some(gene)) {
                child1[pos_child1] = Some(gene);
                pos_child1 = (pos_child1 + 1) % len;
            }
            pos_parent2 = (pos_parent2 + 1) % len;
        }

        let mut pos_child2 = (end + 1) % len;
        let mut pos_parent1 = (end + 1) % len;
        while child2.iter().any(|gene| gene.is_none()) {
            let gene = parent1[pos_parent1];
            if !child2.contains(&Some(gene)) {
                child2[pos_child2] = Some(gene);
                pos_child2 = (pos_child2 + 1) % len;
            }
            pos_parent1 = (pos_parent1 + 1) % len;
        }

        (
            child1.into_iter().map(|g| g.unwrap()).collect(),
            child2.into_iter().map(|g| g.unwrap()).collect(),
        )
    }

    /// OPTIMIZATION: Edge Recombination Crossover
    /// Better at preserving adjacency relationships in permutations
    pub fn edge_recombination_crossover(parent1: &[usize], parent2: &[usize]) -> Vec<usize> {
        use std::collections::{HashMap, HashSet};
        let len = parent1.len();
        let mut child = Vec::with_capacity(len);
        
        // Build adjacency list for each node
        let mut edge_map: HashMap<usize, HashSet<usize>> = HashMap::new();
        
        // Helper function to add adjacency relationship
        let add_adj = |map: &mut HashMap<usize, HashSet<usize>>, from: usize, to: usize| {
            map.entry(from).or_insert_with(HashSet::new).insert(to);
        };
        
        // Process parent1
        for i in 0..len {
            let curr = parent1[i];
            let prev = if i == 0 { parent1[len-1] } else { parent1[i-1] };
            let next = if i == len-1 { parent1[0] } else { parent1[i+1] };
            
            add_adj(&mut edge_map, curr, prev);
            add_adj(&mut edge_map, curr, next);
        }
        
        // Process parent2
        for i in 0..len {
            let curr = parent2[i];
            let prev = if i == 0 { parent2[len-1] } else { parent2[i-1] };
            let next = if i == len-1 { parent2[0] } else { parent2[i+1] };
            
            add_adj(&mut edge_map, curr, prev);
            add_adj(&mut edge_map, curr, next);
        }
        
        // Start with a random node from either parent
        let mut rng = rand::thread_rng();
        let start_from_p1 = rng.gen_bool(0.5);
        let mut current = if start_from_p1 { parent1[0] } else { parent2[0] };
        
        // Build child
        child.push(current);
        
        // Use a set to track remaining nodes to select from
        let mut all_nodes: HashSet<usize> = (1..=len).collect();
        all_nodes.remove(&current);
        
        while child.len() < len {
            // Remove current node from all adjacency lists
            for adj_list in edge_map.values_mut() {
                adj_list.remove(&current);
            }
            
            // Find next node
            let next_node = if let Some(adj_list) = edge_map.get(&current) {
                if !adj_list.is_empty() {
                    // Find neighbor with fewest neighbors
                    adj_list.iter()
                        .filter(|&&n| !child.contains(&n)) // Only consider nodes not in child yet
                        .min_by_key(|&&n| edge_map.get(&n).map_or(usize::MAX, |s| s.len()))
                        .cloned()
                } else {
                    None
                }
            } else {
                None
            };
            
            if let Some(next) = next_node {
                current = next;
                child.push(current);
                all_nodes.remove(&current);
            } else {
                // If we get here, we need to select a node not yet in the child
                if let Some(&node) = all_nodes.iter().next() {
                    current = node;
                    child.push(current);
                    all_nodes.remove(&current);
                } else {
                    break; // Should not happen, but as a safeguard
                }
            }
        }
        
        // Safety check - make sure we have a complete permutation
        if child.len() < len {
            for i in 1..=len {
                if !child.contains(&i) {
                    child.push(i);
                }
            }
        }
        
        child
    }

    /// IMPROVEMENT: Cycle Crossover (CX) 
    /// Preserves absolute positions from parents
    pub fn cycle_crossover(parent1: &[usize], parent2: &[usize]) -> (Vec<usize>, Vec<usize>) {
        let len = parent1.len();
        let mut child1 = vec![0; len];
        let mut child2 = vec![0; len];
        
        // Create a lookup from value to position for parent2
        let mut p2_pos = std::collections::HashMap::new();
        for (i, &val) in parent2.iter().enumerate() {
            p2_pos.insert(val, i);
        }
        
        // Track which positions have been filled
        let mut filled = vec![false; len];
        
        // Randomly decide which parent to start from
        let mut rng = rand::thread_rng();
        let start_from_p1 = rng.gen_bool(0.5);
        
        // Process each unfilled position
        for i in 0..len {
            if !filled[i] {
                // Determine which parent to copy from for this cycle
                let use_p1 = if start_from_p1 {
                    i % 2 == 0 // Alternate based on cycle number
                } else {
                    i % 2 == 1
                };
                
                // Follow the cycle
                let mut pos = i;
                while !filled[pos] {
                    filled[pos] = true;
                    
                    if use_p1 {
                        child1[pos] = parent1[pos];
                        child2[pos] = parent2[pos];
                    } else {
                        child1[pos] = parent2[pos];
                        child2[pos] = parent1[pos];
                    }
                    
                    // Move to next position in cycle
                    let value = if use_p1 { parent1[pos] } else { parent2[pos] };
                    pos = *p2_pos.get(&value).unwrap_or(&pos);
                    
                    // Break if we've come back to the starting position
                    if pos == i {
                        break;
                    }
                }
            }
        }
        
        (child1, child2)
    }

    /// OPTIMIZED: Optimized mutation selection function
    pub fn apply_mutations(perm: &mut Vec<usize>, mutation_rate: f64) {
        let mut rng = rand::thread_rng();
        
        // Use a single random number to determine if and which mutation to apply
        let roll = rng.gen_range(0.0..1.0);
        
        if roll < mutation_rate { // Only roll once
            // Select mutation type
            if roll < mutation_rate * 0.3 {
                // Inversion (30% of mutations)
                let len = perm.len();
                let i = rng.gen_range(0..len);
                let j = rng.gen_range(0..len);
                let (start, end) = if i < j { (i, j) } else { (j, i) };
                perm[start..=end].reverse();
            } else if roll < mutation_rate * 0.6 {
                // Insertion (30% of mutations)
                let len = perm.len();
                let from = rng.gen_range(0..len);
                let to = rng.gen_range(0..len);
                
                if from != to {
                    let value = perm.remove(from);
                    let adjusted_to = if to > from { to - 1 } else { to };
                    perm.insert(adjusted_to, value);
                }
            } else if roll < mutation_rate * 0.9 {
                // Swap (30% of mutations)
                let len = perm.len();
                let i = rng.gen_range(0..len);
                let j = rng.gen_range(0..len);
                
                if i != j {
                    perm.swap(i, j);
                }
            } else {
                // Scramble (10% of mutations)
                let len = perm.len();
                if len >= 4 {
                    let i = rng.gen_range(0..len);
                    let j = rng.gen_range(0..len);
                    let (start, end) = if i < j { (i, j) } else { (j, i) };
                    
                    if end - start >= 1 {
                        let mut segment = perm[start..=end].to_vec();
                        segment.shuffle(&mut rng);
                        for (idx, &value) in segment.iter().enumerate() {
                            perm[start + idx] = value;
                        }
                    }
                }
            }
        }
    }

    // OPTIMIZATION: Generate a diverse initial population with mixed strategies
    pub fn generate_population(
        population_size: usize,
        instance: &ProblemInstance,
    ) -> Vec<Solution> {
        let time_window_count = population_size / 10; // 10% time-window based
        let nearest_neighbor_count = population_size / 10; // 10% nearest-neighbor based
        let cluster_count = population_size / 10; // 10% cluster-based
        let random_count = population_size - time_window_count - nearest_neighbor_count - cluster_count;
        
        let mut population = Vec::with_capacity(population_size);
        
        // Add time-window based solutions
        for _ in 0..time_window_count {
            population.push(crate::models::generate_time_window_solution(instance));
        }
        
        // Add nearest-neighbor based solutions
        for _ in 0..nearest_neighbor_count {
            population.push(crate::models::generate_nearest_neighbor_solution(instance));
        }
        
        // Add cluster-based solutions
        for _ in 0..cluster_count {
            population.push(crate::models::generate_cluster_solution(instance));
        }
        
        // Add random solutions
        for _ in 0..random_count {
            population.push(crate::models::generate_valid_solution(instance));
        }
        
        population
    }

    /// Crossover-operatør: flater løsningen, bruker OX1, og "reparer" løsningen ved splitting.
    pub fn crossover(
        parent1: &Solution,
        parent2: &Solution,
        instance: &ProblemInstance,
        crossover_type: usize,
    ) -> Solution {
        let perm1 = parent1.flat.clone();
        let perm2 = parent2.flat.clone();
        
        let child_perm = match crossover_type {
            0 => {
                let (child1, _) = order_one_crossover(&perm1, &perm2);
                child1
            },
            1 => {
                edge_recombination_crossover(&perm1, &perm2)
            },
            2 => {
                let (child1, _) = cycle_crossover(&perm1, &perm2);
                child1
            },
            _ => edge_recombination_crossover(&perm1, &perm2),
        };
        
        let routes = split_permutation_into_routes(&child_perm, instance);
        Solution {
            routes,
            flat: child_perm,
        }
    }

    /// Mutasjon-operatør: flater løsningen, muterer og reparerer.
    pub fn mutate(solution: &Solution, mutation_rate: f64, instance: &ProblemInstance) -> Solution {
        let mut perm = solution.flat.clone();
        
        // Apply various mutations
        apply_mutations(&mut perm, mutation_rate);
        
        let routes = split_permutation_into_routes(&perm, instance);
        Solution { routes, flat: perm }
    }

    /// Calculate fitness for a solution (total travel time + penalty for extra routes)
    pub fn fitness(solution: &Solution, instance: &ProblemInstance) -> f64 {
        let total_travel_time: f64 = solution
            .routes
            .iter()
            .filter_map(|route| crate::models::simulate_route(route, instance).map(|(t, _)| t))
            .sum();
        
        // Apply penalty for exceeding number of nurses
        let penalty_weight = 10000.0;
        let extra_routes = if solution.routes.len() > instance.nbr_nurses {
            solution.routes.len() - instance.nbr_nurses
        } else {
            0
        };
        
        // OPTIMIZATION: Add a small penalty for route imbalance
        let route_durations: Vec<f64> = solution
            .routes
            .iter()
            .filter_map(|route| crate::models::simulate_route(route, instance).map(|(_, d)| d))
            .collect();
            
        let imbalance_penalty = if route_durations.len() >= 2 {
            let avg_duration = route_durations.iter().sum::<f64>() / route_durations.len() as f64;
            let variance = route_durations.iter()
                .map(|&d| (d - avg_duration).powi(2))
                .sum::<f64>() / route_durations.len() as f64;
                
            // Apply small weight to avoid dominating travel time
            variance.sqrt() * 0.1
        } else {
            0.0
        };
        
        // IMPROVEMENT: Add a small penalty for empty routes if we have fewer than required nurses
        let empty_route_penalty = if solution.routes.len() < instance.nbr_nurses {
            (instance.nbr_nurses - solution.routes.len()) as f64 * 5.0
        } else {
            0.0
        };
        
        total_travel_time + penalty_weight * (extra_routes as f64) + imbalance_penalty + empty_route_penalty
    }

    // OPTIMIZED: Improved fitness sharing with clustering to reduce O(n²) complexity
    pub fn compute_shared_fitness(
        population: &[Solution],
        instance: &ProblemInstance,
        sigma_share: f64,
        alpha: f64,
        fitness_cache: &mut HashMap<Vec<usize>, f64>,
    ) -> Vec<(Solution, f64)> {
        // Calculate raw fitness first and cache it
        let raw_fitness_values: Vec<f64> = population
            .par_iter()
            .map(|individual| {
                // Get fitness, either from cache or by calculating it
                match fitness_cache.get(&individual.flat) {
                    Some(&cached_fit) => cached_fit,
                    None => crate::ga::fitness(individual, instance)
                }
            })
            .collect();

        // Update cache sequentially after parallel computation
        for (individual, &fitness) in population.iter().zip(raw_fitness_values.iter()) {
            fitness_cache.insert(individual.flat.clone(), fitness);
        }
        
        // OPTIMIZATION: Cluster solutions to reduce sharing complexity
        // Determine number of clusters based on population size
        let num_clusters = (population.len() as f64).sqrt().ceil() as usize;
        let mut clusters = vec![Vec::new(); num_clusters];
        
        // Simple clustering by taking the first n as centroids
        // and assigning others to nearest centroid
        if population.len() <= num_clusters {
            // If population is small, just put one in each cluster
            for (i, soln) in population.iter().enumerate() {
                clusters[i % num_clusters].push(i);
            }
        } else {
            // Use first num_clusters solutions as representatives
            let representatives: Vec<&Vec<usize>> = 
                population.iter().take(num_clusters).map(|s| &s.flat).collect();
            
            // Assign each solution to a cluster based on nearest representative
            for (i, solution) in population.iter().enumerate() {
                let mut best_dist = usize::MAX;
                let mut best_cluster = 0;
                
                for (j, rep) in representatives.iter().enumerate() {
                    let dist = solution.flat.iter()
                        .zip(rep.iter())
                        .filter(|(a, b)| a != b)
                        .count();
                        
                    if dist < best_dist {
                        best_dist = dist;
                        best_cluster = j;
                    }
                }
                
                clusters[best_cluster].push(i);
            }
        }
        
        // Calculate shared fitness within each cluster
        let mut shared_fitness = vec![0.0; population.len()];
        
        // Process each cluster
        for cluster in &clusters {
            if cluster.is_empty() {
                continue;
            }
            
            // Calculate sharing within this cluster only
            for &i in cluster {
                let raw_fit = raw_fitness_values[i];
                let individual = &population[i];
                
                // Calculate niche count only using members of this cluster
                // This dramatically reduces the O(n²) complexity
                let niche_count: f64 = cluster
                    .iter()
                    .map(|&j| {
                        if i == j {
                            return 0.0; // Don't compare with self
                        }
                        
                        let other = &population[j];
                        // Simplified distance calculation - only count differences
                        let d = individual.flat.iter()
                            .zip(other.flat.iter())
                            .filter(|(a, b)| a != b)
                            .count();
                            
                        if (d as f64) < sigma_share {
                            1.0 - (d as f64) / sigma_share
                        } else {
                            0.0
                        }
                    })
                    .sum();
                
                shared_fitness[i] = raw_fit * (1.0 + alpha * niche_count);
            }
        }
        
        // Return the solutions with their shared fitness
        population.iter().cloned()
            .zip(shared_fitness.into_iter())
            .collect()
    }

    // Enkel turneringsseleksjon basert på delt fitness.
    pub fn tournament_selection(
        shared_pop: &[(Solution, f64)],
        tournament_size: usize,
    ) -> Solution {
        let mut rng = rand::thread_rng();
        let mut best: Option<(Solution, f64)> = None;
        let mut candidates = Vec::with_capacity(tournament_size);
        for _ in 0..tournament_size {
            let candidate = shared_pop.choose(&mut rng).unwrap();
            candidates.push(candidate.clone());
            best = match best {
                None => Some(candidate.clone()),
                Some((_, best_fit)) if candidate.1 < best_fit => Some(candidate.clone()),
                Some(b) => Some(b),
            }
        }
        best.unwrap().0
    }
    
    /// OPTIMIZATION: Calculate population diversity 
    pub fn calculate_population_diversity(population: &[Solution]) -> f64 {
        if population.len() <= 1 {
            return 0.0;
        }
        
        // Sample a subset of pairs for efficiency
        let mut rng = rand::thread_rng();
        let mut sample_count = 0;
        let mut total_diff = 0;
        
        // Target number of pairs to sample (increases with small populations)
        let target_samples = (population.len() * 5).min(100);
        
        for _ in 0..target_samples {
            let i = rng.gen_range(0..population.len());
            let j = rng.gen_range(0..population.len());
            
            if i != j {
                let sol1 = &population[i];
                let sol2 = &population[j];
                
                // Count differences between flat representations
                let diff = sol1.flat.iter()
                    .zip(sol2.flat.iter())
                    .filter(|(a, b)| a != b)
                    .count();
                
                total_diff += diff;
                sample_count += 1;
            }
        }
        
        if sample_count > 0 {
            total_diff as f64 / sample_count as f64
        } else {
            0.0
        }
    }
}

mod local_search {
    use crate::models::{is_route_valid, simulate_route, ProblemInstance, Solution};
    use rand::seq::SliceRandom;
    use rand::Rng;

    /// OPTIMIZATION: Improve pre-check to determine if a route is worth optimizing
    fn is_route_worth_optimizing(route: &Vec<usize>, instance: &ProblemInstance) -> bool {
        if route.len() <= 3 {
            return false; // Too short to optimize efficiently
        }
        
        // For longer routes, check if there are any "obvious" improvements possible
        // by looking for crossovers or long detours
        let check_sample_size = route.len().min(6); // Don't check every point on very long routes
        
        // Sample points to check (checking every point is too expensive)
        let mut rng = rand::thread_rng();
        let mut indices: Vec<usize> = (0..route.len()).collect();
        indices.shuffle(&mut rng);
        
        // Check for potential crossovers (simple case)
        for &i in indices.iter().take(check_sample_size/2) {
            for &j in indices.iter().take(check_sample_size/2) {
                if i + 1 < route.len() && j + 1 < route.len() && i != j && i + 1 != j && j + 1 != i {
                    let i_next = i + 1;
                    let j_next = j + 1;
                    
                    // Get travel times between points
                    let current_cost = instance.travel_times[route[i]][route[i_next]] + 
                                       instance.travel_times[route[j]][route[j_next]];
                    
                    let alternative_cost = instance.travel_times[route[i]][route[j]] + 
                                          instance.travel_times[route[i_next]][route[j_next]];
                    
                    // If we can save 10% by rerouting, it's worth optimizing
                    if alternative_cost < current_cost * 0.9 {
                        return true;
                    }
                }
            }
        }
        
        // If route is long, it's usually worth optimizing anyway
        route.len() >= 8
    }

    /// To‑opt optimalisering innenfor én rute.
    pub fn two_opt(route: &Vec<usize>, instance: &ProblemInstance) -> Vec<usize> {
        let mut best_route = route.clone();
        let (mut best_travel, _) =
            simulate_route(&best_route, instance).unwrap_or((f64::INFINITY, 0.0));
        let n = best_route.len();
        let mut improved = true;
        while improved {
            improved = false;
            for i in 0..n {
                for j in i + 1..n {
                    let mut new_route = best_route.clone();
                    new_route[i..=j].reverse();
                    if let Some((new_travel, _)) = simulate_route(&new_route, instance) {
                        if new_travel < best_travel {
                            best_travel = new_travel;
                            best_route = new_route;
                            improved = true;
                        }
                    }
                }
            }
        }
        best_route
    }

    /// OPTIMIZED: More efficient three-opt optimization
    pub fn three_opt(route: &Vec<usize>, instance: &ProblemInstance) -> Vec<usize> {
        let mut best_route = route.clone();
        let (mut best_travel, _) = simulate_route(&best_route, instance).unwrap_or((f64::INFINITY, 0.0));
        let n = best_route.len();
        
        if n <= 4 {
            return two_opt(route, instance); // Fall back to two-opt for small routes
        }
        
        // Only examine a subset of possible segment combinations
        let mut improved = true;
        let mut iterations = 0;
        
        while improved && iterations < 2 { // Limit iterations
            improved = false;
            iterations += 1;
            
            // First try two-opt as it's faster
            let two_opt_result = two_opt(&best_route, instance);
            if let Some((two_opt_travel, _)) = simulate_route(&two_opt_result, instance) {
                if two_opt_travel < best_travel {
                    best_travel = two_opt_travel;
                    best_route = two_opt_result;
                    improved = true;
                    continue; // Skip three-opt if two-opt improved
                }
            }
            
            // Sample only some i,j,k combinations instead of all
            let sample_size = n.min(8); // Limit sampling
            let mut rng = rand::thread_rng();
            let mut sample_points: Vec<usize> = (0..n).collect();
            sample_points.shuffle(&mut rng);
            let sample_points = &sample_points[0..sample_size];
            
            for &i in sample_points {
                for &j in sample_points {
                    if i >= j || j <= i + 1 { continue; }
                    
                    for &k in sample_points {
                        if k <= j + 1 || k >= n { continue; }
                        
                        // Only try the most promising reconnection configurations
                        for config in [4, 7] { // Try only two of the seven configurations
                            let mut new_route = best_route.clone();
                            match config {
                                4 => { // 3-opt case 1 (often most effective)
                                    let segment1: Vec<usize> = best_route[i+1..=j].to_vec();
                                    let segment2: Vec<usize> = best_route[j+1..=k].to_vec();
                                    new_route.splice(i+1..=k, segment2.into_iter().chain(segment1.into_iter()));
                                },
                                7 => { // 3-opt case 4 (often second most effective)
                                    let mut segment1: Vec<usize> = best_route[i+1..=j].to_vec();
                                    let mut segment2: Vec<usize> = best_route[j+1..=k].to_vec();
                                    segment1.reverse();
                                    segment2.reverse();
                                    new_route.splice(i+1..=k, segment2.into_iter().chain(segment1.into_iter()));
                                },
                                _ => {}
                            }
                            
                            if let Some((new_travel, _)) = simulate_route(&new_route, instance) {
                                if new_travel < best_travel {
                                    best_travel = new_travel;
                                    best_route = new_route;
                                    improved = true;
                                }
                            }
                        }
                    }
                }
            }
        }
        
        best_route
    }

    /// Swap-optimalisering innenfor én rute.
    pub fn swap_optimization(route: &Vec<usize>, instance: &ProblemInstance) -> Vec<usize> {
        let mut best_route = route.clone();
        let mut best_travel = simulate_route(&best_route, instance)
            .map(|(t, _)| t)
            .unwrap_or(f64::INFINITY);
        let n = best_route.len();
        let mut improved = true;
        while improved {
            improved = false;
            for i in 0..n {
                for j in i + 1..n {
                    let mut new_route = best_route.clone();
                    new_route.swap(i, j);
                    if let Some((new_travel, _)) = simulate_route(&new_route, instance) {
                        if new_travel < best_travel {
                            best_travel = new_travel;
                            best_route = new_route;
                            improved = true;
                        }
                    }
                }
            }
        }
        best_route
    }

    /// OPTIMIZED: More selective local search to improve performance
    pub fn selective_local_search(route: &Vec<usize>, instance: &ProblemInstance) -> Vec<usize> {
        if !is_route_worth_optimizing(route, instance) {
            return route.clone(); // Skip optimization for routes unlikely to improve
        }
        
        if route.len() > 7 {
            // For longer routes, only apply two-opt (less expensive)
            let improved_route = two_opt(route, instance);
            swap_optimization(&improved_route, instance)
        } else if route.len() > 3 {
            // Medium routes get three-opt
            let improved_route = three_opt(route, instance);
            swap_optimization(&improved_route, instance)
        } else {
            // Short routes - just return as is
            route.clone()
        }
    }
    
    /// OPTIMIZED: More selective local search with route validity pre-checks
    pub fn apply_local_search(solution: &mut Solution, instance: &ProblemInstance, current_phase: usize) {
        let mut rng = rand::thread_rng();
        
        // Determine local search probability based on current phase
        let ls_probability = match current_phase {
            1 => 0.15,  // Low in exploration phase
            2 => 0.30,  // Medium in balanced phase
            3 => 0.60,  // High in exploitation phase
            _ => 0.30,  // Default
        };
        
        // First, selectively improve individual routes
        let mut improved_routes = Vec::with_capacity(solution.routes.len());
        
        for route in &solution.routes {
            if route.len() <= 1 {
                // Skip trivial routes
                improved_routes.push(route.clone());
                continue;
            }
            
            // Only apply optimization with a certain probability
            // Higher probability for longer routes
            let route_specific_prob = ls_probability * (1.0 + (route.len() as f64 / 10.0).min(1.0));
            if rng.gen::<f64>() < route_specific_prob {
                improved_routes.push(selective_local_search(route, instance));
            } else {
                improved_routes.push(route.clone());
            }
        }
        
        solution.routes = improved_routes;
        
        // Update flat representation
        crate::models::update_flat_from_routes(solution);
        
        // Limit inter-route optimization attempts - they're expensive
        if rng.gen::<f64>() < ls_probability * 0.3 { // Much lower probability for inter-route
            // Try a subset of inter-route improvements
            if solution.routes.len() > 1 {
                if rng.gen_bool(0.5) {
                    relocate_between_routes(solution, instance);
                } else {
                    cross_exchange(solution, instance);
                }
            }
        }
    }
    
    /// OPTIMIZED: More efficient relocation operator with pre-checks
    pub fn relocate_between_routes(solution: &mut Solution, instance: &ProblemInstance) -> bool {
        let num_routes = solution.routes.len();
        if num_routes <= 1 {
            return false;
        }
        
        let mut improved = false;
        let original_cost = crate::ga::fitness(solution, instance);
        let mut best_cost = original_cost;
        let mut best_from_route = 0;
        let mut best_from_pos = 0;
        let mut best_to_route = 0;
        let mut best_to_pos = 0;
        let mut best_patient = 0;
        
        // Try a sample of routes instead of all routes
        let mut rng = rand::thread_rng();
        let sample_size = num_routes.min(4);
        let mut route_indices: Vec<usize> = (0..num_routes).collect();
        route_indices.shuffle(&mut rng);
        
        for &from_route in route_indices.iter().take(sample_size) {
            if solution.routes[from_route].is_empty() {
                continue;
            }
            
            for &to_route in route_indices.iter().take(sample_size) {
                if from_route == to_route {
                    continue;
                }
                
                // Sample patients in the from_route
                let from_route_len = solution.routes[from_route].len();
                let mut from_indices: Vec<usize> = (0..from_route_len).collect();
                from_indices.shuffle(&mut rng);
                
                for &i in from_indices.iter().take(from_route_len.min(3)) {
                    let patient_id = solution.routes[from_route][i];
                    
                    // Sample positions in the to_route
                    let to_route_len = solution.routes[to_route].len();
                    let mut to_positions: Vec<usize> = (0..=to_route_len).collect();
                    to_positions.shuffle(&mut rng);
                    
                    // Sikre at vi ikke prøver å sette inn på en ugyldig posisjon
                    for &j in to_positions.iter().take(to_route_len.min(3) + 1) {
                        if j > solution.routes[to_route].len() {
                            continue;  // Hopp over ugyldige posisjoner
                        }
                        
                        // Klone bare berørte ruter
                        let mut test_from_route = solution.routes[from_route].clone();
                        let mut test_to_route = solution.routes[to_route].clone();
                        
                        // Fjern fra originalruten
                        if i < test_from_route.len() {
                            let patient = test_from_route.remove(i);
                            
                            // Sett inn i destinasjonsruten - sikre at indeksen er gyldig
                            if j <= test_to_route.len() {
                                test_to_route.insert(j, patient);
                                
                                // Sjekk om begge ruter fortsatt er gyldige
                                if is_route_valid(&test_from_route, instance) && 
                                   is_route_valid(&test_to_route, instance) {
                                    
                                    // Lag en full løsning for validering og fitness
                                    let mut test_sol = solution.clone();
                                    test_sol.routes[from_route] = test_from_route;
                                    test_sol.routes[to_route] = test_to_route;
                                    
                                    // Fjern tomme ruter og oppdater flat representasjon
                                    test_sol.routes.retain(|r| !r.is_empty());
                                    crate::models::update_flat_from_routes(&mut test_sol);
                                    
                                    // Sjekk komplett løsningsgyldighet og fitness
                                    let new_cost = crate::ga::fitness(&test_sol, instance);
                                    if new_cost < best_cost {
                                        best_cost = new_cost;
                                        best_from_route = from_route;
                                        best_from_pos = i;
                                        best_to_route = to_route;
                                        best_to_pos = j;
                                        best_patient = patient_id;
                                        improved = true;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        
        // Hvis forbedring funnet, anvend det beste trekket direkte
        if improved {
            // Vi må fjerne pasienten fra den opprinnelige ruten
            if best_from_pos < solution.routes[best_from_route].len() {
                solution.routes[best_from_route].remove(best_from_pos);
                
                // For to_route trenger vi den justerte indeksen hvis from_route < to_route
                // og en tom rute ble fjernet
                let actual_to_route = if best_from_route < best_to_route && 
                                        solution.routes[best_from_route].is_empty() {
                    best_to_route - 1
                } else {
                    best_to_route
                };
                
                // Sikre at actual_to_route er gyldig før innsetting
                if actual_to_route < solution.routes.len() {
                    // Sikre at best_to_pos er gyldig med den nåværende rutelengden
                    let insert_pos = best_to_pos.min(solution.routes[actual_to_route].len());
                    solution.routes[actual_to_route].insert(insert_pos, best_patient);
                } else {
                    // I sjeldne tilfeller hvor actual_to_route er ugyldig, legg til en ny rute
                    solution.routes.push(vec![best_patient]);
                }
                
                // Fjern tomme ruter
                solution.routes.retain(|r| !r.is_empty());
                
                // Oppdater flat representasjon
                crate::models::update_flat_from_routes(solution);
            }
        }
        
        improved
    }

    /// OPTIMIZED: More efficient cross-exchange with sampling
    pub fn cross_exchange(solution: &mut Solution, instance: &ProblemInstance) -> bool {
        let num_routes = solution.routes.len();
        if num_routes <= 1 {
            return false;
        }
        
        let mut improved = false;
        let original_cost = crate::ga::fitness(solution, instance);
        let mut best_cost = original_cost;
        
        // Only try a sample of route pairs
        let mut rng = rand::thread_rng();
        let sample_size = num_routes.min(5);
        let mut route_indices: Vec<usize> = (0..num_routes).collect();
        route_indices.shuffle(&mut rng);
        
        'outer: for &route1_idx in route_indices.iter().take(sample_size) {
            if route1_idx >= solution.routes.len() || solution.routes[route1_idx].len() < 2 {
                continue;
            }
            
            for &route2_idx in route_indices.iter().take(sample_size) {
                if route1_idx == route2_idx || 
                   route2_idx >= solution.routes.len() || 
                   solution.routes[route2_idx].len() < 2 {
                    continue;
                }
                
                // Try exchanging only single segments (not multiple lengths)
                let len1 = 1;
                let len2 = 1;
                
                // Only try sample positions, not all
                let r1_pos_count = solution.routes[route1_idx].len().min(4);
                let r2_pos_count = solution.routes[route2_idx].len().min(4);
                
                // Sikre at vi ikke lager ugyldige ranges
                let max_pos1 = solution.routes[route1_idx].len().saturating_sub(len1);
                let max_pos2 = solution.routes[route2_idx].len().saturating_sub(len2);
                
                if max_pos1 == 0 || max_pos2 == 0 {
                    continue;  // Ikke nok elementer å bytte
                }
                
                let mut r1_positions: Vec<usize> = (0..=max_pos1).collect();
                let mut r2_positions: Vec<usize> = (0..=max_pos2).collect();
                r1_positions.shuffle(&mut rng);
                r2_positions.shuffle(&mut rng);
                
                for &i in r1_positions.iter().take(r1_pos_count.min(r1_positions.len())) {
                    for &j in r2_positions.iter().take(r2_pos_count.min(r2_positions.len())) {
                        // Sjekk gyldighet av i+len1 og j+len2
                        if i + len1 > solution.routes[route1_idx].len() || 
                           j + len2 > solution.routes[route2_idx].len() {
                            continue;
                        }
                        
                        // Create a copy of the solution
                        let mut test_sol = solution.clone();
                        
                        // Extract segments safely
                        let segment1: Vec<usize> = if i + len1 <= test_sol.routes[route1_idx].len() {
                            test_sol.routes[route1_idx].drain(i..i+len1).collect()
                        } else {
                            Vec::new()  // Tom hvis ugyldig
                        };
                        
                        let segment2: Vec<usize> = if j + len2 <= test_sol.routes[route2_idx].len() {
                            test_sol.routes[route2_idx].drain(j..j+len2).collect()
                        } else {
                            Vec::new()  // Tom hvis ugyldig
                        };
                        
                        // Bare forsøke byttet hvis begge segmenter er gyldige
                        if !segment1.is_empty() && !segment2.is_empty() {
                            // Insert segments safely
                            if i <= test_sol.routes[route1_idx].len() {
                                test_sol.routes[route1_idx].splice(i..i, segment2);
                            }
                            
                            if j <= test_sol.routes[route2_idx].len() {
                                test_sol.routes[route2_idx].splice(j..j, segment1);
                            }
                            
                            // Update flat representation
                            crate::models::update_flat_from_routes(&mut test_sol);
                            
                            // Quick check: Are both routes still valid?
                            if route1_idx < test_sol.routes.len() && 
                               route2_idx < test_sol.routes.len() &&
                               is_route_valid(&test_sol.routes[route1_idx], instance) &&
                               is_route_valid(&test_sol.routes[route2_idx], instance) {
                                
                                // Full solution validation only if routes are valid
                                let new_cost = crate::ga::fitness(&test_sol, instance);
                                if new_cost < best_cost {
                                    *solution = test_sol;
                                    best_cost = new_cost;
                                    improved = true;
                                    break 'outer;
                                }
                            }
                        }
                    }
                }
            }
        }
        
        improved
    }
}

mod plotting {
    use crate::models::{ProblemInstance, Solution};
    use plotters::prelude::*;

    /// Plotter den beste ruteplanen med ulike farger for hver rute.
    pub fn plot_solution(
        instance: &ProblemInstance,
        solution: &Solution,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let depot = &instance.depot;
        let mut min_x = depot.x_coord;
        let mut max_x = depot.x_coord;
        let mut min_y = depot.y_coord;
        let mut max_y = depot.y_coord;
        for patient in instance.patients.values() {
            min_x = min_x.min(patient.x_coord);
            max_x = max_x.max(patient.x_coord);
            min_y = min_y.min(patient.y_coord);
            max_y = max_y.max(patient.y_coord);
        }
        let margin = 10.0;
        min_x -= margin;
        max_x += margin;
        min_y -= margin;
        max_y += margin;
        let root = BitMapBackend::new("solution.png", (800, 600)).into_drawing_area();
        root.fill(&WHITE)?;
        let mut chart = ChartBuilder::on(&root)
            .caption("Beste ruteplan", ("sans-serif", 40))
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(40)
            .build_cartesian_2d(min_x..max_x, min_y..max_y)?;
        chart.configure_mesh().draw()?;
        let colors = vec![&RED, &BLUE, &GREEN, &MAGENTA, &CYAN, &YELLOW, &BLACK];
        for (i, route) in solution.routes.iter().enumerate() {
            let color = colors[i % colors.len()];
            let mut points = Vec::new();
            points.push((depot.x_coord, depot.y_coord));
            for &patient_id in route {
                let patient = instance.patients.get(&patient_id.to_string()).unwrap();
                points.push((patient.x_coord, patient.y_coord));
            }
            points.push((depot.x_coord, depot.y_coord));
            chart.draw_series(LineSeries::new(points, color.stroke_width(3)))?;
        }
        chart.draw_series(std::iter::once(Circle::new(
            (depot.x_coord, depot.y_coord),
            5,
            ShapeStyle::from(&BLACK).filled(),
        )))?;
        for patient in instance.patients.values() {
            chart.draw_series(std::iter::once(Circle::new(
                (patient.x_coord, patient.y_coord),
                3,
                ShapeStyle::from(&RGBColor(128, 128, 128)).filled(),
            )))?;
        }
        root.present()?;
        println!("Plot lagret som solution.png");
        Ok(())
    }

    /// Plotter fitnesshistorikk (raw fitness per generasjon).
    pub fn plot_fitness_history(
        history: &Vec<(usize, f64)>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let data = if history.len() > 1 {
            &history[1..]
        } else {
            &history[..]
        };
        let gen_min = data.first().map(|(g, _)| *g).unwrap_or(0);
        let gen_max = data.last().map(|(g, _)| *g).unwrap_or(0);
        let y_min = data
            .iter()
            .map(|(_, fit)| *fit)
            .fold(f64::INFINITY, f64::min);
        let y_max = data
            .iter()
            .map(|(_, fit)| *fit)
            .fold(f64::NEG_INFINITY, f64::max);
        let root = BitMapBackend::new("fitness_history.png", (800, 600)).into_drawing_area();
        root.fill(&WHITE)?;
        let mut chart = ChartBuilder::on(&root)
            .caption("Fitness Historikk", ("sans-serif", 40))
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(60)
            .build_cartesian_2d(gen_min..gen_max, y_min..y_max)?;
        chart.configure_mesh().draw()?;
        chart.draw_series(LineSeries::new(data.to_vec(), &RED))?;
        root.present()?;
        println!("Fitness-historikk plottet som fitness_history.png");
        Ok(())
    }
}

mod island {
    use crate::ga;
    use crate::local_search;
    use crate::models::{ProblemInstance, Solution};
    use rand::seq::SliceRandom;
    use rand::Rng;
    use rayon::prelude::*;
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    /// OPTIMIZATION: Improved island model with additional strategies and performance enhancements
    pub fn run_island_model(
        instance: &ProblemInstance,
        num_islands: usize,
        population_size: usize,
        num_generations: usize,
        migration_interval: usize,
        migration_size: usize,
        initial_mutation_rate: f64,
        mid_mutation_rate: f64,
        final_mutation_rate: f64,
        crossover_rate: f64,
        elitism_count: usize,
        tournament_size: usize,
        sigma_share: f64,
        alpha: f64,
        initial_memetic_rate: f64,
        mid_memetic_rate: f64,
        final_memetic_rate: f64,
        phase1_end: f64,
        phase2_end: f64,
        stagnation_generations: usize,
        restart_keep_percentage: f64,
        time_limit: Option<Duration>,
    ) -> (Solution, f64, Vec<(usize, f64)>) {
        let island_pop_size = population_size / num_islands;
        
        // Generate initial populations with mixed initialization strategies
        let mut islands: Vec<Vec<Solution>> = (0..num_islands)
            .map(|_| ga::generate_population(island_pop_size, instance))
            .collect();
            
        let mut best_solution_overall: Option<Solution> = None;
        let mut best_fitness_overall = f64::INFINITY;
        let mut fitness_history = Vec::new();
        
        // Track stagnation
        let mut generations_without_improvement = 0;
        
        // For time limit tracking
        let start_time = Instant::now();
        let should_stop = Arc::new(AtomicBool::new(false));
        
        // Calculate phase boundaries
        let phase1_end_gen = (num_generations as f64 * phase1_end) as usize;
        let phase2_end_gen = (num_generations as f64 * phase2_end) as usize;
        
        // Parameters for early stopping
        let convergence_threshold = 0.03;  // Increased from 0.02
        let min_generations = 300;        // Reduced from 500
        
        // Add progress tracking
        let min_progress_threshold = 0.00001; // Minimum expected progress
        let progress_check_interval = 1000;
        let mut prev_best_fitness = f64::INFINITY;
        
        // Initialize fitness cache
        let mut fitness_cache = HashMap::new();
        
        // OPTIMIZATION: Track frequency of fitness calculations and cache hits
        let mut fitness_calc_count = 0;
        let mut cache_hit_count = 0;
        
        // The main algorithm - iterate for generations
        for gen in 0..num_generations {
            // Check time limit if set
            if let Some(limit) = time_limit {
                if start_time.elapsed() >= limit {
                    println!("Time limit reached after {} generations", gen);
                    break;
                }
            }
            
            // Check if stopping flag is set
            if should_stop.load(Ordering::Relaxed) {
                println!("Early stopping triggered at generation {}", gen);
                break;
            }
            
            // Periodically clear the fitness cache to prevent memory issues
            // But keep the best solutions' fitness
            if gen % 20 == 0 && fitness_cache.len() > 10000 {
                // Keep only fitness of elite solutions from each island
                let mut elite_flats = Vec::new();
                for island in &islands {
                    for solution in island.iter().take((island_pop_size / 10).max(2)) {
                        elite_flats.push(solution.flat.clone());
                    }
                }
                
                // Create new cache with only elite solutions
                let mut new_cache = HashMap::new();
                for flat in elite_flats {
                    if let Some(&fitness) = fitness_cache.get(&flat) {
                        new_cache.insert(flat, fitness);
                    }
                }
                
                fitness_cache = new_cache;
            }
            
            // Determine the current phase and set appropriate parameters
            let (current_phase, current_mutation_rate, current_memetic_rate) = 
                if gen < phase1_end_gen {
                    // Phase 1: High exploration
                    (1, initial_mutation_rate, initial_memetic_rate)
                } else if gen < phase2_end_gen {
                    // Phase 2: Balanced exploration/exploitation
                    (2, mid_mutation_rate, mid_memetic_rate)
                } else {
                    // Phase 3: High exploitation
                    (3, final_mutation_rate, final_memetic_rate)
                };
            
            // OPTIMIZATION: Process islands in parallel
            // FIX: Use local fitness caches to avoid mutable borrow issues
            let processed_islands: Vec<(Vec<(Solution, f64)>, HashMap<Vec<usize>, f64>, usize, usize)> = islands
                .par_iter()
                .map(|island| {
                    // Create a local fitness cache for this thread
                    let mut local_fitness_cache = HashMap::new();
                    let mut local_calc_count = 0;
                    let mut local_hit_count = 0;
                    
                    // First copy over known fitness values from the global cache
                    for individual in island {
                        if let Some(&fitness) = fitness_cache.get(&individual.flat) {
                            local_fitness_cache.insert(individual.flat.clone(), fitness);
                        }
                    }
                    
                    // Calculate any missing fitness values
                    for individual in island {
                        if !local_fitness_cache.contains_key(&individual.flat) {
                            let new_fitness = ga::fitness(individual, instance);
                            local_fitness_cache.insert(individual.flat.clone(), new_fitness);
                            local_calc_count += 1;
                        } else {
                            local_hit_count += 1;
                        }
                    }
                    
                    // Compute shared fitness with clustering to avoid O(n²) complexity
                    let shared_pop = ga::compute_shared_fitness(
                        island, instance, sigma_share, alpha, 
                        &mut local_fitness_cache
                    );
                    
                    // Return both the shared population and the local cache
                    (shared_pop, local_fitness_cache, local_calc_count, local_hit_count)
                })
                .collect();
            
            // Update the global fitness cache with values from local caches
            let mut thread_calc_count = 0;
            let mut thread_hit_count = 0;
            
            // Create new populations with genetic operations
            let mut new_islands = Vec::with_capacity(islands.len());
            
            for (i, (shared_pop, local_cache, local_calc, local_hit)) in processed_islands.iter().enumerate() {
                // Merge local cache into global cache
                for (k, v) in local_cache {
                    fitness_cache.insert(k.clone(), *v);
                }
                
                // Update statistics
                thread_calc_count += local_calc;
                thread_hit_count += local_hit;
                
                let island = &islands[i];
                
                // Sort by shared fitness
                let mut sorted_shared = shared_pop.clone();
                sorted_shared.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                
                // Check if we found a better solution
                if !sorted_shared.is_empty() {
                    let best_in_island = &sorted_shared[0].0;
                    let best_in_island_raw_fit = fitness_cache.get(&best_in_island.flat)
                                                .copied()
                                                .unwrap_or_else(|| ga::fitness(best_in_island, instance));
                    
                    if best_in_island_raw_fit < best_fitness_overall {
                        best_fitness_overall = best_in_island_raw_fit;
                        best_solution_overall = Some(best_in_island.clone());
                        generations_without_improvement = 0;
                        
                        // Record fitness progress more frequently when we improve
                        fitness_history.push((gen, best_fitness_overall));
                    }
                }
                
                // Log fitness periodically
                if gen % 250 == 0 || gen == num_generations - 1 {
                    if !sorted_shared.is_empty() {
                        let raw = fitness_cache.get(&sorted_shared[0].0.flat)
                                .copied()
                                .unwrap_or_else(|| ga::fitness(&sorted_shared[0].0, instance));
                                
                        // Only add to history if we haven't already done so above
                        if fitness_history.is_empty() || fitness_history.last().unwrap().0 != gen {
                            fitness_history.push((gen, raw));
                        }
                        
                        // Calculate diversity for this island
                        let diversity = ga::calculate_population_diversity(island);
                        
                        println!(
                            "Gen {}: Best fitness (island{}) = {:.2}, diversity = {:.2}, phase = {}",
                            gen, i, raw, diversity, current_phase
                        );
                    }
                }
                
                // Keep elites
                let mut new_population: Vec<Solution> = sorted_shared
                    .iter()
                    .take(elitism_count)
                    .map(|(ind, _)| ind.clone())
                    .collect();
                
                // Apply local search to elites with adaptive probability
                if current_memetic_rate > 0.0 {
                    for solution in &mut new_population {
                        local_search::apply_local_search(solution, instance, current_phase);
                    }
                }
                
                // Create new offspring via crossover and mutation
                let mut rng = rand::thread_rng();
                
                // OPTIMIZATION: Use fast parent selection for large populations
                let tournament_candidates: Vec<&(Solution, f64)> = if shared_pop.len() > 50 {
                    // For large populations, pre-select a subset to speed up tournament selection
                    shared_pop.choose_multiple(&mut rng, shared_pop.len() / 2).collect()
                } else {
                    // For small populations, use all individuals
                    shared_pop.iter().collect()
                };
                
                while new_population.len() < island_pop_size {
                    // OPTIMIZATION: Faster tournament selection from pre-selected candidates
                    let parent1_idx = rng.gen_range(0..tournament_candidates.len());
                    let mut best_idx = parent1_idx;
                    let mut best_fitness = tournament_candidates[parent1_idx].1;
                    
                    for _ in 1..tournament_size {
                        let idx = rng.gen_range(0..tournament_candidates.len());
                        if tournament_candidates[idx].1 < best_fitness {
                            best_idx = idx;
                            best_fitness = tournament_candidates[idx].1;
                        }
                    }
                    let parent1 = &tournament_candidates[best_idx].0;
                    
                    // Select second parent
                    let parent2_idx = rng.gen_range(0..tournament_candidates.len());
                    let mut best_idx = parent2_idx;
                    let mut best_fitness = tournament_candidates[parent2_idx].1;
                    
                    for _ in 1..tournament_size {
                        let idx = rng.gen_range(0..tournament_candidates.len());
                        if tournament_candidates[idx].1 < best_fitness {
                            best_idx = idx;
                            best_fitness = tournament_candidates[idx].1;
                        }
                    }
                    let parent2 = &tournament_candidates[best_idx].0;
                    
                    // Use adaptive crossover type based on phase
                    let mut child = if rng.gen::<f64>() < crossover_rate {
                        let crossover_type = match current_phase {
                            1 => 0, // OX1 for exploration
                            2 => rng.gen_range(0..3), // Random in middle phase
                            3 => 1, // Edge recombination for exploitation
                            _ => rng.gen_range(0..3),
                        };
                        
                        ga::crossover(parent1, parent2, instance, crossover_type)
                    } else {
                        parent1.clone()
                    };
                    
                    // Apply mutation with adaptive rate
                    // Increase mutation for stagnation
                    let adaptive_mutation_rate = if generations_without_improvement > 100 {
                        current_mutation_rate * 1.5
                    } else {
                        current_mutation_rate
                    };
                    
                    child = ga::mutate(&child, adaptive_mutation_rate, instance);
                    
                    // Apply local search with low probability
                    if rng.gen::<f64>() < current_memetic_rate * 0.2 {
                        local_search::apply_local_search(&mut child, instance, current_phase);
                    }
                    
                    new_population.push(child);
                }
                
                new_islands.push(new_population);
            }
            
            // Update fitness calculation counters
            fitness_calc_count += thread_calc_count;
            cache_hit_count += thread_hit_count;
            
            // Update islands
            islands = new_islands;
            
            // Handle stagnation by tracking generations without improvement
            if best_solution_overall.is_some() {
                generations_without_improvement += 1;
            }
            
            // Perform migration between islands at intervals
            if gen > 0 && gen % migration_interval == 0 {
                // OPTIMIZATION: Adaptive migration strategy based on phase
                let mut rng = rand::thread_rng(); // Flytt denne linjen hit
                let migration_topology = match current_phase {
                    1 => 0, // Random in exploration phase
                    2 => 1, // Ring in balanced phase
                    3 => 2, // All-to-all in exploitation phase
                    _ => rng.gen_range(0..3),
                };
                
                // Use different topologies for migration
                match migration_topology {
                    0 => {
                        // Random topology
                        for i in 0..num_islands {
                            let mut next_island = rng.gen_range(0..num_islands);
                            while next_island == i {
                                next_island = rng.gen_range(0..num_islands);
                            }
                            perform_migration(&mut islands, i, next_island, migration_size, instance, &mut fitness_cache);
                        }
                    },
                    1 => {
                        // Ring topology (each island sends to next)
                        for i in 0..num_islands {
                            let next_island = (i + 1) % num_islands;
                            perform_migration(&mut islands, i, next_island, migration_size, instance, &mut fitness_cache);
                        }
                    },
                    2 => {
                        // All-to-all topology (for maximum diversity exchange)
                        for i in 0..num_islands {
                            for j in 0..num_islands {
                                if i != j {
                                    // Use fewer migrants per connection in this dense topology
                                    let migrants_per_link = migration_size / (num_islands - 1).max(1);
                                    if migrants_per_link > 0 {
                                        perform_migration(&mut islands, i, j, migrants_per_link, instance, &mut fitness_cache);
                                    }
                                }
                            }
                        }
                    },
                    _ => {}
                }
            }
            
            // Check for stagnation and perform restart if needed
            if stagnation_generations > 0 && generations_without_improvement >= stagnation_generations {
                println!("Stagnation detected after {} generations without improvement. Performing partial restart.", 
                    generations_without_improvement);
                
                generations_without_improvement = 0;
                
                // OPTIMIZATION: Dynamic restart strategy based on phase
                // Keep fewer solutions in later phases for more disruption
                let restart_keep = match current_phase {
                    1 => restart_keep_percentage,
                    2 => restart_keep_percentage * 0.7,
                    3 => restart_keep_percentage * 0.5,
                    _ => restart_keep_percentage,
                };
                
                // Perform restart for each island
                for island in &mut islands {
                    // Sort island by fitness
                    island.sort_by(|a, b| {
                        let fit_a = fitness_cache.get(&a.flat)
                                    .copied()
                                    .unwrap_or_else(|| ga::fitness(a, instance));
                        let fit_b = fitness_cache.get(&b.flat)
                                    .copied()
                                    .unwrap_or_else(|| ga::fitness(b, instance));
                        fit_a.partial_cmp(&fit_b).unwrap()
                    });
                    
                    // Keep top percentage
                    let keep_count = (island.len() as f64 * restart_keep).ceil() as usize;
                    let kept_solutions: Vec<Solution> = island.iter()
                        .take(keep_count)
                        .cloned()
                        .collect();
                    
                    // Generate new solutions with more diversity
                    let mut new_solutions = Vec::with_capacity(island.len() - keep_count);
                    
                    // Always include time-window based solutions for stability
                    new_solutions.push(crate::models::generate_time_window_solution(instance));
                    
                    // Include nearest-neighbor solution for quality
                    new_solutions.push(crate::models::generate_nearest_neighbor_solution(instance));
                    
                    // Add some completely random solutions for diversity
                    let random_count = (island.len() - keep_count - 2) / 3;
                    for _ in 0..random_count {
                        new_solutions.push(crate::models::generate_valid_solution(instance));
                    }
                    
                    // Add some solutions based on clustering
                    let cluster_count = (island.len() - keep_count - 2 - random_count) / 2;
                    for _ in 0..cluster_count {
                        new_solutions.push(crate::models::generate_cluster_solution(instance));
                    }
                    
                    // Add mutated versions of elite solutions
                    let mutation_count = island.len() - keep_count - 2 - random_count - cluster_count;
                    for i in 0..mutation_count {
                        // Use different mutation rates for variety
                        let mutation_rate = 0.1 + (i as f64 / mutation_count as f64) * 0.3;
                        let base_idx = i % kept_solutions.len();
                        let mutated = ga::mutate(&kept_solutions[base_idx], mutation_rate, instance);
                        new_solutions.push(mutated);
                    }
                    
                    // Combine kept solutions with new ones
                    island.clear();
                    island.extend(kept_solutions);
                    island.append(&mut new_solutions);
                }
                
                // Clear fitness cache after restart since we've modified many solutions
                fitness_cache.clear();
            }
            
            // Check for early stopping if benchmark is available
            if gen > min_generations && instance.benchmark > 0.0 {
                // Check for relative difference from benchmark
                let rel_diff = (best_fitness_overall - instance.benchmark).abs() / instance.benchmark;
                
                // Stop if we are near benchmark
                if rel_diff < convergence_threshold {
                    println!("Near optimal solution found (within {:.1}% of benchmark). Stopping early at generation {}.", 
                             convergence_threshold * 100.0, gen);
                    should_stop.store(true, Ordering::Relaxed);
                }
            }
            
            // Check for minimal progress over time
            if gen > min_generations && gen % progress_check_interval == 0 {
                let relative_progress = if prev_best_fitness != f64::INFINITY {
                    (prev_best_fitness - best_fitness_overall) / prev_best_fitness
                } else {
                    1.0 // First check, assume progress
                };
                
                if relative_progress < min_progress_threshold {
                    println!("Minimal progress detected ({:.3}%). Early stopping at generation {}.", 
                            relative_progress * 100.0, gen);
                    should_stop.store(true, Ordering::Relaxed);
                }
                
                prev_best_fitness = best_fitness_overall;
            }
        }
        
        // Final improvement: Apply intensive local search to best solution
        if let Some(ref mut solution) = best_solution_overall {
            println!("Applying final optimization to best solution...");
            
            // Apply intra-route optimization to each route
            for route in &mut solution.routes {
                if route.len() > 1 {
                    *route = local_search::three_opt(route, instance);
                    *route = local_search::swap_optimization(route, instance);
                }
            }
            
            // Try inter-route optimization a few times
            let mut improved = true;
            let mut iterations = 0;
            while improved && iterations < 5 {
                improved = false;
                if local_search::cross_exchange(solution, instance) {
                    improved = true;
                }
                if local_search::relocate_between_routes(solution, instance) {
                    improved = true;
                }
                iterations += 1;
            }
            
            // Update flat representation
            crate::models::update_flat_from_routes(solution);
            
            // Update best fitness
            best_fitness_overall = ga::fitness(solution, instance);
            
            // Add final entry to fitness history
            fitness_history.push((num_generations, best_fitness_overall));
        }
        
        // Print cache statistics
        let total_fitness_evals = fitness_calc_count + cache_hit_count;
        if total_fitness_evals > 0 {
            let cache_hit_rate = (cache_hit_count as f64 / total_fitness_evals as f64) * 100.0;
            println!("Fitness cache hit rate: {:.1}% ({} hits out of {} lookups)", 
                    cache_hit_rate, cache_hit_count, total_fitness_evals);
        }
        
        (
            best_solution_overall.unwrap_or_else(|| islands[0][0].clone()),
            best_fitness_overall,
            fitness_history,
        )
    }
    
    /// OPTIMIZED: More efficient migration with better selection strategy
    fn perform_migration(
        islands: &mut [Vec<Solution>],
        from_island: usize,
        to_island: usize,
        migration_size: usize,
        instance: &ProblemInstance,
        fitness_cache: &mut HashMap<Vec<usize>, f64>
    ) {
        let mut rng = rand::thread_rng();
        
        // Hopp over hvis en av øyene er tomme
        if islands[from_island].is_empty() || islands[to_island].is_empty() {
            return;
        }
        
        // Få tak i fitness for kilde-øyas løsninger
        let mut source_with_fitness: Vec<(usize, f64)> = islands[from_island]
            .iter()
            .enumerate()
            .map(|(idx, sol)| {
                let fitness = *fitness_cache.entry(sol.flat.clone())
                    .or_insert_with(|| ga::fitness(sol, instance));
                (idx, fitness)
            })
            .collect();
        
        // Sorter etter fitness (beste først)
        source_with_fitness.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        
        // Begrens migrasjonssize til tilgjengelige løsninger
        let actual_migration_size = migration_size.min(source_with_fitness.len());
        
        // Velg migranter (beste løsninger)
        let migrant_indices: Vec<usize> = source_with_fitness
            .iter()
            .take(actual_migration_size)
            .map(|(idx, _)| *idx)
            .collect();
        
        // Få tak i fitness for destinasjons-øyas løsninger
        let mut dest_with_fitness: Vec<(usize, f64)> = islands[to_island]
            .iter()
            .enumerate()
            .map(|(idx, sol)| {
                let fitness = *fitness_cache.entry(sol.flat.clone())
                    .or_insert_with(|| ga::fitness(sol, instance));
                (idx, fitness)
            })
            .collect();
        
        // Sorter etter fitness (verste først)
        dest_with_fitness.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        // Begrens erstatninger til faktisk tilgjengelige indekser
        let mut dest_indices_to_replace = Vec::new();
        let mut replaced_indices = std::collections::HashSet::new();
        
        // Gruppe etter likhet
        let mut similarity_groups = std::collections::HashMap::new();
        
        // Først, ta en del av de verste løsningene som kandidater
        let worst_count = (migration_size * 2).min(dest_with_fitness.len());
        let worst_candidates = dest_with_fitness.iter()
            .take(worst_count)
            .map(|(idx, _)| *idx)
            .collect::<Vec<_>>();
        
        // Grupper etter likhet
        for &idx in &worst_candidates {
            let solution = &islands[to_island][idx];
            let hash_key = solution.flat.iter().take(5).fold(0, |acc, &x| acc * 10 + x as usize % 10);
            similarity_groups.entry(hash_key).or_insert_with(Vec::new).push(idx);
        }
        
        // Ta en fra hver likhetsgruppe, prioriter verste fitness
        for group in similarity_groups.values() {
            if group.len() > 1 && dest_indices_to_replace.len() < actual_migration_size {
                // Finn verste i gruppen
                let worst_idx = group.iter()
                    .max_by(|&&a, &&b| {
                        let fitness_a = fitness_cache.get(&islands[to_island][a].flat)
                                        .copied()
                                        .unwrap_or_else(|| crate::ga::fitness(&islands[to_island][a], instance));
                        let fitness_b = fitness_cache.get(&islands[to_island][b].flat)
                                        .copied()
                                        .unwrap_or_else(|| crate::ga::fitness(&islands[to_island][b], instance));
                        fitness_a.partial_cmp(&fitness_b).unwrap()
                    })
                    .copied()
                    .unwrap();
                
                dest_indices_to_replace.push(worst_idx);
                replaced_indices.insert(worst_idx);
            }
        }
        
        // Fyll resten med verste løsninger som ikke er erstattet ennå
        for (idx, _) in dest_with_fitness {
            if !replaced_indices.contains(&idx) && 
               idx < islands[to_island].len() && // Ekstra sjekk for gyldighet
               dest_indices_to_replace.len() < actual_migration_size {
                dest_indices_to_replace.push(idx);
                replaced_indices.insert(idx);
            }
        }
        
        // Utfør migrasjon med minimal kloning
        for (i, &migrant_idx) in migrant_indices.iter().enumerate() {
            if i < dest_indices_to_replace.len() {
                let dest_idx = dest_indices_to_replace[i];
                
                // Sikkerhetsjekk for å unngå index-out-of-bounds
                if migrant_idx < islands[from_island].len() && dest_idx < islands[to_island].len() {
                    // Klone bare når nødvendig
                    islands[to_island][dest_idx] = islands[from_island][migrant_idx].clone();
                } else if migrant_idx < islands[from_island].len() {
                    // Hvis dest_idx er ugyldig men migrant_idx er gyldig, legg til i stedet
                    islands[to_island].push(islands[from_island][migrant_idx].clone());
                }
                // Hvis begge indeksene er ugyldige, ikke gjør noe (unngå panikk)
            }
        }
    }
}

// Legg til main-funksjon nederst i filen, utenfor andre moduler
fn main() {
    println!("Velkommen til Hjemmesykepleie-optimalisering");
    
    // Last inn instansen fra fil
    let args: Vec<String> = std::env::args().collect();
    let filename = args.get(1).unwrap_or(&String::from("train/train_4.json")).clone();
    
    println!("Laster instans fra: {}", filename);
    let file = File::open(filename).expect("Kunne ikke åpne instansfil");
    let reader = BufReader::new(file);
    let instance: models::ProblemInstance = serde_json::from_reader(reader).expect("Kunne ikke lese instans");
    
    println!("Instans: {}, {} pasienter, {} sykepleiere", 
        instance.instance_name, instance.patients.len(), instance.nbr_nurses);
    
    // Kjør algoritmen
    let (best_solution, best_fitness, fitness_history) = island::run_island_model(
        &instance,        // Problem instance
        4,                // Number of islands
        2000,              // Population size
        8000,             // Number of generations
        50,               // Migration interval
        5,                // Migration size
        0.1,              // Initial mutation rate
        0.05,             // Mid mutation rate
        0.01,             // Final mutation rate
        0.8,              // Crossover rate
        2,                // Elitism count
        5,                // Tournament size
        10.0,             // Sigma share
        10.0,              // Alpha
        0.1,              // Initial memetic rate
        0.3,              // Mid memetic rate
        0.5,              // Final memetic rate
        0.2,              // Phase 1 end
        0.8,              // Phase 2 end
        3000,              // Stagnation generations
        0.2,              // Restart keep percentage
        None //Some(Duration::from_secs(300)), // Time limit
    );
    
    println!("Beste løsning funnet med fitness: {:.2}", best_fitness);
    println!("Antall ruter: {}", best_solution.routes.len());
    println!("{:?}", best_solution.flat);
    println!("Benchmark: {:.2}", instance.benchmark);
    let abs_diff = (best_fitness - instance.benchmark).abs();
    println!("Absolutt differanse fra benchmark: {:.2}", abs_diff);
    let percent_diff = ((best_fitness - instance.benchmark) / instance.benchmark) * 100.0;
    println!("Relativ feil: {:.2}%", percent_diff);
    
    // Valider løsningen
    let valid = models::validate_solution(&best_solution, &instance);
    println!("Løsningen er {}", if valid { "gyldig" } else { "ugyldig" });
    
    // Lag plot
    match plotting::plot_solution(&instance, &best_solution) {
        Ok(_) => println!("Løsningsplott generert"),
        Err(e) => println!("Kunne ikke generere plott: {}", e),
    }
    
    // Plott fitness-historikk
    match plotting::plot_fitness_history(&fitness_history) {
        Ok(_) => println!("Fitness-historikk plottet"),
        Err(e) => println!("Kunne ikke plotte fitness-historikk: {}", e),
    }
}