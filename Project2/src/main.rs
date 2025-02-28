use serde_json;
use std::fs::File;
use std::io::BufReader; // For deserialisering

mod models {
    use rand::seq::SliceRandom; // Needed for shuffle
    use std::collections::HashMap; // Needed for patients lookup

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
        // Estimer antall ruter basert på permutasjonslengde og sykepleierkapasitet
        let est_num_routes = (permutation.len() + instance.nbr_nurses - 1) / instance.nbr_nurses;
        let mut routes = Vec::with_capacity(est_num_routes);
        let mut current_route = Vec::with_capacity(permutation.len() / est_num_routes + 1);
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
                routes.push(current_route);
                // Start en ny rute med denne pasienten fra depot
                let travel_time = instance.travel_times[depot_index][patient_id];
                let patient = instance.patients.get(&patient_id.to_string()).unwrap();
                let arrival_time = travel_time;
                let start_service_time = arrival_time.max(patient.start_time);
                let finish_time = start_service_time + patient.care_time;
                current_route = Vec::with_capacity(permutation.len() / est_num_routes + 1);
                current_route.push(patient_id);
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

    /// Validerer at løsningen overholder alle constraints.
    // Optimaliser validate_solution ved å forhåndsallokere HashSet
    pub fn validate_solution(solution: &Solution, instance: &ProblemInstance) -> bool {
        let mut visited = std::collections::HashSet::with_capacity(instance.patients.len());
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
    use rayon::prelude::*;
    use std::collections::{HashMap, HashSet};
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

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

    
    /// OPTIMIZATION: Insertion Mutation
    /// Selects and moves a random gene to a new position
    pub fn insertion_mutation(perm: &mut Vec<usize>, mutation_rate: f64) {
        let mut rng = rand::thread_rng();
        if rng.gen::<f64>() < mutation_rate && perm.len() >= 2 {
            let len = perm.len();
            let from = rng.gen_range(0..len);
            let to = rng.gen_range(0..len);
            
            // Only perform if source and destination are different
            if from != to {
                let value = perm.remove(from);
                perm.insert(to, value);
            }
        }
    }

    // Bruk SmallRng også i mutation og crossover operasjoner
    pub fn inversion_mutation(perm: &mut Vec<usize>, mutation_rate: f64) {
        let mut rng = SmallRng::from_entropy();
        if rng.gen::<f64>() < mutation_rate && perm.len() >= 2 {
            let len = perm.len();
            let i = rng.gen_range(0..len);
            let j = rng.gen_range(0..len);
            let (start, end) = if i < j { (i, j) } else { (j, i) };
            perm[start..=end].reverse();
        }
    }
    
    /// OPTIMIZATION: Swap Mutation
    /// Swaps two random positions in the permutation
    pub fn swap_mutation(perm: &mut Vec<usize>, mutation_rate: f64) {
        let mut rng = rand::thread_rng();
        if rng.gen::<f64>() < mutation_rate && perm.len() >= 2 {
            let len = perm.len();
            let i = rng.gen_range(0..len);
            let j = rng.gen_range(0..len);
            
            if i != j {
                perm.swap(i, j);
            }
        }
    }

    /// OPTIMIZATION: Apply multiple mutation operators with different probabilities
    pub fn apply_mutations(solution: &mut Solution, instance: &ProblemInstance, mutation_rate: f64) {
        let mut rng = rand::thread_rng();
        
        // Choose mutation type randomly with different weights
        let mutation_selector = rng.gen::<f64>();
        
        // Give route-focused mutations higher probability
        if mutation_selector < 0.20 {  // 20% chance to attempt a route split
            // Try the route split mutation
            if route_split_mutation(solution, instance, mutation_rate * 1.5) {
                return; // If split successful, we're done
            }
            // If split fails, fall through to other mutations
        }
        
        // Standard mutations (apply to flat representation)
        let mutation_type = rng.gen_range(0..3);
        let perm = &mut solution.flat;
        
        match mutation_type {
            0 => inversion_mutation(perm, mutation_rate),
            1 => insertion_mutation(perm, mutation_rate),
            _ => swap_mutation(perm, mutation_rate),
        }
        
        // Update routes from flat permutation
        solution.routes = crate::models::split_permutation_into_routes(perm, instance);
    }

    /// OPTIMIZATION: Generate a diverse initial population with mixed strategies
    pub fn generate_population(
        population_size: usize,
        instance: &ProblemInstance,
    ) -> Vec<Solution> {
        let time_window_count = population_size / 10; // 10% time-window based
        let nearest_neighbor_count = population_size / 10; // 10% nearest-neighbor based
        let multi_route_count = population_size / 5; // 20% solutions with more routes
        let random_count = population_size - time_window_count - nearest_neighbor_count - multi_route_count;
        
        let mut population = Vec::with_capacity(population_size);
        
        // Add time-window based solutions
        for _ in 0..time_window_count {
            population.push(crate::models::generate_time_window_solution(instance));
        }
        
        // Add nearest-neighbor based solutions
        for _ in 0..nearest_neighbor_count {
            population.push(crate::models::generate_nearest_neighbor_solution(instance));
        }
        
        // Add solutions with intentionally more routes
        for _ in 0..multi_route_count {
            let mut sol = crate::models::generate_valid_solution(instance);
            
            // Apply multiple route splits to create more routes
            for _ in 0..5 {
                route_split_mutation(&mut sol, instance, 0.8);
            }
            
            population.push(sol);
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
        let mut new_solution = solution.clone();
        
        // Apply various mutations
        apply_mutations(&mut new_solution, instance, mutation_rate);
        
        new_solution
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
        let max_nurses = instance.nbr_nurses;
        
        let extra_routes = if solution.routes.len() > max_nurses {
            solution.routes.len() - max_nurses
        } else {
            0
        };
        
        // NEW: Add a small incentive for using more routes (up to the maximum)
        // The coefficient should be small enough not to overwhelm travel time optimization
        let route_usage_incentive = if solution.routes.len() < max_nurses {
            // Small negative term (improves fitness) for using more routes
            -0.5 * (max_nurses - solution.routes.len()) as f64
        } else {
            0.0
        };
        
        // Existing route imbalance penalty
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
        
        total_travel_time + penalty_weight * (extra_routes as f64) + imbalance_penalty + route_usage_incentive
    }

    /// Route split mutation: Tries to split a randomly selected route into two routes
    pub fn route_split_mutation(solution: &mut Solution, instance: &ProblemInstance, mutation_rate: f64) -> bool {
        let mut rng = rand::thread_rng();
        
        // Only apply with the given probability and if we have routes to split
        if rng.gen::<f64>() >= mutation_rate || solution.routes.is_empty() {
            return false;
        }
        
        // Don't try to split if we're already at max routes
        if solution.routes.len() >= instance.nbr_nurses {
            return false;
        }
        
        // Choose a route to split (prefer longer routes)
        let route_weights: Vec<usize> = solution.routes.iter().map(|r| r.len()).collect();
        let total_weight: usize = route_weights.iter().sum();
        
        if total_weight == 0 {
            return false; // No routes with patients
        }
        
        // Select a route with probability proportional to its length
        let mut selected_route_idx = 0;
        let mut cumulative = 0;
        let threshold = rng.gen_range(0..total_weight);
        
        for (idx, &weight) in route_weights.iter().enumerate() {
            cumulative += weight;
            if cumulative > threshold {
                selected_route_idx = idx;
                break;
            }
        }
        
        let route = &solution.routes[selected_route_idx];
        
        // Don't try to split routes that are too short
        if route.len() < 2 {
            return false;
        }
        
        // Try different split points
        let split_point = rng.gen_range(1..route.len());
        
        // Create the two new routes
        let route1 = route[0..split_point].to_vec();
        let route2 = route[split_point..].to_vec();
        
        // Validate both routes
        if crate::models::simulate_route(&route1, instance).is_some() && 
        crate::models::simulate_route(&route2, instance).is_some() {
            
            // Apply the split
            solution.routes[selected_route_idx] = route1;
            solution.routes.push(route2);
            
            // Update the flat representation
            crate::models::update_flat_from_routes(solution);
            
            return true;
        }
        
        false // Split wasn't feasible
    }

    /// Parallellisert beregning av delt fitness med fitness sharing.
    pub fn compute_shared_fitness(
        population: &[Solution],
        instance: &ProblemInstance,
        sigma_share: f64,
        alpha: f64,
    ) -> Vec<(usize, f64)> {
        population
            .par_iter()
            .enumerate()  // Inkluder indeksen i stedet for å klone
            .map(|(idx, individual)| {
                let raw_fit = fitness(individual, instance);
                let niche_count: f64 = population
                    .iter()
                    .map(|other| {
                        let flat1 = &individual.flat;
                        let flat2 = &other.flat;
                        let d = flat1
                            .iter()
                            .zip(flat2.iter())
                            .filter(|(a, b)| a != b)
                            .count();
                        if (d as f64) < sigma_share {
                            1.0 - (d as f64) / sigma_share
                        } else {
                            0.0
                        }
                    })
                    .sum();
                let shared_fit = raw_fit + alpha * niche_count;
                (idx, shared_fit)  // Returner indeks i stedet for klonet løsning
            })
            .collect()
    }

    // Enkel turneringsseleksjon basert på delt fitness.
    // Legg til disse importene


    // Enum for å representere ulike seleksjonsmetoder
    #[derive(Debug, Clone, Copy)]
    pub enum SelectionMethod {
        Tournament,  // Turneringsseleksjon (opprinnelig)
        Ranked,      // Rangbasert seleksjon
        Roulette,    // Ruletthjulseleksjon (fitness-proporsjonal)
        Elitist      // Elitistisk seleksjon
    }

    // Den opprinnelige turneringsseleksjonen (optimalisert)
    pub fn tournament_selection(
        shared_pop: &[(Solution, f64)],
        tournament_size: usize,
    ) -> Solution {
        let mut rng = SmallRng::from_entropy();
        let mut best: Option<(Solution, f64)> = None;
        
        for _ in 0..tournament_size {
            let candidate = shared_pop.choose(&mut rng).unwrap();
            best = match best {
                None => Some(candidate.clone()),
                Some((_, best_fit)) if candidate.1 < best_fit => Some(candidate.clone()),
                Some(b) => Some(b),
            }
        }
        
        best.unwrap().0
    }

    // Rangbasert seleksjon (gir høyere seleksjonssannsynlighet til bedre rangerte individer)
    pub fn ranked_selection(shared_pop: &[(Solution, f64)]) -> Solution {
        let mut rng = SmallRng::from_entropy();
        
        // Sorter etter fitness (lavere er bedre)
        let mut sorted_pop = shared_pop.to_vec();
        sorted_pop.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        
        // Tildel rang (1 er best)
        let n = sorted_pop.len();
        let selection_pressure = 1.5; // Justerbar parameter (1.0-2.0)
        
        // Beregn sannsynligheter basert på rang (bedre individer får høyere sannsynlighet)
        let mut probabilities = Vec::with_capacity(n);
        let mut cum_prob = 0.0;
        
        for i in 0..n {
            let prob = (2.0 - selection_pressure) / n as f64 + 
                    ((2.0 * (i as f64)) * (selection_pressure - 1.0)) / (n * (n - 1)) as f64;
            
            cum_prob += prob;
            probabilities.push(cum_prob);
        }
        
        // Velg basert på kumulativ sannsynlighet
        let r = rng.gen::<f64>();
        for i in 0..n {
            if r <= probabilities[i] {
                return sorted_pop[i].0.clone();
            }
        }
        
        // Fallback: returner det best rangerte individet
        sorted_pop[0].0.clone()
    }

    // Ruletthjulseleksjon (fitness-proporsjonal seleksjon)
    pub fn roulette_selection(shared_pop: &[(Solution, f64)]) -> Solution {
        let mut rng = SmallRng::from_entropy();
        
        // Finn minimums- og maksimumsfitness
        let min_fitness = shared_pop.iter()
            .map(|(_, fit)| *fit)
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0);
        
        let max_fitness = shared_pop.iter()
            .map(|(_, fit)| *fit)
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(1.0);
        
        // Konverter fitness til seleksjonssannsynlighet (mindre fitness = høyere sannsynlighet)
        // Vi "inverterer" fitness-verdien siden dette er et minimeringsproblem
        let total_inverted_fitness: f64 = shared_pop.iter()
            .map(|(_, fit)| max_fitness + min_fitness - *fit)
            .sum();
        
        if total_inverted_fitness <= 0.0 {
            // Fallback ved 0 eller negativ total inverted fitness
            let idx = rng.gen_range(0..shared_pop.len());
            return shared_pop[idx].0.clone();
        }
        
        // Generer tilfeldig punkt for seleksjon
        let r = rng.gen::<f64>() * total_inverted_fitness;
        
        // Finn hvilket individ som ble valgt
        let mut cum_prob = 0.0;
        for (individual, fit) in shared_pop {
            cum_prob += max_fitness + min_fitness - fit;
            if cum_prob >= r {
                return individual.clone();
            }
        }
        
        // Fallback: returner siste individ
        shared_pop.last().unwrap().0.clone()
    }

    // Elitistisk seleksjon (velger fra topp 20% med høy sannsynlighet)
    pub fn elitist_selection(shared_pop: &[(Solution, f64)]) -> Solution {
        let mut rng = SmallRng::from_entropy();
        
        // Sorter etter fitness (lavere er bedre)
        let mut sorted_pop = shared_pop.to_vec();
        sorted_pop.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        
        // Velg blant de beste 20% med 80% sannsynlighet
        let elite_count = (sorted_pop.len() as f64 * 0.2).ceil() as usize;
        let elite_count = elite_count.max(1).min(sorted_pop.len());
        
        if rng.gen::<f64>() < 0.8 {
            // Velg fra eliten
            let idx = rng.gen_range(0..elite_count);
            sorted_pop[idx].0.clone()
        } else {
            // Velg fra resten av populasjonen
            let remaining = sorted_pop.len() - elite_count;
            if remaining > 0 {
                let idx = elite_count + rng.gen_range(0..remaining);
                sorted_pop[idx].0.clone()
            } else {
                // Fallback hvis det ikke er noen ikke-elite individer
                sorted_pop[0].0.clone()
            }
        }
    }

    // Samlet funksjon for å utføre seleksjon basert på den valgte metoden
    /// Hjelpefunksjon for å utføre seleksjon basert på indekser i stedet for løsninger
    pub fn perform_selection_idx(
        shared_pop: &[(usize, f64)],
        method: SelectionMethod,
        tournament_size: usize,
    ) -> usize {
        match method {
            SelectionMethod::Tournament => tournament_selection_idx(shared_pop, tournament_size),
            SelectionMethod::Ranked => ranked_selection_idx(shared_pop),
            SelectionMethod::Roulette => roulette_selection_idx(shared_pop),
            SelectionMethod::Elitist => elitist_selection_idx(shared_pop),
        }
    }

    /// Turneringsseleksjon som returnerer indeks i stedet for løsning
    pub fn tournament_selection_idx(
        shared_pop: &[(usize, f64)],
        tournament_size: usize,
    ) -> usize {
        let mut rng = SmallRng::from_entropy();
        let mut best_idx = None;
        let mut best_fit = f64::INFINITY;
        
        for _ in 0..tournament_size {
            let idx = rng.gen_range(0..shared_pop.len());
            let candidate = &shared_pop[idx];
            
            if best_idx.is_none() || candidate.1 < best_fit {
                best_idx = Some(candidate.0);
                best_fit = candidate.1;
            }
        }
        
        best_idx.unwrap()
    }

    /// Rangbasert seleksjon som returnerer indeks
    pub fn ranked_selection_idx(shared_pop: &[(usize, f64)]) -> usize {
        let mut rng = SmallRng::from_entropy();
        
        // Sorter etter fitness (lavere er bedre)
        let mut sorted_pop = shared_pop.to_vec();
        sorted_pop.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        
        let n = sorted_pop.len();
        let selection_pressure = 1.5;
        
        let mut probabilities = Vec::with_capacity(n);
        let mut cum_prob = 0.0;
        
        for i in 0..n {
            let prob = (2.0 - selection_pressure) / n as f64 + 
                    ((2.0 * (i as f64)) * (selection_pressure - 1.0)) / (n * (n - 1)) as f64;
            
            cum_prob += prob;
            probabilities.push(cum_prob);
        }
        
        let r = rng.gen::<f64>();
        for i in 0..n {
            if r <= probabilities[i] {
                return sorted_pop[i].0;
            }
        }
        
        sorted_pop[0].0
    }

    /// Ruletthjulseleksjon som returnerer indeks
    pub fn roulette_selection_idx(shared_pop: &[(usize, f64)]) -> usize {
        let mut rng = SmallRng::from_entropy();
        
        let min_fitness = shared_pop.iter()
            .map(|(_, fit)| *fit)
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0);
        
        let max_fitness = shared_pop.iter()
            .map(|(_, fit)| *fit)
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(1.0);
        
        let total_inverted_fitness: f64 = shared_pop.iter()
            .map(|(_, fit)| max_fitness + min_fitness - *fit)
            .sum();
        
        if total_inverted_fitness <= 0.0 {
            let idx = rng.gen_range(0..shared_pop.len());
            return shared_pop[idx].0;
        }
        
        let r = rng.gen::<f64>() * total_inverted_fitness;
        
        let mut cum_prob = 0.0;
        for (individual_idx, fit) in shared_pop {
            cum_prob += max_fitness + min_fitness - fit;
            if cum_prob >= r {
                return *individual_idx;
            }
        }
        
        shared_pop.last().unwrap().0
    }

    /// Elitistisk seleksjon som returnerer indeks
    pub fn elitist_selection_idx(shared_pop: &[(usize, f64)]) -> usize {
        let mut rng = SmallRng::from_entropy();
        
        let mut sorted_pop = shared_pop.to_vec();
        sorted_pop.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        
        let elite_count = (sorted_pop.len() as f64 * 0.2).ceil() as usize;
        let elite_count = elite_count.max(1).min(sorted_pop.len());
        
        if rng.gen::<f64>() < 0.8 {
            let idx = rng.gen_range(0..elite_count);
            sorted_pop[idx].0
        } else {
            let remaining = sorted_pop.len() - elite_count;
            if remaining > 0 {
                let idx = elite_count + rng.gen_range(0..remaining);
                sorted_pop[idx].0
            } else {
                sorted_pop[0].0
            }
        }
    }


    pub fn calculate_fitness_diversity(population: &[Solution], instance: &ProblemInstance) -> f64 {
        if population.len() <= 1 {
            return 0.0;
        }
        
        let fitness_values: Vec<f64> = population.iter()
            .map(|sol| fitness(sol, instance))
            .collect();
        
        let mean = fitness_values.iter().sum::<f64>() / fitness_values.len() as f64;
        let variance = fitness_values.iter()
            .map(|&f| (f - mean).powi(2))
            .sum::<f64>() / fitness_values.len() as f64;
        
        variance.sqrt() / mean  // Coefficient of variation
    }

    /// Fast approximation of population diversity with minimal performance impact
    /// Omfattende diversitetsberegning som kombinerer flere mål
    /// Returnerer en verdi mellom 0.0 (ingen diversitet) og 100.0 (maksimal diversitet)
    pub fn calculate_comprehensive_diversity(
        population: &[Solution], 
        instance: &ProblemInstance
    ) -> (f64, String) {
        if population.len() <= 1 {
            return (0.0, "Population for liten".to_string());
        }
        
        // 1. Genotypisk diversitet - Beregn gjennomsnittlig Hamming-avstand
        let mut total_hamming_distance = 0.0;
        let mut pair_count = 0;
        
        // Sammenlign alle par av løsninger (O(n²), men OK siden dette bare kjøres sjelden)
        for i in 0..population.len() {
            for j in (i+1)..population.len() {
                let sol1 = &population[i];
                let sol2 = &population[j];
                
                // Finn den minste lengden for å unngå indekseringsfeiler
                let min_len = sol1.flat.len().min(sol2.flat.len());
                if min_len == 0 { continue; }
                
                // Beregn Hamming-avstand (antall posisjoner som er forskjellige)
                let mut diff_count = 0;
                for pos in 0..min_len {
                    if sol1.flat[pos] != sol2.flat[pos] {
                        diff_count += 1;
                    }
                }
                
                // Legg til forskjeller for ulik lengde
                let len_diff = (sol1.flat.len() as isize - sol2.flat.len() as isize).abs() as usize;
                diff_count += len_diff;
                
                // Normaliser til prosent
                let hamming_distance = (diff_count as f64 / min_len.max(1) as f64) * 100.0;
                total_hamming_distance += hamming_distance;
                pair_count += 1;
            }
        }
        
        let genotypic_diversity = if pair_count > 0 {
            total_hamming_distance / pair_count as f64
        } else {
            0.0
        };
        
        // 2. Fenotypisk diversitet - Fitness-variasjon
        let fitness_values: Vec<f64> = population.iter()
            .map(|sol| fitness(sol, instance))
            .collect();
        
        let mean_fitness = fitness_values.iter().sum::<f64>() / fitness_values.len() as f64;
        let fitness_variance = fitness_values.iter()
            .map(|&f| (f - mean_fitness).powi(2))
            .sum::<f64>() / fitness_values.len() as f64;
        
        // Koeffisient for variasjon (CV)
        let fitness_cv = if mean_fitness != 0.0 {
            (fitness_variance.sqrt() / mean_fitness) * 100.0
        } else {
            0.0
        };
        
        // 3. Strukturell diversitet - Variasjon i antall ruter
        let route_counts: Vec<usize> = population.iter()
            .map(|sol| sol.routes.len())
            .collect();
        
        let unique_route_counts: std::collections::HashSet<_> = route_counts.iter().cloned().collect();
        let structural_diversity = (unique_route_counts.len() as f64 / population.len() as f64) * 100.0;
        
        // 4. Rutelikhet - Sammenlign om de samme pasientene er i samme rekkefølge
        let mut route_similarity_sum = 0.0;
        let mut route_pair_count = 0;
        
        // Begrens til maks 20 løsninger for å spare tid (tilfeldig utvalg hvis flere)
        let samples = if population.len() > 20 {
            let mut rng = rand::thread_rng();
            let mut indices: Vec<usize> = (0..population.len()).collect();
            indices.shuffle(&mut rng);
            indices.truncate(20);
            indices.iter().map(|&i| &population[i]).collect()
        } else {
            population.iter().collect()
        };
        
        let samples: Vec<&Solution> = samples;
        
        for i in 0..samples.len() {
            for j in (i+1)..samples.len() {
                let sol1 = samples[i];
                let sol2 = samples[j];
                
                // Sammenlign ruter
                let mut matched_routes = 0;
                
                for route1 in &sol1.routes {
                    for route2 in &sol2.routes {
                        if route1.len() != route2.len() { continue; }
                        
                        // Tell antall like elementer
                        let mut matches = 0;
                        for k in 0..route1.len() {
                            if route1[k] == route2[k] {
                                matches += 1;
                            }
                        }
                        
                        // Hvis mer enn 70% samsvar, regn det som en match
                        if matches as f64 / route1.len() as f64 > 0.7 {
                            matched_routes += 1;
                            break; // Gå til neste route1
                        }
                    }
                }
                
                // Beregn rutelikhet (lavere = mer diversitet)
                let similarity = if sol1.routes.len() > 0 && sol2.routes.len() > 0 {
                    matched_routes as f64 / sol1.routes.len().min(sol2.routes.len()) as f64
                } else {
                    0.0
                };
                
                route_similarity_sum += similarity;
                route_pair_count += 1;
            }
        }
        
        let route_diversity = if route_pair_count > 0 {
            (1.0 - route_similarity_sum / route_pair_count as f64) * 100.0
        } else {
            0.0
        };
        
        // Beregn vektet gjennomsnitt av de ulike diversitetsmålene
        let weights = [0.35, 0.25, 0.15, 0.25]; // Justerbare vekter
        let diversity_metrics = [genotypic_diversity, fitness_cv, structural_diversity, route_diversity];
        
        let total_diversity = weights.iter()
            .zip(diversity_metrics.iter())
            .map(|(&w, &m)| w * m)
            .sum::<f64>();
        
        // Lag en detaljert diversitetsrapport
        let detail = format!(
            "Diversitet: {:.1}% (Gen: {:.1}%, Fit: {:.1}%, Str: {:.1}%, Rut: {:.1}%)",
            total_diversity, genotypic_diversity, fitness_cv, structural_diversity, route_diversity
        );
        
        (total_diversity, detail)
    }
}

mod local_search {
    use crate::models::{simulate_route, ProblemInstance, Solution};

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

    /// Kombinerer to‑opt og swap for et utvidet lokalt søk.
    pub fn extended_local_search(route: &Vec<usize>, instance: &ProblemInstance) -> Vec<usize> {
        let route_after_two_opt = two_opt(route, instance);
        swap_optimization(&route_after_two_opt, instance)
    }
    
    /// OPTIMIZATION: Applies local search to a complete solution
    pub fn apply_local_search(solution: &mut Solution, instance: &ProblemInstance) {
        let mut improved_routes = Vec::with_capacity(solution.routes.len());
        
        for route in &solution.routes {
            if route.len() > 1 {
                let improved_route = extended_local_search(route, instance);
                improved_routes.push(improved_route);
            } else {
                improved_routes.push(route.clone());
            }
        }
        
        solution.routes = improved_routes;
        
        // Update flat representation to match the improved routes
        crate::models::update_flat_from_routes(solution);
    }
    
    /// OPTIMIZATION: Relocate operator that moves a patient between routes
    pub fn relocate_between_routes(solution: &mut Solution, instance: &ProblemInstance) -> bool {
        let num_routes = solution.routes.len();
        if num_routes <= 1 {
            return false;
        }
        
        let mut improved = false;
        let mut best_cost = crate::ga::fitness(solution, instance);
        
        // Try all possible patient relocations between routes
        'outer: for from_route in 0..num_routes {
            if solution.routes[from_route].is_empty() {
                continue;
            }
            
            for to_route in 0..num_routes {
                if from_route == to_route {
                    continue;
                }
                
                // Iterate safely by using indices up to the current length
                let from_route_len = solution.routes[from_route].len();
                for i in 0..from_route_len {
                    // Skip if the route has been modified and is now shorter
                    if i >= solution.routes[from_route].len() {
                        continue;
                    }
                    
                    let patient_id = solution.routes[from_route][i];
                    
                    // Try inserting at each position in the destination route
                    for j in 0..=solution.routes[to_route].len() {
                        // Create a copy of the solution
                        let mut test_sol = solution.clone();
                        
                        // Remove from original route
                        let patient = test_sol.routes[from_route].remove(i);
                        
                        // Insert into destination route
                        test_sol.routes[to_route].insert(j, patient);
                        
                        // Remove empty routes
                        test_sol.routes.retain(|r| !r.is_empty());
                        
                        // Update flat representation
                        crate::models::update_flat_from_routes(&mut test_sol);
                        
                        // Check if solution is valid and better
                        if crate::models::validate_solution(&test_sol, instance) {
                            let new_cost = crate::ga::fitness(&test_sol, instance);
                            if new_cost < best_cost {
                                *solution = test_sol;
                                best_cost = new_cost;
                                improved = true;
                                
                                // Since we modified the solution, we need to break and restart
                                // as the routes structure has changed
                                break 'outer;
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
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    // Modifiser island::run_island_model for å bruke ulike seleksjonsmetoder per øy
    pub fn run_island_model(
        instance: &ProblemInstance,
        num_islands: usize,
        population_size: usize,
        num_generations: usize,
        migration_interval: usize,
        migration_size: usize,
        initial_mutation_rate: f64,
        final_mutation_rate: f64, 
        crossover_rate: f64,
        elitism_count: usize,
        tournament_size: usize,
        sigma_share: f64,
        alpha: f64,
        memetic_rate: f64,
        stagnation_generations: usize,
        time_limit: Option<Duration>,
    ) -> (Solution, f64, Vec<(usize, f64)>) {
        let island_pop_size = population_size / num_islands;
        
        // Tildel ulike seleksjonsmetoder til hver øy
        let selection_methods = [
            ga::SelectionMethod::Tournament,
            ga::SelectionMethod::Ranked,
            ga::SelectionMethod::Roulette,
            ga::SelectionMethod::Elitist
        ];
        
        // Generer initial populasjon for hver øy
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
        
        // OPTIMALISERING: Alloker indeks-vektorer én gang for hver øy
        let mut islands_indices: Vec<Vec<usize>> = (0..num_islands)
            .map(|_| Vec::with_capacity(island_pop_size))
            .collect();
        
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
            
            // Calculate adaptive mutation rate based on progress
            let progress = (gen as f64) / (num_generations as f64);
            let current_mutation_rate = initial_mutation_rate 
                - progress * (initial_mutation_rate - final_mutation_rate);
            
            // Evolve each island
            for island_idx in 0..islands.len() {
                let island = &mut islands[island_idx];
                
                // Velg seleksjonsmetode for denne øya
                let selection_method = selection_methods[island_idx % selection_methods.len()];
                
                // Compute shared fitness for current island (returnerer nå indekser)
                let shared_pop = ga::compute_shared_fitness(island, instance, sigma_share, alpha);
                
                // Gjenbruk indeks-vektoren for denne øya
                let indices = &mut islands_indices[island_idx];
                indices.clear();
                indices.extend(0..shared_pop.len());
                
                // Sort by shared fitness
                indices.sort_by(|&a, &b| shared_pop[a].1.partial_cmp(&shared_pop[b].1).unwrap());
                
                // Best løsning er nå island[shared_pop[indices[0]].0]
                let best_idx = shared_pop[indices[0]].0;
                let best_fitness = shared_pop[indices[0]].1;
                
                if best_fitness < best_fitness_overall {
                    best_fitness_overall = best_fitness;
                    best_solution_overall = Some(island[best_idx].clone());
                    generations_without_improvement = 0;
                } else {
                    generations_without_improvement += 1;
                }
                
                // Log fitness periodically
                if gen % 250 == 0 || gen == num_generations - 1 {
                    let raw = ga::fitness(&island[best_idx], instance);
                    fitness_history.push((gen, raw));
    
                    let (diversity_value, diversity_detail) = ga::calculate_comprehensive_diversity(island, instance);
    
                    println!(
                        "Gen {}: Best shared fitness (island{}, {:?}) = {:.2}, raw = {:.2}",
                        gen, island_idx, selection_method, best_fitness, raw
                    );
                    println!("    {}", diversity_detail);
                }
                
                // Apply local search to a subset of the population (memetic algorithm)
                if memetic_rate > 0.0 {
                    // Kjør memetic-algoritmen mindre hyppig i tidlige generasjoner
                    let should_apply_memetic = gen % (5 + (10.0 * (1.0 - progress)) as usize) == 0;
                    
                    if should_apply_memetic {
                        let adaptive_memetic_rate = memetic_rate * (0.5 + 0.5 * progress);
                        let ls_count = (island_pop_size as f64 * adaptive_memetic_rate).ceil() as usize;
                        
                        for i in 0..ls_count.min(island.len()) {
                            local_search::apply_local_search(&mut island[i], instance);
                            
                            let inter_route_probability = 0.2 + 0.6 * progress;
                            if i < ls_count / 3 || rand::random::<f64>() < inter_route_probability {
                                let iterations = 1 + (2.0 * progress) as usize;
                                for _ in 0..iterations {
                                    local_search::relocate_between_routes(&mut island[i], instance);
                                }
                            }
                        }
                    }
                }
                
                // Keep elites - bruk nå indekser for å hente ut elitene
                let mut new_population: Vec<Solution> = indices
                    .iter()
                    .take(elitism_count)
                    .map(|&i| island[shared_pop[i].0].clone())
                    .collect();
                
                // Create new offspring via crossover and mutation
                let mut rng = rand::thread_rng();
                while new_population.len() < island_pop_size {
                    // Bruk øyas tildelte seleksjonsmetode og indeksene for å velge foreldre
                    let parent1_idx = ga::perform_selection_idx(&shared_pop, selection_method, tournament_size);
                    let parent2_idx = ga::perform_selection_idx(&shared_pop, selection_method, tournament_size);
                    
                    let parent1 = &island[parent1_idx];
                    let parent2 = &island[parent2_idx];
                    
                    let mut child = if rng.gen::<f64>() < crossover_rate {
                        let crossover_type = rng.gen_range(0..2);
                        ga::crossover(parent1, parent2, instance, crossover_type)
                    } else {
                        parent1.clone()
                    };
                    
                    child = ga::mutate(&child, current_mutation_rate, instance);
                    new_population.push(child);
                }
                
                *island = new_population;
            }
            
            // Migration logic...
            if gen > 0 && gen % migration_interval == 0 {
                println!("Performing migration at generation {}", gen);
                for i in 0..num_islands {
                    let to_island = (i + 1) % num_islands;
                    perform_migration(&mut islands, i, to_island, migration_size, instance);
                }
            }
            
            // Early stopping check...
            if generations_without_improvement >= stagnation_generations {
                println!("Early stopping due to {} generations without improvement", stagnation_generations);
                should_stop.store(true, Ordering::Relaxed);
            }
        }
        
        (
            best_solution_overall.unwrap(),
            best_fitness_overall,
            fitness_history,
        )
    }
    
    /// OPTIMIZATION: Improved migration strategy
    fn perform_migration(
        islands: &mut Vec<Vec<Solution>>, 
        from_island: usize, 
        to_island: usize, 
        migration_size: usize,
        instance: &ProblemInstance
    ) {
        // Sort from_island by fitness (best first)
        islands[from_island].sort_by(|a, b| {
            ga::fitness(a, instance)
                .partial_cmp(&ga::fitness(b, instance))
                .unwrap()
        });
        
        // Select migrants (best solutions)
        let migrants: Vec<Solution> = islands[from_island]
            .iter()
            .take(migration_size)
            .cloned()
            .collect();
        
        // Sort to_island by fitness (worst first)
        islands[to_island].sort_by(|a, b| {
            ga::fitness(b, instance)
                .partial_cmp(&ga::fitness(a, instance))
                .unwrap()
        });
        
        // Replace worst solutions with migrants
        for j in 0..migration_size {
            if j < islands[to_island].len() {
                islands[to_island][j] = migrants[j].clone();
            }
        }
    }
}

fn load_instance(filename: &str) -> Result<models::ProblemInstance, Box<dyn std::error::Error>> {
    let file = File::open(filename)?;
    let reader = BufReader::new(file);
    let instance = serde_json::from_reader(reader)?;
    Ok(instance)
}

fn main() {
    // OPTIMIZATION: Improved parameter settings
    let total_population = 2000;
    let num_generations = 4000;
    let initial_mutation_rate = 0.1;
    let final_mutation_rate = 0.05;
    let crossover_rate = 0.7;
    let elitism_count = 4;
    let tournament_size = 5;
    let sigma_share = 20.0;
    let alpha = 0.2;
    
    // Island model parameters
    let num_islands = 4;
    let migration_interval = 400;
    let migration_size = 5;
    
    // Memetic algorithm parameters
    let memetic_rate = 0.15;  // Apply local search to 5% of population each generation
    
    // Restart parameters
    let stagnation_generations = 1000;
    
    // Optional time limit (e.g. 10 minutes)
    let time_limit = None; //Some(std::time::Duration::from_secs(600));

    let filename = "train/train_3.json";
    match load_instance(filename) {
        Ok(instance) => {
            println!("Starting optimization for instance: {}", instance.instance_name);
            println!("Benchmark value: {:.2}", instance.benchmark);
            
            let start_time = std::time::Instant::now();
            
            let (best_solution, best_fit, fitness_history) = island::run_island_model(
                &instance,
                num_islands,
                total_population,
                num_generations,
                migration_interval,
                migration_size,
                initial_mutation_rate,
                final_mutation_rate,
                crossover_rate,
                elitism_count,
                tournament_size,
                sigma_share,
                alpha,
                memetic_rate,
                stagnation_generations,
                time_limit,
            );
            
            let elapsed = start_time.elapsed();
            println!("\nOptimization completed in {:.2} seconds", elapsed.as_secs_f64());
            
            println!("\nBeste løsning funnet med raw fitness {:.2}:", best_fit);
            println!("Benchmark: {:.2}", instance.benchmark);
            let abs_diff = (best_fit - instance.benchmark).abs();
            println!("Absolutt differanse fra benchmark: {:.2}", abs_diff);
            let percent_diff = ((best_fit - instance.benchmark) / instance.benchmark) * 100.0;
            println!("Relativ feil: {:.2}%", percent_diff);
            
            println!("\nStatistics for best solution:");
            println!("Number of routes: {}", best_solution.routes.len());
            println!("Required nurses: {}", instance.nbr_nurses);
            
            // Format solution for output
            println!("\nDetailed solution:");
            println!("Nurse capacity: {}", instance.capacity_nurse);
            println!("Depot return time: {}", instance.depot.return_time);
            println!("{}", "-".repeat(80));
            
            for (i, route) in best_solution.routes.iter().enumerate() {
                let (_travel_time, duration) = models::simulate_route(route, &instance)
                    .unwrap_or((0.0, 0.0));
                    
                let demand: f64 = route.iter()
                    .map(|&pid| instance.patients.get(&pid.to_string()).unwrap().demand)
                    .sum();
                    
                println!("Nurse {}: Route duration {:.2}, Covered demand {:.2}, Patients: {:?}",
                    i+1, duration, demand, route);
            }
            println!("{}", "-".repeat(80));
            println!("Total travel time: {:.2}", best_fit);

            if let Err(e) = plotting::plot_solution(&instance, &best_solution) {
                println!("Feil under plotting av løsning: {}", e);
            }
            if let Err(e) = plotting::plot_fitness_history(&fitness_history) {
                println!("Feil under plotting av fitnesshistorikk: {}", e);
            }
        }
        Err(e) => println!("Feil ved lasting av instans: {}", e),
    }
}