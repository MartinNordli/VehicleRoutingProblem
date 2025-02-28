use rand::seq::SliceRandom; // For shuffle
use serde_json;
use std::collections::HashMap;
use std::fs::File;
use std::io::BufReader; // For deserialisering

use plotters::prelude::*;
use rand::Rng;
use rayon::prelude::*;

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

    // Løsningen representeres eksplisitt som en liste over ruter og en cache for den flate
    // permutasjonen.
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

    /// Konstruerer en gyldig løsning basert på en tilfeldig permutasjon.
    pub fn generate_valid_solution(instance: &ProblemInstance) -> Solution {
        let mut perm: Vec<usize> = (1..=instance.patients.len()).collect();
        perm.shuffle(&mut rand::thread_rng());
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
}

mod ga {
    use crate::models::{split_permutation_into_routes, ProblemInstance, Solution};
    use rand::seq::SliceRandom;
    use rand::Rng;
    use rayon::prelude::*;

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

    /// Inversjonsmutasjon: inverterer en del av permutasjonen med gitt sannsynlighet.
    pub fn inversion_mutation(perm: &mut Vec<usize>, mutation_rate: f64) {
        let mut rng = rand::thread_rng();
        if rng.gen::<f64>() < mutation_rate && perm.len() >= 2 {
            let len = perm.len();
            let i = rng.gen_range(0..len);
            let j = rng.gen_range(0..len);
            let (start, end) = if i < j { (i, j) } else { (j, i) };
            perm[start..=end].reverse();
        }
    }

    /// Genererer en populasjon med gyldige løsninger.
    pub fn generate_population(
        population_size: usize,
        instance: &ProblemInstance,
    ) -> Vec<Solution> {
        (0..population_size)
            .map(|_| crate::models::generate_valid_solution(instance))
            .collect()
    }

    /// Crossover-operatør: flater løsningen, bruker OX1, og "reparer" løsningen ved splitting.
    pub fn crossover(
        parent1: &Solution,
        parent2: &Solution,
        instance: &ProblemInstance,
    ) -> Solution {
        let perm1 = parent1.flat.clone();
        let perm2 = parent2.flat.clone();
        let (child_perm, _child2) = order_one_crossover(&perm1, &perm2);
        let routes = split_permutation_into_routes(&child_perm, instance);
        Solution {
            routes,
            flat: child_perm,
        }
    }

    /// Mutasjon-operatør: flater løsningen, muterer og reparerer.
    pub fn mutate(solution: &Solution, mutation_rate: f64, instance: &ProblemInstance) -> Solution {
        let mut perm = solution.flat.clone();
        inversion_mutation(&mut perm, mutation_rate);
        let routes = split_permutation_into_routes(&perm, instance);
        Solution { routes, flat: perm }
    }

    /// Beregner fitness for en løsning (total reisetid + straff for for mange ruter).
    pub fn fitness(solution: &Solution, instance: &ProblemInstance) -> f64 {
        let total_travel_time: f64 = solution
            .routes
            .iter()
            .filter_map(|route| crate::models::simulate_route(route, instance).map(|(t, _)| t))
            .sum();
        let penalty_weight = 10000.0;
        let extra_routes = if solution.routes.len() > instance.nbr_nurses {
            solution.routes.len() - instance.nbr_nurses
        } else {
            0
        };
        total_travel_time + penalty_weight * (extra_routes as f64)
    }

    /// Parallellisert beregning av delt fitness med fitness sharing.
    pub fn compute_shared_fitness(
        population: &[Solution],
        instance: &ProblemInstance,
        sigma_share: f64,
        alpha: f64,
    ) -> Vec<(Solution, f64)> {
        population
            .par_iter()
            .map(|individual| {
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
                (individual.clone(), shared_fit)
            })
            .collect()
    }

    // Enkel turneringsseleksjon basert på delt fitness.
    pub fn tournament_selection(
        shared_pop: &[(Solution, f64)],
        tournament_size: usize,
    ) -> Solution {
        let mut rng = rand::thread_rng();
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
    use crate::models::{ProblemInstance, Solution};
    use rand::seq::SliceRandom;
    use rand::Rng;
    use rayon::prelude::*;

    /// Kjører islandsmodellen. Populasjonen deles opp i `num_islands` med like store
    /// subpopulasjoner. Hvert 'øy'-individ evolveres i `migration_interval` generasjoner før
    /// migrasjon (bytte av de beste individene) utføres.
    pub fn run_island_model(
        instance: &ProblemInstance,
        num_islands: usize,
        population_size: usize,
        num_generations: usize,
        migration_interval: usize,
        migration_size: usize,
        mutation_rate: f64,
        crossover_rate: f64,
        elitism_count: usize,
        tournament_size: usize,
        sigma_share: f64,
        alpha: f64,
    ) -> (Solution, f64, Vec<(usize, f64)>) {
        let island_pop_size = population_size / num_islands;
        let mut islands: Vec<Vec<Solution>> = (0..num_islands)
            .map(|_| ga::generate_population(island_pop_size, instance))
            .collect();
        let mut best_solution_overall: Option<Solution> = None;
        let mut best_fitness_overall = f64::INFINITY;
        let mut fitness_history = Vec::new();

        for gen in 0..num_generations {
            // Evolver hver øy én generasjon
            for island in &mut islands {
                let shared_pop = ga::compute_shared_fitness(island, instance, sigma_share, alpha);
                let mut sorted_shared = shared_pop.clone();
                sorted_shared.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                if sorted_shared[0].1 < best_fitness_overall {
                    best_fitness_overall = sorted_shared[0].1;
                    best_solution_overall = Some(sorted_shared[0].0.clone());
                }
                if gen % 250 == 0 {
                    let raw = ga::fitness(&sorted_shared[0].0, instance);
                    fitness_history.push((gen, raw));
                    println!(
                        "Generasjon {}: Best delt fitness (island0) = {:.2}, raw fitness = {:.2}",
                        gen, sorted_shared[0].1, raw
                    );
                }
                let mut new_population: Vec<Solution> = sorted_shared
                    .iter()
                    .take(elitism_count)
                    .map(|(ind, _)| ind.clone())
                    .collect();
                while new_population.len() < island_pop_size {
                    let parent1 = ga::tournament_selection(&sorted_shared, tournament_size);
                    let parent2 = ga::tournament_selection(&sorted_shared, tournament_size);
                    let mut child = if rand::random::<f64>() < crossover_rate {
                        ga::crossover(&parent1, &parent2, instance)
                    } else {
                        parent1.clone()
                    };
                    child = ga::mutate(&child, mutation_rate, instance);
                    new_population.push(child);
                }
                *island = new_population;
            }

            // Utfør migrasjon hvert migration_interval generasjoner
            if gen > 0 && gen % migration_interval == 0 {
                for i in 0..num_islands {
                    let next_island = (i + 1) % num_islands;
                    // Sorter øy i etter fitness (laveste først)
                    islands[i].sort_by(|a, b| {
                        let f_a = ga::fitness(a, instance);
                        let f_b = ga::fitness(b, instance);
                        f_a.partial_cmp(&f_b).unwrap()
                    });
                    let migrants: Vec<Solution> =
                        islands[i].iter().take(migration_size).cloned().collect();
                    // Sorter neste øy synkende (dårligste først) og erstatt de dårligste
                    islands[next_island].sort_by(|a, b| {
                        let f_a = ga::fitness(a, instance);
                        let f_b = ga::fitness(b, instance);
                        f_b.partial_cmp(&f_a).unwrap()
                    });
                    for j in 0..migration_size {
                        if j < islands[next_island].len() {
                            islands[next_island][j] = migrants[j].clone();
                        }
                    }
                }
            }
        }
        (
            best_solution_overall.unwrap(),
            best_fitness_overall,
            fitness_history,
        )
    }
}

fn load_instance(filename: &str) -> Result<models::ProblemInstance, Box<dyn std::error::Error>> {
    let file = File::open(filename)?;
    let reader = BufReader::new(file);
    let instance = serde_json::from_reader(reader)?;
    Ok(instance)
}

fn main() {
    // GA-parametere
    let total_population = 2000;
    let num_generations = 8000;
    let mutation_rate = 0.15;
    let crossover_rate = 0.5;
    let elitism_count = 2;
    let tournament_size = 5;
    let sigma_share = 20.0;
    let alpha = 12.5;
    // Islandsmodell-parametere
    let num_islands = 4;
    let migration_interval = 750;
    let migration_size = 5;

    let filename = "train/train_3.json";
    match load_instance(filename) {
        Ok(instance) => {
            let (mut best_solution, best_fit, fitness_history) = island::run_island_model(
                &instance,
                num_islands,
                total_population,
                num_generations,
                migration_interval,
                migration_size,
                mutation_rate,
                crossover_rate,
                elitism_count,
                tournament_size,
                sigma_share,
                alpha,
            );
            println!("\nBeste løsning funnet med raw fitness {:.2}:", best_fit);
            println!("{:?}", best_solution.flat);
            println!("Benchmark: {:.2}", instance.benchmark);
            let abs_diff = (best_fit - instance.benchmark).abs();
            println!("Absolutt differanse fra benchmark: {:.2}", abs_diff);
            let percent_diff = ((best_fit - instance.benchmark) / instance.benchmark) * 100.0;
            println!("Relativ feil: {:.2}%", percent_diff);

            // Kjør lokalt søk (intra-rute) på hver rute for videre forbedring
            for route in &mut best_solution.routes {
                if route.len() > 1 {
                    *route = local_search::extended_local_search(route, &instance);
                }
            }
            let improved_total: f64 = best_solution
                .routes
                .iter()
                .filter_map(|r| models::simulate_route(r, &instance).map(|(t, _)| t))
                .sum();
            println!(
                "Etter lokal forbedring: Total travel time = {:.2}",
                improved_total
            );

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
