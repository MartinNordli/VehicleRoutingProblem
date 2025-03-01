use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;
use std::process::Command;
use rand::prelude::*;
use rand::seq::SliceRandom;
use std::cmp::max;
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, Ordering};

// =================== DATA STRUCTURES FOR INPUT PARSING ===================

// Represents the entire problem instance with all relevant data
#[derive(Deserialize, Debug, Clone)]
struct Problem {
    instance_name: String,       // Identifier for the problem instance
    nbr_nurses: usize,           // Number of available nurses
    capacity_nurse: usize,       // Maximum capacity each nurse can handle (total patient demand)
    benchmark: f64,              // Benchmark solution value for comparison
    depot: Depot,                // Starting/ending location for all nurses
    patients: HashMap<String, Patient>, // Map of patient IDs to their data
    travel_times: Vec<Vec<f64>>, // Matrix of travel times between locations (index 0 is depot)
}

// Represents the depot (starting and ending point for all nurses)
#[derive(Deserialize, Debug, Clone)]
struct Depot {
    return_time: usize,          // Latest time nurses must return to depot
    x_coord: usize,              // X-coordinate for visualization
    y_coord: usize,              // Y-coordinate for visualization
}

// Represents a patient with their location, service requirements, and time window
#[derive(Deserialize, Debug, Clone)]
struct Patient {
    x_coord: usize,              // X-coordinate for visualization
    y_coord: usize,              // Y-coordinate for visualization
    demand: usize,               // Resource demand (counts against nurse capacity)
    start_time: usize,           // Earliest time the visit can start
    end_time: usize,             // Latest time the visit can start
    care_time: usize,            // Duration of the care/service required
}

// Solution representation: a vector of routes, each route is a vector of patient IDs
// The outer vector has length equal to number of nurses, inner vectors contain patient IDs
type Solution = Vec<Vec<usize>>;

// Data structure for generating visualization plots
#[derive(Serialize)]
struct PlotData {
    depot_x: usize,              // Depot X-coordinate
    depot_y: usize,              // Depot Y-coordinate
    patient_x: HashMap<String, usize>, // Map of patient IDs to X-coordinates
    patient_y: HashMap<String, usize>, // Map of patient IDs to Y-coordinates
    routes: Vec<Vec<usize>>,     // Solution routes for visualization
    instance_name: String,       // Problem instance name
}

// =================== MAIN FUNCTION ===================

/// Main function: parses input, runs the genetic algorithm, evaluates the solution,
/// and outputs results including visualization.
fn main() -> Result<(), Box<dyn Error>> {
    // Parse command line arguments
    let args: Vec<String> = std::env::args().collect();
    
    // Hardcoded problem file (would typically be passed as an argument)
    let problem_file = "train/train_2.json";
    
    // Set GA parameters
    let generations = 5000;      // Number of generations to evolve
    let population_size = 4000;  // Size of population in each generation
    
    // Set random seed if provided as 5th command line argument
    if args.len() > 4 {
        if let Ok(seed) = args[4].parse::<u64>() {
            let mut rng = StdRng::seed_from_u64(seed);
            println!("Using seed: {}", seed);
        }
    }
    
    println!("Running GA with {} generations and population size {}", generations, population_size);

    // Read and parse the problem from JSON file
    let problem = read_problem(problem_file)?;
    println!("Problem: {}", problem.instance_name);
    println!("Nurses: {}, Capacity: {}, Patients: {}", 
             problem.nbr_nurses, problem.capacity_nurse, problem.patients.len());
    println!("Benchmark: {}", problem.benchmark);

    // Create shared state for the best solution
    let best_solution = Arc::new(Mutex::new(Vec::new())); // Initially empty
    let best_fitness = Arc::new(Mutex::new(f64::INFINITY));
    let running = Arc::new(AtomicBool::new(true));
    
    // Clone the Arc pointers for the signal handler
    let handler_best_solution = Arc::clone(&best_solution);
    let handler_best_fitness = Arc::clone(&best_fitness);
    let handler_running = Arc::clone(&running);
    let handler_problem = Arc::new(problem.clone());
    
    // Set up Ctrl+C handler
    ctrlc::set_handler(move || {
        println!("\nInterrupted by user. Finishing gracefully...");
        
        // Signal the algorithm to stop
        handler_running.store(false, Ordering::SeqCst);
        
        // Get the best solution found so far
        let solution = handler_best_solution.lock().unwrap().clone();
        let fitness = *handler_best_fitness.lock().unwrap();
        
        if !solution.is_empty() {
            println!("\nBest solution found so far:");
            println!("Travel time: {:.2}", fitness);
            println!("Benchmark: {:.2}", handler_problem.benchmark);
            let gap = ((fitness - handler_problem.benchmark) / handler_problem.benchmark) * 100.0;
            println!("Gap to benchmark: {:.2}%", gap);
            
            // Display result classification based on gap to benchmark
            if gap <= 5.0 {
                println!("Result: EXCELLENT (Within 5% of benchmark)");
            } else if gap <= 10.0 {
                println!("Result: GOOD (Within 10% of benchmark)");
            } else if gap <= 20.0 {
                println!("Result: ACCEPTABLE (Within 20% of benchmark)");
            } else if gap <= 30.0 {
                println!("Result: POOR (Within 30% of benchmark)");
            } else {
                println!("Result: UNACCEPTABLE (More than 30% from benchmark)");
            }
            
            // Output detailed solution information
            output_solution(&solution, &handler_problem);
            
            // Generate visualization of the solution
            let plot_filename = format!("{}_interrupted_solution.png", handler_problem.instance_name);
            if let Err(e) = generate_plot_data(&solution, &handler_problem, &plot_filename) {
                println!("Error generating plot: {}", e);
            } else {
                println!("Plot generated successfully: {}", plot_filename);
            }
        }
        
        // Exit the program
        std::process::exit(0);
    }).expect("Error setting Ctrl+C handler");
    
    // Run the genetic algorithm with the shared state
    let solution = genetic_algorithm(&problem, population_size, generations, &best_solution, &best_fitness, &running);
    
    // Evaluate the solution quality
    let total_travel_time = evaluate_solution(&solution, &problem);
    
    // Print solution quality metrics
    println!("\nSolution Quality:");
    println!("Travel time: {:.2}", total_travel_time);
    println!("Benchmark: {:.2}", problem.benchmark);
    let gap = ((total_travel_time - problem.benchmark) / problem.benchmark) * 100.0;
    println!("Gap to benchmark: {:.2}%", gap);
    
    // Display result classification based on gap to benchmark
    if gap <= 5.0 {
        println!("Result: EXCELLENT (Within 5% of benchmark)");
    } else if gap <= 10.0 {
        println!("Result: GOOD (Within 10% of benchmark)");
    } else if gap <= 20.0 {
        println!("Result: ACCEPTABLE (Within 20% of benchmark)");
    } else if gap <= 30.0 {
        println!("Result: POOR (Within 30% of benchmark)");
    } else {
        println!("Result: UNACCEPTABLE (More than 30% from benchmark)");
    }
    
    // Output detailed solution information
    output_solution(&solution, &problem);
    
    // Generate visualization of the solution
    let plot_filename = format!("{}_solution.png", problem.instance_name);
    generate_plot_data(&solution, &problem, &plot_filename)?;
    
    Ok(())
}

// =================== FILE I/O FUNCTIONS ===================

/// Reads and parses the problem instance from a JSON file.
/// 
/// # Arguments
/// * `filename` - Path to the JSON file containing problem data
/// 
/// # Returns
/// * `Problem` - Parsed problem instance
/// * Error if file cannot be read or parsed
fn read_problem(filename: &str) -> Result<Problem, Box<dyn Error>> {
    let file = File::open(filename)?;
    let reader = BufReader::new(file);
    let problem: Problem = serde_json::from_reader(reader)?;
    Ok(problem)
}

// =================== GENETIC ALGORITHM CORE ===================

/// Core genetic algorithm function that evolves a population of solutions over multiple generations.
/// Implements several adaptive mechanisms including:
/// - Elitism (preserving best solutions)
/// - Tournament selection
/// - Adaptive mutation rate that increases when improvement stagnates
/// - Periodic local search to escape local optima
/// 
/// # Arguments
/// * `problem` - The problem instance
/// * `population_size` - Size of the population to maintain
/// * `generations` - Number of generations to evolve
/// 
/// # Returns
/// * `Solution` - The best solution found
fn genetic_algorithm(
    problem: &Problem, 
    population_size: usize, 
    generations: usize,
    best_solution_ref: &Arc<Mutex<Solution>>,
    best_fitness_ref: &Arc<Mutex<f64>>,
    running: &AtomicBool
) -> Solution {
    // Initialize population with randomized greedy solutions
    let mut population = initialize_population(problem, population_size);
    
    // Evaluate initial population
    let mut fitness_values = evaluate_population(&population, problem);
    
    // Track the best solution found so far
    let mut best_solution = find_best_solution(&population, &fitness_values).clone();
    let mut best_fitness = evaluate_solution(&best_solution, problem);
    
    // Update shared state with initial best solution
    *best_solution_ref.lock().unwrap() = best_solution.clone();
    *best_fitness_ref.lock().unwrap() = best_fitness;
    
    println!("Initial best solution fitness: {:.2}", best_fitness);
    
    // Parameters for adaptive mutation
    let mut mutation_rate = 0.2;  // Initial mutation probability 0.3
    let mut generations_no_improvement = 0;
    let max_no_improvement = 250; // After this many generations without improvement, apply local search
    
    // Tracking variables for summary reports
    let mut local_search_applications = 0;
    let mut local_search_improvements = 0;
    let mut mutation_rate_increases = 0;
    
    // Main GA loop - evolve over specified number of generations
    for generation in 0..generations {
        // Check if we should continue or terminate gracefully
        if !running.load(Ordering::SeqCst) {
            println!("Genetic algorithm gracefully terminated at generation {}", generation);
            break;
        }
        
        // Create new population through selection, crossover, and mutation
        let mut new_population = Vec::with_capacity(population_size);
        
        // Elitism: Keep the best solutions (top 10%)
        let elite_count = (population_size as f64 * 0.1).ceil() as usize;
        let mut indices: Vec<usize> = (0..population_size).collect();
        indices.sort_by(|&a, &b| fitness_values[a].partial_cmp(&fitness_values[b]).unwrap());
        
        for i in 0..elite_count {
            new_population.push(population[indices[i]].clone());
        }
        
        // Fill the rest with crossover and mutation
        while new_population.len() < population_size {
            // Select parents using tournament selection
            let parent1_idx = tournament_selection(&fitness_values, 5);
            let mut parent2_idx = tournament_selection(&fitness_values, 5);
            
            // Ensure different parents are selected when possible
            while parent2_idx == parent1_idx && population_size > 1 {
                parent2_idx = tournament_selection(&fitness_values, 5);
            }
            
            // Perform crossover to create child solution
            let mut child = crossover(&population[parent1_idx], &population[parent2_idx], problem);
            
            // Apply mutation with adaptive probability
            if rand::thread_rng().gen::<f64>() < mutation_rate {
                mutate(&mut child, problem);
            }
            
            // Repair solution to ensure validity
            repair_solution(&mut child, problem);
            
            // Add to new population
            new_population.push(child);
        }
        
        // Replace old population
        population = new_population;
        
        // Evaluate new population
        fitness_values = evaluate_population(&population, problem);
        
        // Update best solution if improved
        let current_best = find_best_solution(&population, &fitness_values);
        let current_best_fitness = evaluate_solution(current_best, problem);
        
        if current_best_fitness < best_fitness {
            best_solution = current_best.clone();
            best_fitness = current_best_fitness;
            
            // Update the shared state with the new best solution
            *best_solution_ref.lock().unwrap() = best_solution.clone();
            *best_fitness_ref.lock().unwrap() = best_fitness;
            
            // Only print significant improvements (fitness improved by at least 0.5%)
            println!("Generation {}: New best fitness = {:.2}", generation, best_fitness);
            
            generations_no_improvement = 0;
            mutation_rate = 0.2; // Reset mutation rate when we find improvement 0.3
        } else {
            generations_no_improvement += 1;
            
            // Adaptively adjust mutation rate if no improvement, but don't print every time
            if generations_no_improvement % 10 == 0 {
                mutation_rate = (mutation_rate + 0.1).min(0.8); // Increase mutation rate but cap at 0.8
                mutation_rate_increases += 1;
            }
            
            // Apply local search if stuck, but don't print every time
            if generations_no_improvement >= max_no_improvement {
                local_search_applications += 1;
                let improved_solution = local_search(&best_solution, problem);
                let improved_fitness = evaluate_solution(&improved_solution, problem);
                
                if improved_fitness < best_fitness {
                    best_solution = improved_solution;
                    best_fitness = improved_fitness;
                    
                    // Update the shared state with the improved solution
                    *best_solution_ref.lock().unwrap() = best_solution.clone();
                    *best_fitness_ref.lock().unwrap() = best_fitness;
                    
                    println!("Local search improved solution to {:.2}", best_fitness);
                    local_search_improvements += 1;
                    generations_no_improvement = 0;
                    mutation_rate = 0.2; // Reset mutation rate 0.3
                }
            }
        }
        
        // Log progress periodically (every 250 generations)
        if generation % 250 == 0 && generation > 0 {
            println!("Progress - Generation {}/{} ({}%)", 
                     generation, generations, (generation * 100) / generations);
            println!("  Current best fitness: {:.2}", best_fitness);
            println!("  Local searches: {} (with {} improvements)", 
                     local_search_applications, local_search_improvements);
            println!("  Mutation rate: {:.2}", mutation_rate);
            
            // Reset counters for next report period
            local_search_applications = 0;
            mutation_rate_increases = 0;
        }
    }
    
    // Print final summary
    println!("\nSearch complete: Best fitness = {:.2}", best_fitness);
    
    // Final local search to improve the best solution
    let final_solution = local_search(&best_solution, problem);
    let final_fitness = evaluate_solution(&final_solution, problem);
    
    if final_fitness < best_fitness {
        println!("Final local search improved solution from {:.2} to {:.2}", 
                 best_fitness, final_fitness);
        
        // Update the shared state with the final improved solution
        *best_solution_ref.lock().unwrap() = final_solution.clone();
        *best_fitness_ref.lock().unwrap() = final_fitness;
        
        return final_solution;
    }
    
    best_solution
}

// =================== LOCAL SEARCH FUNCTIONS ===================

/// Applies local search to improve a solution using multiple neighborhood operators.
/// Continues until no further improvements are found or max iterations reached.
/// 
/// # Arguments
/// * `solution` - The solution to improve
/// * `problem` - The problem instance
/// 
/// # Returns
/// * `Solution` - The improved solution
fn local_search(solution: &Solution, problem: &Problem) -> Solution {
    let mut improved_solution = solution.clone();
    let mut improved = true;
    let mut iterations = 0;
    let max_iterations = 100; // Limit iterations to prevent infinite loops
    
    while improved && iterations < max_iterations {
        improved = false;
        iterations += 1;
        
        // Try various local search operators
        
        // 1. Relocate move: Move a patient from one route to another
        improved |= try_relocate_move(&mut improved_solution, problem);
        
        // 2. Exchange move: Swap two patients between different routes
        improved |= try_exchange_move(&mut improved_solution, problem);
        
        // 3. 2-opt: Reverse a segment within a route
        improved |= try_2opt_move(&mut improved_solution, problem);
        
        // 4. Or-opt: Move a segment of consecutive patients
        improved |= try_or_opt_move(&mut improved_solution, problem);
    }
    
    improved_solution
}

/// Attempts to relocate a patient from one route to another to reduce total travel time.
/// Tries all possible (from_route, patient, to_route, position) combinations.
/// 
/// # Arguments
/// * `solution` - The solution to modify
/// * `problem` - The problem instance
/// 
/// # Returns
/// * `bool` - True if an improvement was found and applied
fn try_relocate_move(solution: &mut Solution, problem: &Problem) -> bool {
    let mut best_improvement = 0.0;
    let mut best_move: Option<(usize, usize, usize, usize)> = None; // (from_nurse, patient_pos, to_nurse, insert_pos)
    
    // For each nurse route
    for from_nurse in 0..solution.len() {
        let from_route = &solution[from_nurse];
        
        if from_route.is_empty() {
            continue;
        }
        
        // For each patient in the route
        for patient_pos in 0..from_route.len() {
            let patient_id = from_route[patient_pos];
            
            // Calculate cost without this patient
            let mut temp_route = from_route.clone();
            temp_route.remove(patient_pos);
            let old_cost = calculate_route_travel_time(from_route, problem);
            let new_cost_from = calculate_route_travel_time(&temp_route, problem);
            let change_from = new_cost_from - old_cost;
            
            // Try inserting into every other route
            for to_nurse in 0..solution.len() {
                if to_nurse == from_nurse {
                    continue;
                }
                
                let to_route = &solution[to_nurse];
                
                // Try every position in the target route
                for insert_pos in 0..=to_route.len() {
                    let mut new_route = to_route.clone();
                    new_route.insert(insert_pos, patient_id);
                    
                    // Check if the new route is valid
                    if is_valid_route(&new_route, to_nurse, problem) {
                        let old_cost_to = calculate_route_travel_time(to_route, problem);
                        let new_cost_to = calculate_route_travel_time(&new_route, problem);
                        let change_to = new_cost_to - old_cost_to;
                        
                        // Calculate total cost change
                        let total_change = change_from + change_to;
                        
                        if total_change < best_improvement {
                            best_improvement = total_change;
                            best_move = Some((from_nurse, patient_pos, to_nurse, insert_pos));
                        }
                    }
                }
            }
        }
    }
    
    // Apply the best move if found
    if let Some((from_nurse, patient_pos, to_nurse, insert_pos)) = best_move {
        let patient_id = solution[from_nurse][patient_pos];
        solution[from_nurse].remove(patient_pos);
        solution[to_nurse].insert(insert_pos, patient_id);
        return true;
    }
    
    false
}

/// Attempts to exchange two patients between different routes to reduce total travel time.
/// Tries all possible (route1, patient1, route2, patient2) combinations.
/// 
/// # Arguments
/// * `solution` - The solution to modify
/// * `problem` - The problem instance
/// 
/// # Returns
/// * `bool` - True if an improvement was found and applied
fn try_exchange_move(solution: &mut Solution, problem: &Problem) -> bool {
    let mut best_improvement = 0.0;
    let mut best_move: Option<(usize, usize, usize, usize)> = None; // (nurse1, pos1, nurse2, pos2)
    
    // For each pair of nurse routes
    for nurse1 in 0..solution.len() {
        let route1 = &solution[nurse1];
        
        if route1.is_empty() {
            continue;
        }
        
        for nurse2 in nurse1+1..solution.len() {
            let route2 = &solution[nurse2];
            
            if route2.is_empty() {
                continue;
            }
            
            // For each patient in the first route
            for pos1 in 0..route1.len() {
                let patient1 = route1[pos1];
                
                // For each patient in the second route
                for pos2 in 0..route2.len() {
                    let patient2 = route2[pos2];
                    
                    // Create new routes with patients swapped
                    let mut new_route1 = route1.clone();
                    let mut new_route2 = route2.clone();
                    
                    new_route1[pos1] = patient2;
                    new_route2[pos2] = patient1;
                    
                    // Check if both new routes are valid
                    if is_valid_route(&new_route1, nurse1, problem) && 
                       is_valid_route(&new_route2, nurse2, problem) {
                        
                        let old_cost1 = calculate_route_travel_time(route1, problem);
                        let old_cost2 = calculate_route_travel_time(route2, problem);
                        let new_cost1 = calculate_route_travel_time(&new_route1, problem);
                        let new_cost2 = calculate_route_travel_time(&new_route2, problem);
                        
                        let total_change = (new_cost1 + new_cost2) - (old_cost1 + old_cost2);
                        
                        if total_change < best_improvement {
                            best_improvement = total_change;
                            best_move = Some((nurse1, pos1, nurse2, pos2));
                        }
                    }
                }
            }
        }
    }
    
    // Apply the best move if found
    if let Some((nurse1, pos1, nurse2, pos2)) = best_move {
        let patient1 = solution[nurse1][pos1];
        let patient2 = solution[nurse2][pos2];
        
        solution[nurse1][pos1] = patient2;
        solution[nurse2][pos2] = patient1;
        return true;
    }
    
    false
}

/// Tries to improve a solution by applying the 2-opt operator (reversing segments within routes).
/// For each route, tries all possible segment inversions that improve travel time.
/// 
/// # Arguments
/// * `solution` - The solution to modify
/// * `problem` - The problem instance
/// 
/// # Returns
/// * `bool` - True if an improvement was found and applied
fn try_2opt_move(solution: &mut Solution, problem: &Problem) -> bool {
    let mut improved = false;
    
    // For each nurse route
    for nurse_id in 0..solution.len() {
        let route = &mut solution[nurse_id];
        
        if route.len() < 3 {
            continue; // Need at least 3 patients for 2-opt to make sense
        }
        
        let orig_cost = calculate_route_travel_time(route, problem);
        
        // Try all possible segment reversals
        for i in 0..route.len()-1 {
            for j in i+1..route.len() {
                // Reverse the segment [i..=j]
                route[i..=j].reverse();
                
                // Check if the new route is valid and better
                if is_valid_route(route, nurse_id, problem) {
                    let new_cost = calculate_route_travel_time(route, problem);
                    
                    if new_cost < orig_cost {
                        improved = true;
                        break; // Accept first improvement
                    } else {
                        // Revert if not improved
                        route[i..=j].reverse();
                    }
                } else {
                    // Revert if not valid
                    route[i..=j].reverse();
                }
            }
            
            if improved {
                break;
            }
        }
    }
    
    improved
}

/// Tries to improve a solution by relocating a segment of consecutive patients.
/// Similar to relocate, but moves multiple patients at once.
/// 
/// # Arguments
/// * `solution` - The solution to modify
/// * `problem` - The problem instance
/// 
/// # Returns
/// * `bool` - True if an improvement was found and applied
fn try_or_opt_move(solution: &mut Solution, problem: &Problem) -> bool {
    let mut best_improvement = 0.0;
    let mut best_move: Option<(usize, usize, usize, usize, Vec<usize>)> = None; // (from_nurse, start_pos, to_nurse, insert_pos, segment)
    let segment_length = 2; // Try moving 2 consecutive patients
    
    // For each nurse route
    for from_nurse in 0..solution.len() {
        let route = &solution[from_nurse];
        
        if route.len() < segment_length + 1 {
            continue; // Need enough patients for segment plus at least one more
        }
        
        // Try relocating a segment of patients
        for start_pos in 0..=route.len()-segment_length {
            // Extract the segment
            let segment: Vec<usize> = route[start_pos..start_pos+segment_length].to_vec();
            
            // Try placing the segment in each route (including the same route)
            for to_nurse in 0..solution.len() {
                let to_route = if to_nurse == from_nurse {
                    // Create a temporary route without the segment
                    let mut temp_route = route.clone();
                    for _ in 0..segment_length {
                        temp_route.remove(start_pos);
                    }
                    temp_route
                } else {
                    solution[to_nurse].clone()
                };
                
                // Try every insertion position
                for insert_pos in 0..=to_route.len() {
                    // Skip invalid insertions in the same route
                    if to_nurse == from_nurse && 
                      (insert_pos >= start_pos && insert_pos <= start_pos + segment_length) {
                        continue;
                    }
                    
                    // Create the new route with the segment inserted
                    let mut new_route = to_route.clone();
                    for (i, &patient_id) in segment.iter().enumerate() {
                        new_route.insert(insert_pos + i, patient_id);
                    }
                    
                    // Check if the new route is valid
                    if is_valid_route(&new_route, to_nurse, problem) {
                        // Calculate cost difference
                        let old_from_cost = calculate_route_travel_time(route, problem);
                        let old_to_cost = if to_nurse == from_nurse {
                            0.0 // Already included in old_from_cost
                        } else {
                            calculate_route_travel_time(&to_route, problem)
                        };
                        
                        // Create temporary solution with the move applied
                        let mut temp_solution = solution.clone();
                        
                        // Remove segment from original route
                        for _ in 0..segment_length {
                            temp_solution[from_nurse].remove(start_pos);
                        }
                        
                        // Insert segment into target route
                        for (i, &patient_id) in segment.iter().enumerate() {
                            let actual_insert_pos = if to_nurse == from_nurse && insert_pos > start_pos {
                                // Adjust position if inserting later in the same route
                                insert_pos - segment_length + i
                            } else {
                                insert_pos + i
                            };
                            
                            temp_solution[to_nurse].insert(actual_insert_pos, patient_id);
                        }
                        
                        let new_from_cost = calculate_route_travel_time(&temp_solution[from_nurse], problem);
                        let new_to_cost = if to_nurse == from_nurse {
                            0.0 // Already included in new_from_cost
                        } else {
                            calculate_route_travel_time(&temp_solution[to_nurse], problem)
                        };
                        
                        let total_change = (new_from_cost + new_to_cost) - (old_from_cost + old_to_cost);
                        
                        if total_change < best_improvement {
                            best_improvement = total_change;
                            best_move = Some((from_nurse, start_pos, to_nurse, insert_pos, segment.clone()));
                        }
                    }
                }
            }
        }
    }
    
    // Apply the best move if found
    if let Some((from_nurse, start_pos, to_nurse, insert_pos, segment)) = best_move {
        // Apply the move to the solution
        
        // Remove segment from original route
        for _ in 0..segment.len() {
            solution[from_nurse].remove(start_pos);
        }
        
        // Insert segment into target route
        for (i, &patient_id) in segment.iter().enumerate() {
            let actual_insert_pos = if to_nurse == from_nurse && insert_pos > start_pos {
                // Adjust position if inserting later in the same route
                insert_pos - segment.len() + i
            } else {
                insert_pos + i
            };
            
            solution[to_nurse].insert(actual_insert_pos, patient_id);
        }
        
        return true;
    }
    
    false
}

// =================== POPULATION INITIALIZATION ===================

/// Initializes a population of solutions for the genetic algorithm.
/// Creates diverse starting solutions using a randomized greedy approach.
/// 
/// # Arguments
/// * `problem` - The problem instance
/// * `population_size` - The number of solutions to generate
/// 
/// # Returns
/// * `Vec<Solution>` - A vector of initial solutions
fn initialize_population(problem: &Problem, population_size: usize) -> Vec<Solution> {
    let mut population = Vec::with_capacity(population_size);
    
    for _ in 0..population_size {
        let mut solution = generate_greedy_solution(problem);
        repair_solution(&mut solution, problem);
        population.push(solution);
    }
    
    population
}

/// Generates a solution using a randomized greedy approach.
/// Assigns patients one by one to the best feasible position.
/// 
/// # Arguments
/// * `problem` - The problem instance
/// 
/// # Returns
/// * `Solution` - A new solution
fn generate_greedy_solution(problem: &Problem) -> Solution {
    let num_patients = problem.patients.len();
    let num_nurses = problem.nbr_nurses;
    
    let mut solution = vec![Vec::new(); num_nurses];
    let mut unassigned: Vec<usize> = (1..=num_patients).collect();
    unassigned.shuffle(&mut rand::thread_rng());
    
    for patient_id in unassigned {
        let mut best_nurse = 0;
        let mut best_position = 0;
        let mut best_cost_increase = f64::INFINITY;
        
        for nurse_id in 0..num_nurses {
            let route = &solution[nurse_id];
            
            // Try each possible insertion position
            for pos in 0..=route.len() {
                let mut new_route = route.clone();
                new_route.insert(pos, patient_id);
                
                // Check if insertion is valid
                if is_valid_route(&new_route, nurse_id, problem) {
                    let old_cost = calculate_route_travel_time(route, problem);
                    let new_cost = calculate_route_travel_time(&new_route, problem);
                    let cost_increase = new_cost - old_cost;
                    
                    if cost_increase < best_cost_increase {
                        best_nurse = nurse_id;
                        best_position = pos;
                        best_cost_increase = cost_increase;
                    }
                }
            }
        }
        
        // If we found a valid position, insert the patient
        if best_cost_increase < f64::INFINITY {
            solution[best_nurse].insert(best_position, patient_id);
        } else {
            // If no valid position found, create a new route for this patient
            let nurse_id = rand::thread_rng().gen_range(0..num_nurses);
            solution[nurse_id] = vec![patient_id];
        }
    }
    
    solution
}

// =================== SOLUTION REPAIR FUNCTIONS ===================

/// Fixes time window violations in a nurse's route.
/// Handles cases where patients can't be visited within their time windows
/// or the nurse can't return to depot on time.
/// 
/// # Arguments
/// * `solution` - The solution to repair
/// * `nurse_id` - The nurse route to check
/// * `problem` - The problem instance
fn fix_time_window_violation(solution: &mut Solution, nurse_id: usize, problem: &Problem) {
    let route = &solution[nurse_id];
    
    if route.is_empty() {
        return; // Empty route is valid
    }
    
    // Check if the route violates time window constraints
    let mut current_time = 0.0;
    let mut current_location = 0; // depot
    let mut violations = Vec::new();
    
    for (idx, &patient_id) in route.iter().enumerate() {
        // Travel to patient
        current_time += problem.travel_times[current_location][patient_id];
        current_location = patient_id;
        
        let patient = problem.patients.get(&patient_id.to_string()).unwrap();
        
        // Check if we arrived too late
        if current_time > patient.end_time as f64 {
            violations.push((idx, patient_id));
        }
        
        // Wait if arrived early
        if current_time < patient.start_time as f64 {
            current_time = patient.start_time as f64;
        }
        
        // Care for patient
        current_time += patient.care_time as f64;
    }
    
    // Check if return to depot is on time
    current_time += problem.travel_times[current_location][0];
    if current_time > problem.depot.return_time as f64 {
        // If can't return on time, add the last patient to violations
        if !route.is_empty() {
            violations.push((route.len() - 1, route[route.len() - 1]));
        }
    }
    
    if violations.is_empty() {
        return; // No violations
    }
    
    let mut route = route.clone();
    
    // Sort violations in reverse order (to remove from the end)
    violations.sort_by(|a, b| b.0.cmp(&a.0));
    
    // Remove violating patients
    for (idx, patient_id) in violations {
        if idx < route.len() { // Safety check
            route.remove(idx);
            
            // Try to insert this patient into another nurse's route
            let mut inserted = false;
            
            for other_nurse_id in 0..solution.len() {
                if other_nurse_id == nurse_id {
                    continue;
                }
                
                let other_route = &solution[other_nurse_id];
                
                // Try each possible insertion position
                for pos in 0..=other_route.len() {
                    let mut new_route = other_route.clone();
                    new_route.insert(pos, patient_id);
                    
                    if is_valid_route(&new_route, other_nurse_id, problem) {
                        solution[other_nurse_id] = new_route;
                        inserted = true;
                        break;
                    }
                }
                
                if inserted {
                    break;
                }
            }
            
            // If not inserted, create a new nurse route if possible
            if !inserted {
                for new_nurse_id in 0..solution.len() {
                    if solution[new_nurse_id].is_empty() {
                        solution[new_nurse_id] = vec![patient_id];
                        inserted = true;
                        break;
                    }
                }
            }
        }
    }
    
    solution[nurse_id] = route;
}

/// Checks if a route is valid by verifying that:
/// 1. The total demand doesn't exceed nurse capacity
/// 2. All patients are visited within their time windows
/// 3. The nurse returns to depot on time
/// 
/// # Arguments
/// * `route` - The route to check
/// * `_nurse_id` - The nurse ID (unused but kept for consistency)
/// * `problem` - The problem instance
/// 
/// # Returns
/// * `bool` - True if the route is valid
fn is_valid_route(route: &[usize], _nurse_id: usize, problem: &Problem) -> bool {
    if route.is_empty() {
        return true;
    }
    
    // Check capacity constraint
    let total_demand: usize = route.iter()
        .map(|&patient_id| problem.patients.get(&patient_id.to_string()).unwrap().demand)
        .sum();
    
    if total_demand > problem.capacity_nurse {
        return false;
    }
    
    // Check time window and return time constraints
    let mut current_time = 0.0;
    let mut current_location = 0; // depot
    
    for &patient_id in route {
        // Travel to patient
        current_time += problem.travel_times[current_location][patient_id];
        current_location = patient_id;
        
        let patient = problem.patients.get(&patient_id.to_string()).unwrap();
        
        // Check if we arrived too late
        if current_time > patient.end_time as f64 {
            return false;
        }
        
        // Wait if arrived early
        if current_time < patient.start_time as f64 {
            current_time = patient.start_time as f64;
        }
        
        // Care for patient
        current_time += patient.care_time as f64;
    }
    
    // Travel back to depot
    current_time += problem.travel_times[current_location][0];
    
    // Check if we return to depot on time
    current_time <= problem.depot.return_time as f64
}

// =================== SOLUTION EVALUATION FUNCTIONS ===================

/// Evaluates all solutions in a population.
/// 
/// # Arguments
/// * `population` - The population of solutions
/// * `problem` - The problem instance
/// 
/// # Returns
/// * `Vec<f64>` - Vector of fitness values for each solution
fn evaluate_population(population: &[Solution], problem: &Problem) -> Vec<f64> {
    population.iter().map(|solution| evaluate_solution(solution, problem)).collect()
}

/// Evaluates a solution based on total travel time across all routes.
/// Lower values are better.
/// 
/// # Arguments
/// * `solution` - The solution to evaluate
/// * `problem` - The problem instance
/// 
/// # Returns
/// * `f64` - The total travel time
fn evaluate_solution(solution: &Solution, problem: &Problem) -> f64 {
    let mut total_travel_time = 0.0;
    
    for route in solution {
        total_travel_time += calculate_route_travel_time(route, problem);
    }
    
    total_travel_time
}

/// Calculates the travel time for a single route.
/// Includes travel from depot to first patient, between patients, and back to depot.
/// 
/// # Arguments
/// * `route` - The route to evaluate
/// * `problem` - The problem instance
/// 
/// # Returns
/// * `f64` - The route's travel time
fn calculate_route_travel_time(route: &[usize], problem: &Problem) -> f64 {
    if route.is_empty() {
        return 0.0;
    }
    
    let mut travel_time = problem.travel_times[0][route[0]]; // Depot to first patient
    
    for i in 0..route.len() - 1 {
        travel_time += problem.travel_times[route[i]][route[i + 1]]; // Between patients
    }
    
    travel_time += problem.travel_times[route[route.len() - 1]][0]; // Last patient to depot
    
    travel_time
}

// =================== GENETIC ALGORITHM OPERATORS ===================

/// Performs tournament selection to choose a solution from the population.
/// Selects the best solution from a random sample of tournament_size solutions.
/// 
/// # Arguments
/// * `fitness_values` - The fitness values for the population
/// * `tournament_size` - The number of solutions to include in each tournament
/// 
/// # Returns
/// * `usize` - The index of the selected solution
fn tournament_selection(fitness_values: &[f64], tournament_size: usize) -> usize {
    let mut rng = rand::thread_rng();
    let mut best_idx = rng.gen_range(0..fitness_values.len());
    let mut best_fitness = fitness_values[best_idx];
    
    for _ in 1..tournament_size {
        let idx = rng.gen_range(0..fitness_values.len());
        if fitness_values[idx] < best_fitness {
            best_idx = idx;
            best_fitness = fitness_values[idx];
        }
    }
    
    best_idx
}

/// Finds the best solution in a population based on fitness values.
/// 
/// # Arguments
/// * `population` - The population of solutions
/// * `fitness_values` - The fitness value for each solution
/// 
/// # Returns
/// * `&Solution` - Reference to the best solution
fn find_best_solution<'a>(population: &'a [Solution], fitness_values: &[f64]) -> &'a Solution {
    let best_idx = fitness_values.iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(idx, _)| idx)
        .unwrap();
    
    &population[best_idx]
}

/// Combines two parent solutions to create a new child solution.
/// Uses a segment-based crossover where some nurse routes are taken from each parent.
/// 
/// # Arguments
/// * `parent1` - First parent solution
/// * `parent2` - Second parent solution
/// * `problem` - The problem instance
/// 
/// # Returns
/// * `Solution` - The new child solution
fn crossover(parent1: &Solution, parent2: &Solution, problem: &Problem) -> Solution {
    let num_nurses = parent1.len();
    let mut child = vec![Vec::new(); num_nurses];
    
    // Select random crossover points
    let crossover_point1 = rand::thread_rng().gen_range(0..num_nurses);
    let crossover_point2 = rand::thread_rng().gen_range(0..num_nurses);
    
    let (start, end) = if crossover_point1 <= crossover_point2 {
        (crossover_point1, crossover_point2)
    } else {
        (crossover_point2, crossover_point1)
    };
    
    // Copy routes from parent1 outside the crossover points
    for i in 0..start {
        child[i] = parent1[i].clone();
    }
    
    for i in end+1..num_nurses {
        child[i] = parent1[i].clone();
    }
    
    // Copy routes from parent2 within the crossover points
    for i in start..=end {
        child[i] = parent2[i].clone();
    }
    
    // At this point, some patients may be missing or duplicated
    // This will be fixed by the repair function
    
    child
}

/// Applies a random mutation to a solution.
/// Uses several different mutation operators:
/// 1. Swap two patients within a route
/// 2. Move a patient from one route to another
/// 3. Reverse a segment within a route
/// 4. Exchange segments between routes
/// 
/// # Arguments
/// * `solution` - The solution to mutate
/// * `problem` - The problem instance
/// 
/// # Returns
/// * `bool` - True if mutation was successful
fn mutate(solution: &mut Solution, problem: &Problem) -> bool {
    let num_nurses = solution.len();
    let mut rng = rand::thread_rng();
    
    // Choose mutation type randomly
    let mutation_type = rng.gen_range(0..4);
    
    match mutation_type {
        0 => {
            // Swap two patients within a route
            let nurse_id = rng.gen_range(0..num_nurses);
            let route = &mut solution[nurse_id];
            
            if route.len() < 2 {
                return false; // Need at least 2 patients to swap
            }
            
            let idx1 = rng.gen_range(0..route.len());
            let mut idx2 = rng.gen_range(0..route.len());
            
            while idx2 == idx1 {
                idx2 = rng.gen_range(0..route.len());
            }
            
            route.swap(idx1, idx2);
            true
        },
        1 => {
            // Move a patient from one route to another
            if num_nurses < 2 {
                return false;
            }
            
            let from_nurse = rng.gen_range(0..num_nurses);
            let mut to_nurse = rng.gen_range(0..num_nurses);
            
            while to_nurse == from_nurse {
                to_nurse = rng.gen_range(0..num_nurses);
            }
            
            if solution[from_nurse].is_empty() {
                return false; // Can't move from empty route
            }
            
            let from_idx = rng.gen_range(0..solution[from_nurse].len());
            let patient_id = solution[from_nurse][from_idx];
            
            solution[from_nurse].remove(from_idx);
            
            let to_idx = if solution[to_nurse].is_empty() {
                0
            } else {
                rng.gen_range(0..=solution[to_nurse].len())
            };
            
            solution[to_nurse].insert(to_idx, patient_id);
            true
        },
        2 => {
            // Reverse a segment of a route (2-opt)
            let nurse_id = rng.gen_range(0..num_nurses);
            let route = &mut solution[nurse_id];
            
            if route.len() < 3 {
                return false; // Need at least 3 patients for meaningful reversal
            }
            
            let start = rng.gen_range(0..route.len()-1);
            let end = rng.gen_range(start+1..route.len());
            
            route[start..=end].reverse();
            true
        },
        3 => {
            // Exchange segments between two routes
            if num_nurses < 2 {
                return false;
            }
            
            let nurse1 = rng.gen_range(0..num_nurses);
            let mut nurse2 = rng.gen_range(0..num_nurses);
            
            while nurse2 == nurse1 {
                nurse2 = rng.gen_range(0..num_nurses);
            }
            
            if solution[nurse1].is_empty() || solution[nurse2].is_empty() {
                return false; // Both routes need patients
            }
            
            let len1 = solution[nurse1].len();
            let len2 = solution[nurse2].len();
            
            let start1 = rng.gen_range(0..len1);
            let length1 = rng.gen_range(1..=len1 - start1);
            
            let start2 = rng.gen_range(0..len2);
            let length2 = rng.gen_range(1..=len2 - start2);
            
            let segment1: Vec<usize> = solution[nurse1][start1..start1+length1].to_vec();
            let segment2: Vec<usize> = solution[nurse2][start2..start2+length2].to_vec();
            
            // Remove segment2 and insert segment1
            solution[nurse2].splice(start2..start2+length2, segment1);
            
            // Remove segment1 and insert segment2
            solution[nurse1].splice(start1..start1+length1, segment2);
            
            true
        },
        _ => false,
    }
}

/// Repairs a solution to ensure validity by:
/// 1. Removing duplicate patients
/// 2. Adding missing patients
/// 3. Fixing capacity violations
/// 4. Fixing time window violations
/// 
/// # Arguments
/// * `solution` - The solution to repair
/// * `problem` - The problem instance
fn repair_solution(solution: &mut Solution, problem: &Problem) {
    // First, collect all patients assigned to each route
    let mut assigned_patients = HashSet::new();
    
    for route in solution.iter() {
        for &patient_id in route {
            assigned_patients.insert(patient_id);
        }
    }
    
    // Find patients that are assigned multiple times
    let mut patient_count = HashMap::new();
    
    for route in solution.iter() {
        for &patient_id in route {
            *patient_count.entry(patient_id).or_insert(0) += 1;
        }
    }
    
    // Remove duplicate patients
    for (nurse_id, route) in solution.iter_mut().enumerate() {
        let mut i = 0;
        
        while i < route.len() {
            let patient_id = route[i];
            
            if patient_count[&patient_id] > 1 {
                route.remove(i);
                *patient_count.get_mut(&patient_id).unwrap() -= 1;
            } else {
                i += 1;
            }
        }
    }
    
    // Find missing patients
    let mut missing_patients = Vec::new();
    
    for patient_id in 1..=problem.patients.len() {
        if !assigned_patients.contains(&patient_id) || patient_count.get(&patient_id).map_or(0, |&c| c) == 0 {
            missing_patients.push(patient_id);
        }
    }
    
    // Assign missing patients
    for &patient_id in &missing_patients {
        let mut best_nurse = 0;
        let mut best_position = 0;
        let mut best_cost_increase = f64::INFINITY;
        
        // Try to insert in each position of each nurse's route
        for nurse_id in 0..solution.len() {
            let route = &solution[nurse_id];
            
            for pos in 0..=route.len() {
                let mut new_route = route.clone();
                new_route.insert(pos, patient_id);
                
                if is_valid_route(&new_route, nurse_id, problem) {
                    let old_cost = calculate_route_travel_time(route, problem);
                    let new_cost = calculate_route_travel_time(&new_route, problem);
                    let cost_increase = new_cost - old_cost;
                    
                    if cost_increase < best_cost_increase {
                        best_nurse = nurse_id;
                        best_position = pos;
                        best_cost_increase = cost_increase;
                    }
                }
            }
        }
        
        // If we found a valid position, insert the patient
        if best_cost_increase < f64::INFINITY {
            solution[best_nurse].insert(best_position, patient_id);
        } else {
            // If no valid position found, create a new route for this patient
            let nurse_id = rand::thread_rng().gen_range(0..solution.len());
            solution[nurse_id] = vec![patient_id];
        }
    }
    
    // Fix capacity violations
    for nurse_id in 0..solution.len() {
        fix_capacity_violation(solution, nurse_id, problem);
    }
    
    // Fix time window violations
    for nurse_id in 0..solution.len() {
        fix_time_window_violation(solution, nurse_id, problem);
    }
}

/// Fixes capacity violations for a specific route by:
/// 1. Identifying routes that exceed capacity
/// 2. Removing patients with highest demand until capacity is met
/// 3. Trying to reassign removed patients to other routes
/// 
/// # Arguments
/// * `solution` - The solution to repair
/// * `nurse_id` - The nurse route to check
/// * `problem` - The problem instance
fn fix_capacity_violation(solution: &mut Solution, nurse_id: usize, problem: &Problem) {
    let route = &solution[nurse_id];
    
    // Calculate total demand
    let total_demand: usize = route.iter()
        .map(|&patient_id| problem.patients.get(&patient_id.to_string()).unwrap().demand)
        .sum();
    
    if total_demand <= problem.capacity_nurse {
        return; // No violation
    }
    
    let mut route = route.clone();
    
    // Remove patients until the capacity constraint is satisfied
    while route.iter()
        .map(|&patient_id| problem.patients.get(&patient_id.to_string()).unwrap().demand)
        .sum::<usize>() > problem.capacity_nurse
    {
        // Find the patient with the highest demand
        let (idx, _) = route.iter().enumerate()
            .max_by_key(|(_, &patient_id)| problem.patients.get(&patient_id.to_string()).unwrap().demand)
            .unwrap();
        
        let removed_patient = route.remove(idx);
        
        // Try to insert this patient into another nurse's route
        let mut inserted = false;
        
        for other_nurse_id in 0..solution.len() {
            if other_nurse_id == nurse_id {
                continue;
            }
            
            let other_route = &solution[other_nurse_id];
            
            // Try each possible insertion position
            for pos in 0..=other_route.len() {
                let mut new_route = other_route.clone();
                new_route.insert(pos, removed_patient);
                
                if is_valid_route(&new_route, other_nurse_id, problem) {
                    solution[other_nurse_id] = new_route;
                    inserted = true;
                    break;
                }
            }
            
            if inserted {
                break;
            }
        }
        
        // If not inserted, create a new nurse route
        if !inserted {
            for new_nurse_id in 0..solution.len() {
                if solution[new_nurse_id].is_empty() {
                    solution[new_nurse_id] = vec![removed_patient];
                    inserted = true;
                    break;
                }
            }
        }
    }
    
    solution[nurse_id] = route;
}

// =================== OUTPUT FUNCTIONS ===================

/// Outputs detailed information about the solution, including:
/// - Route assignments for each nurse
/// - Timing details for patient visits
/// - Travel time for each route
/// - Total objective value
/// 
/// # Arguments
/// * `solution` - The solution to output
/// * `problem` - The problem instance
fn output_solution(solution: &Solution, problem: &Problem) {
    println!("\nDetailed Solution:");
    println!("Nurse capacity: {}", problem.capacity_nurse);
    println!("Depot return time: {}", problem.depot.return_time);
    println!("{:-<80}", "");
    
    for (nurse_id, route) in solution.iter().enumerate() {
        // Skip empty routes
        if route.is_empty() {
            continue;
        }
        
        let route_travel_time = calculate_route_travel_time(route, problem);
        let route_demand: usize = route.iter()
            .map(|&patient_id| problem.patients.get(&patient_id.to_string()).unwrap().demand)
            .sum();
        
        println!("Nurse {:<2} {:<12.2} {:<16} Patient sequence", 
                 nurse_id + 1, route_travel_time, route_demand);
        
        // Calculate and output timing details for each patient
        let mut current_time = 0.0;
        let mut current_location = 0; // depot
        
        print!("         D (0)");
        
        for &patient_id in route {
            // Travel to patient
            current_time += problem.travel_times[current_location][patient_id];
            current_location = patient_id;
            
            let patient = problem.patients.get(&patient_id.to_string()).unwrap();
            
            // Wait if arrived early
            let arrival_time = current_time;
            if current_time < patient.start_time as f64 {
                current_time = patient.start_time as f64;
            }
            
            // Start care
            let care_start = current_time;
            
            // Complete care
            current_time += patient.care_time as f64;
            
            print!(" -> {} ({:.2}-{:.2}) [{}-{}]", 
                   patient_id, care_start, current_time, 
                   patient.start_time, patient.end_time);
        }
        
        // Return to depot
        current_time += problem.travel_times[current_location][0];
        println!(" -> D ({:.2})", current_time);
    }
    
    println!("{:-<80}", "");
    println!("Objective value (total travel time): {:.2}", evaluate_solution(solution, problem));
}

/// Generates plot data to visualize the solution.
/// Creates a PNG file showing depot, patients, and routes.
/// 
/// # Arguments
/// * `solution` - The solution to visualize
/// * `problem` - The problem instance
/// * `output_file` - Filename for the output PNG
/// 
/// # Returns
/// * `Result<(), Box<dyn Error>>` - Success or error
fn generate_plot_data(solution: &Solution, problem: &Problem, output_file: &str) -> Result<(), Box<dyn Error>> {
    use plotters::prelude::*;
    
    // First, find the range of coordinates to set drawing area dimensions
    let mut min_x = problem.depot.x_coord as i32;
    let mut max_x = problem.depot.x_coord as i32;
    let mut min_y = problem.depot.y_coord as i32;
    let mut max_y = problem.depot.y_coord as i32;
    
    for patient in problem.patients.values() {
        min_x = min_x.min(patient.x_coord as i32);
        max_x = max_x.max(patient.x_coord as i32);
        min_y = min_y.min(patient.y_coord as i32);
        max_y = max_y.max(patient.y_coord as i32);
    }
    
    // Add some padding
    let padding = (max(max_x - min_x, max_y - min_y) / 10).max(5);
    min_x -= padding;
    max_x += padding;
    min_y -= padding;
    max_y += padding;
    
    // Create drawing area
    let root_area = BitMapBackend::new(output_file, (1024, 768))
        .into_drawing_area();
    root_area.fill(&WHITE)?;
    
    // Create chart context
    let mut chart = ChartBuilder::on(&root_area)
        .margin(20)
        .caption(format!("Solution for {}", problem.instance_name), ("sans-serif", 20).into_font())
        .build_cartesian_2d(min_x..max_x, min_y..max_y)?;
    
    chart.configure_mesh()
        .disable_x_mesh()
        .disable_y_mesh()
        .draw()?;
    
    // Define a set of colors for routes
    let colors = vec![
        &RED, &BLUE, &GREEN, &CYAN, &MAGENTA, &YELLOW, 
        &BLACK, &RGBColor(255, 165, 0), &RGBColor(128, 0, 128), &RGBColor(165, 42, 42),
        &RGBColor(0, 128, 128), &RGBColor(128, 128, 0), &RGBColor(220, 20, 60),
        &RGBColor(0, 191, 255), &RGBColor(50, 205, 50), &RGBColor(255, 215, 0),
        &RGBColor(75, 0, 130), &RGBColor(255, 20, 147), &RGBColor(0, 255, 127),
        &RGBColor(255, 99, 71)
    ];
    
    // Plot depot as a big square
    chart.draw_series(std::iter::once(
        Rectangle::new(
            [(problem.depot.x_coord as i32 - 2, problem.depot.y_coord as i32 - 2), 
             (problem.depot.x_coord as i32 + 2, problem.depot.y_coord as i32 + 2)],
            BLACK.filled()
        )
    ))?;
    
    // Plot patients as small squares with IDs
    for (id_str, patient) in &problem.patients {
        let id = id_str.parse::<usize>().unwrap_or(0);
        chart.draw_series(std::iter::once(
            Rectangle::new(
                [(patient.x_coord as i32 - 1, patient.y_coord as i32 - 1), 
                 (patient.x_coord as i32 + 1, patient.y_coord as i32 + 1)],
                BLACK.filled()
            )
        ))?;
        
        // Add patient ID text
        chart.draw_series(std::iter::once(
            Text::new(
                format!("{}", id),
                (patient.x_coord as i32 + 2, patient.y_coord as i32 + 2),
                ("sans-serif", 12).into_font()
            )
        ))?;
    }
    
    // Plot routes
    for (nurse_idx, route) in solution.iter().enumerate().filter(|(_, r)| !r.is_empty()) {
        let color_idx = nurse_idx % colors.len();
        let color = colors[color_idx];
        
        // Create points for the route starting from depot
        let mut route_points = Vec::new();
        route_points.push((problem.depot.x_coord as i32, problem.depot.y_coord as i32));
        
        for &patient_id in route {
            let patient = problem.patients.get(&patient_id.to_string()).unwrap();
            route_points.push((patient.x_coord as i32, patient.y_coord as i32));
        }
        
        // Add return to depot
        route_points.push((problem.depot.x_coord as i32, problem.depot.y_coord as i32));
        
        // Draw the route
        chart.draw_series(LineSeries::new(
            route_points,
            color.clone().stroke_width(2)
        ))?.label(format!("Nurse {}", nurse_idx + 1))
         .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], color.clone()));
    }
    
    // Draw the legend
    chart.configure_series_labels()
        .background_style(WHITE.filled())
        .border_style(BLACK)
        .position(SeriesLabelPosition::UpperRight)
        .draw()?;
    
    // Save the plot to file
    root_area.present()?;
    println!("Plot generated successfully: {}", output_file);
    
    Ok(())
}