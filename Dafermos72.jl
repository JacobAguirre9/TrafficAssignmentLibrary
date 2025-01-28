using LinearAlgebra

# Define the data structures
struct Link
    id::Int
    g::Matrix{Float64}
    h::Vector{Float64}
end

struct Path
    id::Int
    links::Vector{Link}
end

struct ODpair
    id::Int
    demand::Vector{Float64}
    paths::Vector{Path}
end

struct Network
    links::Vector{Link}
    od_pairs::Vector{ODpair}
    classes::Int
end

function initialize_flows(network::Network)
    flows = Dict{Tuple{ODpair, Int, Path}, Float64}()
    for od in network.od_pairs
        for (cls, d) in enumerate(od.demand)
            n_paths = length(od.paths)
            flow_per_path = d / n_paths
            for p in od.paths
                flows[(od, cls, p)] = flow_per_path
            end
        end
    end
    return flows
end

function marginal_cost(path::Path, cls::Int, flows::Dict{Tuple{ODpair, Int, Path}, Float64}, network::Network)
    cost = 0.0
    for link in path.links
        f = [sum(flows[(od, c, p)] for od in network.od_pairs for p in od.paths if link in p.links) for c in 1:network.classes]
        cost += sum((link.g[cls, j] + link.g[j, cls]) * f[j] for j in 1:network.classes) + link.h[cls]
    end
    return cost
end

function compute_delta(q_path::Path, r_path::Path, cls::Int, flows::Dict{Tuple{ODpair, Int, Path}, Float64}, network::Network)
    current_flow = sum(flows[(od, cls, p)] for od in network.od_pairs for p in od.paths if p == r_path)
    current_flow = max(current_flow, 0.0)

    denominator = 0.0
    for link in union(q_path.links, r_path.links)
        denominator += 4 * link.g[cls, cls]
    end
    denominator = max(denominator, 1e-6)

    c_q = marginal_cost(q_path, cls, flows, network)
    c_r = marginal_cost(r_path, cls, flows, network)
    delta = (c_r - c_q) / denominator

    return clamp(delta, 0.0, current_flow)
end

function traffic_assignment!(network::Network; max_iters=1000, tolerance=1e-4)
    flows = initialize_flows(network)
    println("Starting Traffic Assignment...")
    println("Initial Flow Distribution:")
    display_flow_summary(flows, network)

    for iter in 1:max_iters
        max_delta = 0.0
        total_flow_change = 0.0
        flow_changes = Dict{Tuple{ODpair, Int, Path}, Float64}()

        for od in network.od_pairs
            for cls in 1:network.classes
                paths = od.paths
                mc = [marginal_cost(p, cls, flows, network) for p in paths]
                used = [flows[(od, cls, p)] > 0.0 for p in paths]
                if !any(used)
                    continue
                end
                mc_used = mc[used]
                paths_used = paths[used]

                q_idx = argmin(mc_used)
                r_idx = argmax(mc_used)
                q_path = paths_used[q_idx]
                r_path = paths_used[r_idx]

                delta = compute_delta(q_path, r_path, cls, flows, network)
                if delta > 0
                    key_q = (od, cls, q_path)
                    key_r = (od, cls, r_path)
                    flows[key_q] += delta
                    flows[key_r] -= delta
                    flow_changes[key_q] = get(flow_changes, key_q, 0.0) + delta
                    flow_changes[key_r] = get(flow_changes, key_r, 0.0) - delta
                    max_delta = max(max_delta, delta)
                    total_flow_change += abs(delta)
                end
            end
        end

        println("\nIteration $iter: ", " Max ΔFlow = $max_delta ", " Total Flow Change = $total_flow_change")
        for ((od, cls, p), change) in flow_changes
            println("    OD $(od.id), Class $cls, Path $(p.id): ΔFlow = $(round(change, digits=4))")
        end

        if max_delta < tolerance
            println("\nConverged in $iter iterations.")
            println("Final Flow Distribution:")
            display_flow_summary(flows, network)
            return flows
        end
    end
    println("\nReached maximum iterations ($max_iters) without full convergence.")
    println("Final Flow Distribution:")
    display_flow_summary(flows, network)
    return flows
end

# Helper function to display a summary of flows
function display_flow_summary(flows::Dict{Tuple{ODpair, Int, Path}, Float64}, network::Network)
    for od in network.od_pairs
        println("\nOD Pair $(od.id):")
        for cls in 1:network.classes
            total_flow = sum(flows[(od, cls, p)] for p in od.paths)
            println("  Class $cls: Total Demand = $(od.demand[cls]), Total Flow = $(round(total_flow, digits=4))")
            for p in od.paths
                flow = flows[(od, cls, p)]
                if flow > 0.0
                    println("    Path $(p.id): Flow = $(round(flow, digits=4))")
                end
            end
        end
    end
end

# Example Data Initialization
links = [
    Link(1, [2 1; 1 4], [1500, 1700]),
    Link(2, [2 2; 2 6], [300, 250]),
    Link(3, [2 2; 2 5], [800, 600]),
    Link(4, [3 3; 3 6], [1200, 100]),
    Link(5, [3 2; 2 6], [100, 100]),
    Link(6, [2 1; 1 5], [100, 1300]),
    Link(7, [2 1; 1 4], [200, 100]),
]

od1_paths = [
    Path(4, [links[4], links[3]]),
    Path(5, [links[2], links[6]]),
]
od2_paths = [
    Path(1, [links[1]]),
    Path(2, [links[6], links[2]]),
    Path(3, [links[3], links[7]]),
]

od_pairs = [
    ODpair(1, [200.0, 80.0], od1_paths),
    ODpair(2, [120.0, 60.0], od2_paths),
]

network = Network(links, od_pairs, 2)

# Run the Traffic Assignment Algorithm
flows = traffic_assignment!(network)

# Display Final Flows
println("\nFinal Flow Distribution:")
for key in keys(flows)
    od, cls, p = key
    if flows[key] > 0.0
        println("OD $(od.id), Class $cls, Path $(p.id): Flow = $(round(flows[key], digits=4))")
    end
end
