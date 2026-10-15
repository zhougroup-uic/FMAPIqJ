# Command line as in args.c/main.c:
#   sim2iq <input.pqr> <grid_size> <grid_spacing> <output.dat> [options]
Base.@kwdef mutable struct CommandArgs
    input_file::String = ""
    output_file::String = ""
    traj_file::Union{Nothing,String} = nothing
    grid_size::Int = 0
    grid_spacing::Float64 = 0.0
    cutoff_offset::Float64 = 2.8
    b_factor::Float64 = 0.0
    steps::Int = 1
    se_weight::Float64 = -1.0
    ou_weight::Float64 = 0.011
    in_weight::Float64 = -0.011
    traj_is_txt::Bool = false
end

# C atoi/atof semantics: leading numeric prefix, 0 otherwise
_atoi(s) = (m = match(r"^\s*[+-]?\d+", s); m === nothing ? 0 : parse(Int, strip(m.match)))
_atof(s) = (m = match(r"^\s*[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?", s); m === nothing ? 0.0 : parse(Float64, strip(m.match)))

function parse_args(argv::Vector{String})
    if length(argv) < 4
        println(stderr, "Usage: sim2iq <input.pqr> <grid_size> <grid_spacing> <output.dat> [options]")
        println(stderr, "Options:")
        println(stderr, "  -b <value>        B-factor (default: 0.0)")
        println(stderr, "  -offset <value>   Cutoff offset for outer shell (default: 2.8)")
        println(stderr, "  -steps <value>    Number of calculation steps to perform (default: 1)")
        println(stderr, "  -se <value>       Solvent excluded volume weight (default: -1.0)")
        println(stderr, "  -ou <value>       Outer shell weight (default: 0.011)")
        println(stderr, "  -in <value>       Inner shell weight (default: -0.011)")
        println(stderr, "  -traj <file>      PD6D trajectory file (optional)")
        println(stderr, "  -txt              Indicate trajectory file is in text format (default: PDB format)")
        return nothing
    end
    a = CommandArgs(input_file = argv[1], grid_size = _atoi(argv[2]),
                    grid_spacing = _atof(argv[3]), output_file = argv[4])
    i = 5
    while i <= length(argv)
        o = argv[i]
        has = i + 1 <= length(argv)
        if o == "-b" && has;        a.b_factor = _atof(argv[i+1]); i += 1
        elseif o == "-offset" && has; a.cutoff_offset = _atof(argv[i+1]); i += 1
        elseif o == "-steps" && has
            a.steps = _atoi(argv[i+1]); i += 1
            if a.steps < 1 || a.steps > 4
                println(stderr, "Error: Steps must be between 1 and 4"); return nothing
            end
        elseif o == "-se" && has;   a.se_weight = _atof(argv[i+1]); i += 1
        elseif o == "-ou" && has;   a.ou_weight = _atof(argv[i+1]); i += 1
        elseif o == "-in" && has;   a.in_weight = _atof(argv[i+1]); i += 1
        elseif o == "-traj" && has; a.traj_file = argv[i+1]; i += 1
        elseif o == "-txt";         a.traj_is_txt = true
        else
            println(stderr, "Unknown option: ", o); return nothing
        end
        i += 1
    end
    a.grid_size <= 0 && (println(stderr, "Error: Grid size must be positive"); return nothing)
    a.grid_spacing <= 0.0 && (println(stderr, "Error: Grid spacing must be positive"); return nothing)

    println("Input file: ", a.input_file)
    println("Grid size: ", a.grid_size)
    @printf("Grid spacing: %.6f Å\n", a.grid_spacing)
    @printf("B-factor: %.6f\n", a.b_factor)
    @printf("Cutoff offset: %.6f Å\n", a.cutoff_offset)
    println("Number of steps: ", a.steps)
    @printf("Solvent excluded volume weight: %.6f\n", a.se_weight)
    @printf("Outer shell weight: %.6f\n", a.ou_weight)
    @printf("Inner shell weight: %.6f\n", a.in_weight)
    println("Output file: ", a.output_file)
    if a.traj_file !== nothing
        println("Trajectory file: ", a.traj_file)
        println("Trajectory format: ", a.traj_is_txt ? "Text" : "PDB")
    end
    return a
end

# Entry point mirroring main(); returns an exit code
function sim2iq_main(argv::Vector{String} = copy(ARGS))
    args = parse_args(argv)
    args === nothing && return 1
    structure = read_pqr_file(args.input_file)
    if args.traj_file !== nothing
        println("Processing trajectory file: ", args.traj_file)
        structure = process_trajectory(structure, args.traj_file, args.traj_is_txt)
        println("Using transformed structure for calculations")
    end
    b_factor = args.b_factor > 0.0 ? args.b_factor : u2B(0.25 * args.grid_spacing)
    @printf("Actual B-factor: %.6f\n", b_factor)
    result = calculate_saxs_data_multi_step(structure, args.grid_size, args.grid_spacing, args.steps,
                                            args.cutoff_offset, args.se_weight, args.ou_weight,
                                            args.in_weight, b_factor)
    if isempty(result)
        println(stderr, "Error: No data points generated"); return 1
    end
    write_output(result, args.output_file) || (println(stderr, "Error: Failed to write output file"); return 1)
    return 0
end
