#!/usr/bin/env julia
# check_result.jl new.stat.txt expected.stat.txt [rtol]
# Compares two Iqs.stat.txt files (q, P/P0, mean/P0, SEM). For q in [0.01, 0.4] the mean/P0 values must
# agree within rtol (relative) or within 3 combined standard errors; P(q)/P0 within 1e-3. Exit 0 = pass.
using DelimitedFiles, Printf
a = readdlm(ARGS[1], Float64; comments = true); b = readdlm(ARGS[2], Float64; comments = true)
rtol = length(ARGS) >= 3 ? parse(Float64, ARGS[3]) : 0.05
size(a, 1) == size(b, 1) || (println("FAIL: $(size(a,1)) rows vs $(size(b,1))"); exit(1))
sel = findall(q -> 0.01 <= q <= 0.4, a[:, 1])
dq = maximum(abs.(a[:, 1] .- b[:, 1]))
dP = maximum(abs.(a[:, 2] .- b[:, 2]) ./ max.(abs.(b[:, 2]), 1e-12))
diff = abs.(a[sel, 3] .- b[sel, 3])
sig = 3 .* sqrt.(a[sel, 4] .^ 2 .+ b[sel, 4] .^ 2)
bad = count(i -> diff[i] > max(rtol * abs(b[sel[i], 3]), sig[i]), eachindex(sel))
dI = maximum(diff ./ abs.(b[sel, 3]))
@printf("q grid max diff %.2e | P(q)/P0 max rel diff %.2e | mean/P0 max rel diff %.3f | outside tolerance: %d of %d points\n",
        dq, dP, dI, bad, length(sel))
ok = dq < 1e-6 && dP < 1e-3 && bad == 0
println(ok ? "PASS" : "FAIL"); exit(ok ? 0 : 1)
