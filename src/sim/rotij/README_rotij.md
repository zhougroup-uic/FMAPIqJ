# Rotij.jl
Builds all rotation lookup tables from a `rot.eul` (Euler ZXZ angles, rad) in one go, on demand; no external dependencies (standard library only),
multi-threaded. Same tables and file formats as the C pipeline in `../src` (`rotij_idps3.x`, `rotij_idps3a.x`,
`rotchkmin`, `build_rotm`, `txt2short.py`, `txt2double.py`), and it works for any `rot.eul` (exact ball-tree
nearest-sample search, see `../CHANGELOG.md`).

    julia -t auto --project=julia julia/bin/rotij.jl rot.eul outdir [nbin=120] [exact|lookup]

writes `outdir/{rot.dat.gz, rotm.dat.gz, rot.eul.chkmin.nk3, angijk.dat.gz, angij2k.dat.gz, angij2a.dat.gz}`; with `lookup` the pair tables `angij2k`/`angij2a` are filled approximately from the `angijk` table (see the last section). As a library:

```julia
using Rotij
S = load_eul("rot.eul")           # samples + rotation matrices + ball tree (O(n log n))
nearest(S, a, b, c)               # (row, angle): nearest sample to Euler angles (a,b,c) (1-based row)
A  = angijk(S; nbin=120)          # A[ai+1,aj+1,ak+1], 0-based sample index of every (alpha, cos beta, gamma) bin
Pk = angij(S; inv=true)           # A2-A1: P[i+1,j+1] = nearest to R_i' R_j     (angij2k)
Pa = angij(S; inv=false)          # A1+A2: nearest to R_i R_j                    (angij2a)
Rm = rotm(S)                      # 9×n inverse rotation matrices (build_rotm)
Pk = angij(S; inv=true, A=A)      # same, approximate: O(1) lookup in A instead of n^2 searches
bin_lookup(A, q)                  # approximate nearest sample of a rotation matrix q from the bin table
j, d = chkmin(S)                  # nearest other sample and rotation angle of every sample
write_tables(S, "outdir")         # or build_all("rot.eul", "outdir")
```
Index tables are 0-based like the `.dat.gz` files (16-bit, so at most 32768 samples; `index_type=Int32` for more).
Test: `julia --project=julia -e 'using Pkg; Pkg.test()'` (exhaustive-search checks on three sample sets).

## Folded pair tables (prototype)
`angij2k`/`angij2a` need n^2 entries, but the sample list is invariant (to the 6-decimal rounding of `rot.eul`, ~1e-6 rad) under a group of 6 left and 2 right rotations, so most rows/columns are copies of others:
`P[s(i),s(j)] = P[i,j]` (`angij2k`), `Q[s(i),j] = s(Q[i,j])`, `Q[i,t(j)] = t(Q[i,j])` (`angij2a`).

```julia
sym = symmetries(S)                    # permutations induced by the symmetry rotations (6 left, 2 right for the sequence samples)
T   = fold_angij(S; inv=true)          # only the orbit representatives are searched (n^2/6 for angij2k, n^2/12 for angij2a) and bit-packed
T[i, j]                                # O(1) read (1-based i, j) of the 0-based sample index, like angij(S)[i, j]
write_folded("angij2k.fold", T); T = read_folded("angij2k.fold")   # memory-mapped, only touched pages are read
```
| 4608 samples | stored | file | build (8 threads) | entries differing from `angij` | worst excess |
|---|---|---|---|---|---|
| `angij2k` (inv=true) | 768 x 4608 x 13 bit | 5.9 MB | 6.2 s (full 22.8 s) | 0.60% | 2.0e-4 deg |
| `angij2a` (inv=false) | 768 x 2304 x 13 bit | 3.1 MB | 2.7 s (full 22.7 s) | 0.67% | 2.6e-4 deg |

(16-bit tables: 42.5 MB each.) The differing entries are near-ties: the returned sample is at most ~3e-4 degrees (the rounding of `rot.eul`) farther than the true nearest one, against 11% of entries / 0.09 deg mean / 8 deg worst for the `angijk` lookup. So the result is not byte-identical to the `.dat.gz` files; for a symmetric-exact result, round the samples to a symmetric set first. `rotation_samples/oim20.eul` (1854): 18 left rotations, 0.42 MB (vs 6.9 MB), 3x faster build, no entry differs for `angij2k`; `oim15.eul` (4392): 24 left rotations, 1.8 MB (vs 38.6 MB), 13x faster build (1.7 s vs 21.6 s), no entry differs by more than 7e-5 deg. 576 samples: 0.09 / 0.06 MB, 2.1% / 1.9% near-ties, worst 2.1e-4 deg.

## Same as the C tools?
Compared with the C pipeline of this repository (`-t 8`):
- `rot.dat.gz` identical. `rotm.dat.gz` equal to 1 ulp (2e-16; different `sin`/`cos` libraries).
- `angijk`, `angij2k`, `angij2a`: same nearest sample except exact ties (two samples at the same distance, gap <= 3e-13 rad): rot72 14 / 12 / 3 entries, rot576 9 / 379 / 19, rot4608 9 / 2060 / 32 (of 1.728e6 / n^2 / n^2); `oim20.eul`: 0 differences in all three.
- `rot.eul.chkmin.nk3`: same angles for every sample, same neighbour except 5 ties at 4608; the order of equal-angle lines is `LC_ALL=C sort -nk3` (the shell pipeline orders them by locale).
- The pair tables use the product matrix directly; the C code converts it to Euler angles and back (same result up to rounding).

## Time (8 threads, whole `build_all`)
| samples | time |
|---|---|
| 72 / 576 | 2.4 s / 3.5 s (including compilation) |
| 1854 (`oim20`) | 9.9 s |
| 4608 | 55 s: `angijk` 1.5 s, `angij2k` 23 s, `angij2a` 23 s |
Single thread, 576: `angijk` 3.1 s, each pair table 1.1 s. The bin table is cheap because neighbouring bins have neighbouring
answers (warm start); the pair tables are n^2 independent queries (~9 us each at 4608), so they grow ~n^2 log n. For 42048
samples build only the bin table (`tables=(:angijk,:rotm,:rot)`); `angij2k` would be ~3.5 GB.

## Approximate pair tables from `angijk` (`pair=:lookup`, `angij(S; A=...)`)
Instead of n^2 searches, take the relative rotation `R_i' R_j` (or `R_i R_j`), read its Euler angles, bin them and look up `angijk`
(`round` for alpha and gamma, `floor` for beta because the table's beta bin centres are half a bin offset; at beta ~ 0 or pi gamma = 0
is used). The result is the nearest sample of the bin centre. Compared with the exact tables (4608 samples, all n^2 pairs):
same sample for 88.6% of the pairs, mean excess angle 0.09 deg, worst 8-9 deg (near beta = 0 the first beta bin is ~10 deg wide);
random pairs of 576 samples: 92% same, 99% within 1.3 deg; `oim15` (4392): 89%, 99% within 1.9 deg. Whole `build_all` at 4608
samples: 8 s instead of 55 s (576: 3 s). Using `round` for beta instead gives 1-2% of pairs over 2 deg off and rare errors of ~170 deg (the polar wrap
of the `round(b2)` bin lookup). Use the exact tables to check, or when the sample spacing is not much larger than a bin (~3 deg).
