"""An `ERA5Grid` is the reduced Gaussian grid used by ECMWF's IFS and the ERA5 reanalysis.
Like the `OctahedralGaussianGrid` it uses the Gaussian latitudes of the equivalent `FullGaussianGrid`
and a reduced number of longitude points towards the poles. In contrast to the octahedral grids the
number of longitude points per ring does not follow a simple formula but is tabulated (the "classical"
or non-octahedral reduced Gaussian grid). All longitude points are equally spaced per ring, starting
at 0˚E (no offset, like the octahedral grids).

Only the `nlat_half = 320` resolution (the ECMWF "N320" grid with 542,080 grid points) is supported,
as this is the native grid of ERA5. The longitude counts per ring are all of the form ``2^a 3^b 5^c``
so that the longitudinal FFT of the spherical harmonic transform is efficient.

The first dimension of data on this grid (a `Field`) represents the horizontal dimension,
in ring order (0 to 360˚E, then north to south), other dimensions can be used for the vertical and/or
time or other dimensions. Note that a `Grid` does not contain any data, it only describes
the discretization of the space, see `Field` for data on a `Grid`.

The resolution parameter of the horizontal grid is `nlat_half` (number of latitude rings on one hemisphere,
Equator included), `rings` are the precomputed ring indices, e.g. `rings = [1:18, 19:43, ...]`.
`whichring` is a precomputed vector of ring indices for each grid point ij, i.e. `whichring[ij]` gives
the ring index j of grid point ij. For efficient looping see `eachring` and `eachgrid`.
Fields are
$(TYPEDFIELDS)"""
struct ERA5Grid{A, V, W, IntType} <: AbstractReducedGrid{A}
    nlat_half::IntType              # number of latitudes on one hemisphere
    architecture::A                 # information about device, CPU/GPU
    rings::V                        # precomputed ring indices (ring j -> grid point ij)
    whichring::W                    # precomputed ring index for each grid point ij
end

# TYPES
Architectures.nonparametric_type(::Type{<:ERA5Grid}) = ERA5Grid
full_grid_type(::Type{<:ERA5Grid}) = FullGaussianGrid

# FIELD
const ERA5Field{T, N} = Field{T, N, ArrayType, Grid} where {ArrayType, Grid <: ERA5Grid}

# define grid_type (i) without T, N, (ii) with T, (iii) with T, N but not with <:?Field
# to not have precendence over grid_type(::Type{Field{...})
grid_type(::Type{ERA5Field}) = ERA5Grid
grid_type(::Type{ERA5Field{T}}) where {T} = ERA5Grid
grid_type(::Type{ERA5Field{T, N}}) where {T, N} = ERA5Grid

function Base.showarg(io::IO, F::Field{T, N, ArrayType, Grid}, toplevel) where {T, N, ArrayType, Grid <: ERA5Grid{A}} where {A <: AbstractArchitecture}
    print(io, "ERA5Field{$T, $N}")
    toplevel && print(io, " as ", nonparametric_type(ArrayType))
    return toplevel && print(io, " on ", F.grid.architecture)
end

# SIZE
# Number of longitude points per ring for the N320 grid, north pole to Equator (320 entries).
# The southern hemisphere mirrors this. Matches the "reduced points (standard)" column of ECMWF's
# N320 definition (https://www.ecmwf.int/en/forecasts/documentation-and-support/gaussian_n320).
const _N320_NLON = (
    18, 25, 36, 40, 45, 50, 60, 64, 72, 72,
    75, 81, 90, 96, 100, 108, 120, 120, 125, 135,
    144, 144, 150, 160, 180, 180, 180, 192, 192, 200,
    216, 216, 216, 225, 240, 240, 240, 250, 256, 270,
    270, 288, 288, 288, 300, 300, 320, 320, 320, 324,
    360, 360, 360, 360, 360, 360, 375, 375, 384, 384,
    400, 400, 405, 432, 432, 432, 432, 450, 450, 450,
    480, 480, 480, 480, 480, 486, 500, 500, 500, 512,
    512, 540, 540, 540, 540, 540, 576, 576, 576, 576,
    576, 576, 600, 600, 600, 600, 640, 640, 640, 640,
    640, 640, 640, 648, 648, 675, 675, 675, 675, 720,
    720, 720, 720, 720, 720, 720, 720, 720, 729, 750,
    750, 750, 750, 768, 768, 768, 768, 800, 800, 800,
    800, 800, 800, 810, 810, 864, 864, 864, 864, 864,
    864, 864, 864, 864, 864, 864, 900, 900, 900, 900,
    900, 900, 900, 900, 960, 960, 960, 960, 960, 960,
    960, 960, 960, 960, 960, 960, 960, 960, 972, 972,
    1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1024, 1024,
    1024, 1024, 1024, 1024, 1080, 1080, 1080, 1080, 1080, 1080,
    1080, 1080, 1080, 1080, 1080, 1080, 1080, 1080, 1125, 1125,
    1125, 1125, 1125, 1125, 1125, 1125, 1125, 1125, 1125, 1125,
    1125, 1125, 1152, 1152, 1152, 1152, 1152, 1152, 1152, 1152,
    1152, 1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200,
    1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200, 1215,
    1215, 1215, 1215, 1215, 1215, 1215, 1280, 1280, 1280, 1280,
    1280, 1280, 1280, 1280, 1280, 1280, 1280, 1280, 1280, 1280,
    1280, 1280, 1280, 1280, 1280, 1280, 1280, 1280, 1280, 1280,
    1280, 1280, 1280, 1280, 1280, 1280, 1280, 1280, 1280, 1280,
    1280, 1280, 1280, 1280, 1280, 1280, 1280, 1280, 1280, 1280,
    1280, 1280, 1280, 1280, 1280, 1280, 1280, 1280, 1280, 1280,
    1280, 1280, 1280, 1280, 1280, 1280, 1280, 1280, 1280, 1280,
    1280, 1280, 1280, 1280, 1280, 1280, 1280, 1280, 1280, 1280,
)

const N320_NLAT_HALF = 320                          # the only supported resolution
const N320_NPOINTS = 2sum(_N320_NLON)               # total number of grid points (542,080)

nlat_odd(::Type{<:ERA5Grid}) = false

# the ERA5Grid only tabulates the N320 grid, error otherwise
@inline _assert_n320(nlat_half::Integer) = nlat_half == N320_NLAT_HALF ||
    throw(ArgumentError("ERA5Grid only supports nlat_half=$N320_NLAT_HALF (the ERA5 N320 grid), got $nlat_half."))

function get_npoints(::Type{<:ERA5Grid}, nlat_half::Integer)
    _assert_n320(nlat_half)
    return N320_NPOINTS
end

function get_nlat_half(::Type{<:ERA5Grid}, npoints2D::Integer)
    npoints2D == N320_NPOINTS ||
        throw(ArgumentError("ERA5Grid only supports npoints=$N320_NPOINTS (the ERA5 N320 grid), got $npoints2D."))
    return N320_NLAT_HALF
end

function get_nlon_per_ring(Grid::Type{<:ERA5Grid}, nlat_half::Integer, j::Integer)
    _assert_n320(nlat_half)
    nlat = get_nlat(Grid, nlat_half)
    @assert 0 < j <= nlat "Ring $j is outside N$nlat_half grid."
    j = j > nlat_half ? nlat - j + 1 : j        # flip north south due to symmetry
    return @inbounds _N320_NLON[j]
end

## COORDINATES
get_latd(::Type{<:ERA5Grid}, nlat_half::Integer) = get_latd(FullGaussianGrid, nlat_half)
function get_lond_per_ring(Grid::Type{<:ERA5Grid}, nlat_half::Integer, j::Integer)
    nlon = get_nlon_per_ring(Grid, nlat_half, j)
    return collect(0:(360 / nlon):(360 - 180 / nlon))
end

## QUADRATURE
get_quadrature_weights(::Type{<:ERA5Grid}, nlat_half::Integer) = gaussian_weights(nlat_half)

## INDEXING
"""$(TYPEDSIGNATURES) precompute a `Vector{UnitRange{Int}} to index grid points on
every ring `j` (elements of the vector) of `Grid` at resolution `nlat_half`.
See `eachring` and `eachgrid` for efficient looping over grid points."""
function each_index_in_ring!(
        rings,
        Grid::Type{<:ERA5Grid},
        nlat_half::Integer
    ) # resolution param

    nlat = length(rings)
    @boundscheck nlat == get_nlat(Grid, nlat_half) || throw(BoundsError)

    index_end = 0
    @inbounds for j in 1:nlat
        index_1st = index_end + 1                       # 1st index is +1 from prev ring's last index
        index_end += get_nlon_per_ring(Grid, nlat_half, j)  # add number of grid points per ring
        rings[j] = index_1st:index_end                  # turn into UnitRange
    end
    return rings
end
