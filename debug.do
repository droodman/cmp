cap cd "Y:\Documents\Macros\cmp"
cap cd "/Users/davidroodman/Library/CloudStorage/OneDrive-Personal/Documents/Macros/cmp"
cap cd "D:\OneDrive\Documents\Macros\cmp"

// jl AddPkg IrrationalConstants
// jl AddPkg LoopVectorization
// jl AddPkg StatsFuns
// jl AddPkg SparseArrays

jl: pushfirst!(LOAD_PATH, raw"D:\\OneDrive\\Documents\\Macros\\cmp")
// jl: using CMP

sysuse auto, clear

cmp setup
// cmp (price = mpg), nolr ind(1)
cmp (price = mpg), nolr ind(1) julia
