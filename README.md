# StaticArraysLite.jl

Experiment in implementing StaticArrays.jl without complex techniques like generated
functions.

Mostly relies on heurestics for inlining, constant propagation and loop unrolling, with a
twist of using [Freeze.jl](https://github.com/andyferris/Freeze.jl) for immutable semantics
and the guarantee of returning immutable values from "functional" operations like `map`.

So far, seems to generate roughly equivalent code for static arrays for 3-vectors of `Float64` with
some really basic functions.