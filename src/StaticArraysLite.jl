module StaticArraysLite

using Base: @propagate_inbounds

using Freeze
using LinearAlgebra

export StaticArrayLite, SArrayLite, MArrayLite

abstract type StaticArrayLite{T,N} <: AbstractArray{T, N} end

include("mapreduce.jl")
include("linalg.jl")

struct SArrayLite{S<:Tuple,T,N,L} <: StaticArrayLite{T, N}
    data::NTuple{L,T}
end

Base.length(::SArrayLite{<:Any, <:Any, <:Any, L}) where {L} = L
Base.size(::SArrayLite{Tuple{}}) = ()
Base.size(::SArrayLite{Tuple{S1}}) where S1 = (S1,)
Base.size(::SArrayLite{Tuple{S1, S2}}) where {S1, S2} = (S1, S2)
Base.axes(::SArrayLite{Tuple{}}) = ()
Base.axes(::SArrayLite{Tuple{S1}}) where S1 = (Base.OneTo(S1),)
Base.axes(::SArrayLite{Tuple{S1, S2}}) where {S1, S2} = (Base.OneTo(S1), Base.OneTo(S2))
Base.IndexStyle(::SArrayLite) = Base.IndexLinear()
Freeze.issettable(::SArrayLite) = false

@propagate_inbounds function Base.getindex(a::SArrayLite, i::Int)
    a.data[i]
end

mutable struct MArrayLite{S<:Tuple,T,N,L} <: StaticArrayLite{T, N}
    data::NTuple{L,T}

    function MArrayLite{S,T,N,L}(x::NTuple{L,T}) where {S,T,N,L}
        new{S,T,N,L}(x)
    end

    function MArrayLite{S,T,N,L}(::UndefInitializer) where {S,T,N,L}
        new{S,T,N,L}()
    end
end

Base.length(::MArrayLite{<:Any, <:Any, <:Any, L}) where {L} = L
Base.size(::MArrayLite{Tuple{}}) = ()
Base.size(::MArrayLite{Tuple{S1}}) where S1 = (S1,)
Base.size(::MArrayLite{Tuple{S1, S2}}) where {S1, S2} = (S1, S2)
Base.axes(::MArrayLite{Tuple{}}) = ()
Base.axes(::MArrayLite{Tuple{S1}}) where S1 = (Base.OneTo(S1),)
Base.axes(::MArrayLite{Tuple{S1, S2}}) where {S1, S2} = (Base.OneTo(S1), Base.OneTo(S2))
Base.IndexStyle(::MArrayLite) = Base.IndexLinear()
Freeze.issettable(::MArrayLite) = true

@propagate_inbounds function Base.getindex(a::MArrayLite, i::Int)
    @boundscheck checkbounds(a,i)
    T = eltype(a)

    if isbitstype(T)
        return GC.@preserve a unsafe_load(Base.unsafe_convert(Ptr{T}, pointer_from_objref(a)), i)
    end
    a.data[i]
end

@propagate_inbounds function Base.setindex!(a::MArrayLite, val, i::Int)
    @boundscheck checkbounds(a,i)
    T = eltype(a)

    if isbitstype(T)
        GC.@preserve a unsafe_store!(Base.unsafe_convert(Ptr{T}, pointer_from_objref(a)), convert(T, val), i)
    else
        # This one is unsafe (#27)
        # unsafe_store!(Base.unsafe_convert(Ptr{Ptr{Nothing}}, pointer_from_objref(v.data)), pointer_from_objref(val), i)
        error("setindex!() with non-isbitstype eltype is not supported by StaticArrays. Consider using SizedArray.")
    end

    return val
end

Freeze.thaw(a::SArrayLite{S,T,N,L}) where {S,T,N,L} = MArrayLite{S,T,N,L}(a.data)
Freeze.freeze(a::MArrayLite{S,T,N,L}) where {S,T,N,L} = SArrayLite{S,T,N,L}(a.data)
Base.similar(a::StaticArrayLite, ::Type{T}, sz::Tuple{}) where {T} = MArrayLite{Tuple{},T,0,1}(undef)
@inline Base.similar(a::StaticArrayLite, ::Type{T}, sz::Tuple{Int}) where {T} = MArrayLite{Tuple{sz[1]},T,1,sz[1]}(undef)
@inline Base.similar(a::StaticArrayLite, ::Type{T}, sz::Tuple{Int,Int}) where {T} = MArrayLite{Tuple{sz[1], sz[2]},T,2,sz[1]*sz[2]}(undef)

end # module
