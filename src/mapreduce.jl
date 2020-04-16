function Base.map(f, a::StaticArrayLite)
    T = Core.Compiler.return_type(f, Tuple{eltype(a)})
    out = similar(a, T, size(a))
    map!(f, out, a)
    return freeze(out)
end

function Base.map(f, a::StaticArrayLite, b::StaticArrayLite)
    T = Core.Compiler.return_type(f, Tuple{eltype(a), eltype(b)})
    out = similar(a, T, size(a))
    map!(f, out, a, b)
    return freeze(out)
end

function Base.map!(f, out::StaticArrayLite)
    for i in 1:length(out)
        @inbounds out[i] = f()
    end
end

function Base.map!(f, out::StaticArrayLite, a::StaticArrayLite)
    for i in 1:length(out)
        @inbounds out[i] = f(a[i])
    end
end

function Base.map!(f, out::StaticArrayLite, a::StaticArrayLite, b::StaticArrayLite)
    for i in 1:length(out)
        @inbounds out[i] = f(a[i], b[i])
    end
end

function Base.mapreduce(f, op, a::StaticArrayLite; init)
    out = init
    for i in 1:length(a)
        out = op(f(@inbounds(a[i])), out)
    end
    return out
end

function Base.mapreduce(f, op, a::StaticArrayLite, b::StaticArrayLite; init)
    out = init
    for i in 1:length(a)
        out = op(f(@inbounds(a[i]),@inbounds(b[i])), out)
    end
    return out
end