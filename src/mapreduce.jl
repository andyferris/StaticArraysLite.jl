function Base.map(f, a::StaticArrayLite)
    if length(a) == 0
        T = Core.Compiler.return_type(f, Tuple{eltype(a)})
        return freeze(similar(a, T, size(a)))
    end
    @inbounds x = f(a[first(LinearIndices(a))])
    out = similar(a, typeof(x), size(a))
    freeze(_map_widen!(out, 0, x, f, a))
end

function _map_widen!(out, offset, x, f, a)
    a_i1 = first(LinearIndices(a))
    out_i1 = first(LinearIndices(out))
    T = eltype(out)
    while true
        @inbounds out[out_i1+offset] = x
        offset += 1
        if offset >= length(a)
            break
        end
        @inbounds x = f(a[a_i1+offset])
        if !(typeof(x) === T || x isa T)
            T2 = Base.promote_typejoin(T, typeof(x))
            out2 = similar(a, T2, size(a))
            copyto!(out2, out_i1, out, out_i1, offset)
            return _map_widen!(out2, )
        end
    end
    return out
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
