import Base: +, -, *, /, \

#--------------------------------------------------
# Vector space algebra

# Unary ops
-(a::StaticArrayLite) = map(-, a)

# Binary ops
# Between arrays
+(a::StaticArrayLite, b::StaticArrayLite) = map(+, a, b)
+(a::AbstractArray, b::StaticArrayLite) = map(+, a, b)
+(a::StaticArrayLite, b::AbstractArray) = map(+, a, b)

-(a::StaticArrayLite, b::StaticArrayLite) = map(-, a, b)
-(a::AbstractArray, b::StaticArrayLite) = map(-, a, b)
-(a::StaticArrayLite, b::AbstractArray) = map(-, a, b)

# Scalar-array
*(a::Number, b::StaticArrayLite) = map(x -> a*x, b)
*(a::StaticArrayLite, b::Number) = map(x -> x*b, a)

/(a::StaticArrayLite, b::Number) = map(x -> x/b, a)
\(a::Number, b::StaticArrayLite) = map(x -> a\x, b)

#--------------------------------------------------
# Matrix algebra

# Transpose, conjugate, etc
conj(a::StaticArrayLite) = map(conj, a)

dot(a::StaticArrayLite, b::LinearAlgebra.Adjoint{<:StaticArrayLite}) = a*adjoint(b)
bilinear_vecdot(a::StaticArrayLite, b::StaticArrayLite) = a*transpose(b)

*(a::LinearAlgebra.Transpose{<:StaticArrayLite{<:Number, 1}}, b::StaticArrayLite{<:Number,1}) = mapreduce(*, +, a, b)
*(a::LinearAlgebra.Adjoint{<:StaticArrayLite{<:Real, 1}}, b::StaticArrayLite{<:Number,1}) = mapreduce(*, +, a, b)
*(a::LinearAlgebra.Adjoint{<:StaticArrayLite{<:Complex, 1}}, b::StaticArrayLite{<:Number,1}) = mapreduce((x, y) -> conj(x)*y, +, a, b)

#--------------------------------------------------
# Norms
LinearAlgebra.norm_sqr(a::StaticArrayLite) = mapreduce(abs2, +, a; init=zero(real(eltype(a))))
LinearAlgebra.norm(a::StaticArrayLite) = sqrt(LinearAlgebra.norm_sqr(a))

LinearAlgebra.normalize(a::StaticArrayLite) = inv(norm(a))*a

function LinearAlgebra.normalize!(a::StaticArrayLite)
    c = inv(norm(a))
    map!(x -> c*x, a, a)
    return a
end
