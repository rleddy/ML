using LinearAlgebra

A = [1 2 3 4; 1 1 7 1; 5 3 3 3; 3 2 3 4]
I = [1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1]

A = Float64.(A)
SaveA = copy(A)
#
println("Starting with Matrix A")
display(A)
println("determinant")
display(det(A))
println("condition number")
display(cond(A))
#
C = copy(I)
C[1,4] = -1
D = copy(I)
D[2,4] = -1
E = copy(I)
E[3,4] = -1
F = E*D*C
#
c = 3
for i in 1:1:4
	aval = A[i,c]
	A[i,:] = (1/aval)*A[i,:]
end
#
println("Next op (1) with Matrix A")
display(A)
println("Matrix F")
display(F)
println("Matrix F*A")
MultFA = F*A
display(MultFA)
println("condition number F*A")
display(cond(MultFA))

G = Float64.([1 0 -1 0; 0 1 -1 0; 0 0 1 0; 0 0 -1 1])

A = copy(SaveA)
c = 3
for i in 1:1:4
	aval = A[i,c]
	A[i,:] = (1/aval)*A[i,:]
end

#
println("Next op (2) with Matrix A")
display(A)
println("Matrix G")
display(G)
MultGA = G*A
println("Matrix G*A")
display(MultGA)
println("condition number G*A")
display(cond(MultGA))

# H is a row 2 pivot
H = copy(I)
H[1,2] = -1
H[3,2] = -1
H[4,2] = -1

A = copy(SaveA)
c = 3
for i in 1:1:4
	aval = A[i,c]
	A[i,:] = (1/aval)*A[i,:]
end

println("Next op (3) with Matrix A")
display(A)
println("Matrix H")
display(H)
println("Matrix H*A")
MultHA = H*A
display(MultHA)
println("condition number H*A")
display(cond(MultHA))

println("Next op (4) with Matrix A -- convert pivot colum to a list of ones")
print("Selecting a pivot at (2,3)\n")

A = copy(SaveA)
println("Matrix A")
c = 3
for i in 1:1:4
	aval = A[i,c]
	A[i,:] = (1/aval)*A[i,:]
end

println("Matrix A after op")
display(A)
println("Matrix H")
display(H)
MultHA = H*A
println("Matrix H*A")
display(MultHA)
println("condition number H*A")
display(cond(MultHA))

#
println("Next op (5) with Matrix A -- use the common pivot ratio for elimination and show condition numbers for different resolutions")
print("\nSelecting a pivot at (2,3)\n")
A = copy(SaveA)
println("Matrix A")
c = 3
p_row = 2
pivot = A[p_row,c]
display(pivot)
for i in 1:4
	if ( i != p_row )
		a = A[i,c]
		A[i,:] = (pivot/a)*A[i,:]
		A[i,:] -= A[p_row,:]
	end
end
#
SaveUpdatedA = copy(A)
println("Matrix A after op and dividing pivot row by pivot")
#
A[p_row,:] = (1/pivot)*A[p_row,:]
display(A)
println("condition number A")
display(cond(A))

A = copy(SaveUpdatedA)
A = A/pivot

println("Matrix A after scaling by 1/pivot")
display(A)
println("condition number A")
display(cond(A))
#
println("\nSolve a systems of equations, using A")
A = copy(SaveA)
println("Matrix A")
display(A)
println("Determinate of A")
display(det(A))
println("Condition number of A")
display(cond(A))

b = Float64.([ 3 2 5 8 ])
save_b = copy(b)

println("Solve given this choice of b")
display(b)
#
x = A/b
println("Resulting in x")
display(x)

println("\nNow scale A and b by the same amount. Pick 3.5 arbitrarily")
A = 3.5*A
b = 3.5*b

println("Matrix A")
display(A)
println("Determinate of A")
display(det(A))
println("Condition number of A")
display(cond(A))
#
println("Solve given this choice of b")
display(b)
x = A/b
println("Resulting in x")
display(x)
#
println("\nNow scale a by 1/pivot that we had before:")
A = copy(SaveA)
b = copy(save_b)  # b was altered previously
pivot = A[p_row,c]
#
A = (1/pivot)*A
b = (1/pivot)*b
println("Matrix A")
display(A)
println("Determinate of A")
display(det(A))
println("Condition number of A")
display(cond(A))
#
println("Solve given this choice of b")
display(b)
x = A/b
println("Resulting in x")
display(x)
