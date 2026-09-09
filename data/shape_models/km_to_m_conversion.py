from stl import mesh

# Učitaj STL
model = mesh.Mesh.from_file("Apophis.stl")

# km -> m
model.vectors *= 1000.0

# Sačuvaj novi STL
model.save("Apophis_m.stl")