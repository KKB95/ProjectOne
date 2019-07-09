# Object Oriented Programming in Python
# Basics of OOP in Python:

# lets implement Hashing !!

import hashlib as hl

# list of algorithms guaranteed by the module
hl.algorithms_guaranteed
name = "Danteroom124#"
hashed = hl.sha256(name.encode())
hashed.hexdigest()


class Bird:

    def __init__(self, feature, species, name):
        self.feature = feature
        self.species = species
        self.name = name
