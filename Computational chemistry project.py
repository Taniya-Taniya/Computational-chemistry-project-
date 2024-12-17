# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 00:42:54 2024

@author: TANIYA
"""

import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# 1. Full Periodic Table for Atomic Properties
element_properties = {
    "H": {"atomic_number": 1, "atomic_weight": 1.008, "partial_charge": 0.34, "electronegativity": 2.20},
    "He": {"atomic_number": 2, "atomic_weight": 4.0026, "partial_charge": 0.0, "electronegativity": None},
    "Li": {"atomic_number": 3, "atomic_weight": 6.94, "partial_charge": 0.5, "electronegativity": 0.98},
    "Be": {"atomic_number": 4, "atomic_weight": 9.0122, "partial_charge": 0.0, "electronegativity": 1.57},
    "B": {"atomic_number": 5, "atomic_weight": 10.81, "partial_charge": 0.2, "electronegativity": 2.04},
    "C": {"atomic_number": 6, "atomic_weight": 12.011, "partial_charge": 0.0, "electronegativity": 2.55},
    "N": {"atomic_number": 7, "atomic_weight": 14.007, "partial_charge": -0.3, "electronegativity": 3.04},
    "O": {"atomic_number": 8, "atomic_weight": 15.999, "partial_charge": -0.68, "electronegativity": 3.44},
    "F": {"atomic_number": 9, "atomic_weight": 18.998, "partial_charge": -0.9, "electronegativity": 3.98},
    "Ne": {"atomic_number": 10, "atomic_weight": 20.180, "partial_charge": 0.0, "electronegativity": None},
   "Na": {"atomic_number":11, "atomic_weight":22.9897, "partial_charge":+1, "electronegativity":0.93},
    "Mg": {"atomic_number":12, "atomic_weight":24.305, "partial_charge":+2, "electronegativity":1.31},
    "Al": {"atomic_number":13, "atomic_weight":26.9815, "partial_charge":+3, "electronegativity":1.61},
    "Si": {"atomic_number":14, "atomic_weight":28.0855, "partial_charge":+4, "electronegativity":1.90},
    "P": {"atomic_number":15, "atomic_weight":30.9738, "partial_charge":-3, "electronegativity":2.19},
    "S": {"atomic_number":16, "atomic_weight":32.06, "partial_charge":-2, "electronegativity":2.58},
    "Cl": {"atomic_number":17, "atomic_weight":35.45, "partial_charge":-1, "electronegativity":3.16},
    "Ar": {"atomic_number":18, "atomic_weight":39.948, "partial_charge":0, "electronegativity":None},
    "K": {"atomic_number":19, "atomic_weight":39.0983, "partial_charge":+1, "electronegativity":0.82},
    "Ca": {"atomic_number":20, "atomic_weight":40.078, "partial_charge":+2, "electronegativity":1.00},
    "Sc": {"atomic_number":21, "atomic_weight":44.9559, "partial_charge":"+3", "electronegativity":"1.36"},
    "Ti": {"atomic_number":22, "atomic_weight":47.867, "partial_charge":+4, "electronegativity":1.54},
    "V": {"atomic_number":23, "atomic_weight":50.9415, "partial_charge":+5, "electronegativity":1.63},
    "Cr": {"atomic_number":24, "atomic_weight":51.9961, "partial_charge":+6, "electronegativity":1.66},
    "Mn": {"atomic_number":25, "atomic_weight":54.9380, "partial_charge":+7, "electronegativity":1.55},
    "Fe": {"atomic_number":26, "atomic_weight":55.845, "partial_charge":+3, "electronegativity":1.83},
    "Co": {"atomic_number":27, "atomic_weight":58.9332, "partial_charge":+3, "electronegativity":1.88},
    "Ni": {"atomic_number":28, "atomic_weight":58.6934, "partial_charge":+2, "electronegativity":1.91},
    "Cu": {"atomic_number":29, "atomic_weight":63.546, "partial_charge":+2, "electronegativity":1.90},
    "Zn": {"atomic_number":30, "atomic_weight":65.38, "partial_charge":+2, "electronegativity":1.65},
    "Ga": {"atomic_number":31, "atomic_weight":69.723, "partial_charge":+3, "electronegativity":1.81},
    "Ge": {"atomic_number":32, "atomic_weight":72.630, "partial_charge":+4, "electronegativity":2.01},
    "As": {"atomic_number":33, "atomic_weight":74.9216, "partial_charge":-3, "electronegativity":2.18},
    "Se": {"atomic_number":34, "atomic_weight":78.971, "partial_charge":-2, "electronegativity":2.55},
    "Br": {"atomic_number":35, "atomic_weight":79.904, "partial_charge":-1, "electronegativity":2.96},
    "Kr": {"atomic_number":36, "atomic_weight":83.798, "partial_charge":0, "electronegativity":None},
    "Rb": {"atomic_number":37, "atomic_weight":85.4678, "partial_charge":+1, "electronegativity":0.82},
    "Sr": {"atomic_number":38, "atomic_weight":87.62, "partial_charge":+2, "electronegativity":0.95},
    "Y": {"atomic_number":39, "atomic_weight":88.9059, "partial_charge":+3, "electronegativity":1.22},
    "Zr": {"atomic_number":40, "atomic_weight":91.224, "partial_charge":+4, "electronegativity":1.33},
    "Nb": {"atomic_number":41, "atomic_weight":92.90637, "partia3l_charge":+5, "electronegativity":1.60},
   "Mo": {"atomic_number":42, "atomic_weight":95.95, "partial_charge":+6, "electronegativity":2.16},
   "Tc": {"atomic_number":43, "atomic_weight":98, "partial_charge":+7, "electronegativity":1.9},
   "Ru": {"atomic_number":44, "atomic_weight":101.07, "partial_charge":+4, "electronegativity":2.2},
   "Rh": {"atomic_number":45, "atomic_weight":102.90550, "partial_charge":+3, "electronegativity":2.28},
   "Pd": {"atomic_number":46, "atomic_weight":106.42, "partial_charge":+2, "electronegativity":2.20},
   "Ag": {"atomic_number":47, "atomic_weight":107.8682, "partial_charge":+1, "electronegativity":1.93},
   "Cd": {"atomic_number":48, "atomic_weight":112.414, "partial_charge":+2, "electronegativity":1.69},
   "In": {"atomic_number":49, "atomic_weight":114.818, "partial_charge":+3, "electronegativity":1.78},
   "Sn": {"atomic_number":50, "atomic_weight":118.710, "partial_charge":+4, "electronegativity":1.96},
   "Sb": {"atomic_number":51, "atomic_weight":121.760, "partial_charge":+3, "electronegativity":2.05},
    "Te": {"atomic_number":52, "atomic_weight":127.60, "partial_charge":-2, "electronegativity":2.1},
    "I": {"atomic_number":53, "atomic_weight":126.90447, "partial_charge":-1, "electronegativity":2.66},
    "Xe": {"atomic_number":54, "atomic_weight":131.293, "partial_charge":0, "electronegativity":2.6},
    "Cs": {"atomic_number":55, "atomic_weight":132.9054519, "partial_charge":+1, "electronegativity":0.79},
    "Ba": {"atomic_number":56, "atomic_weight":137.327, "partial_charge":+2, "electronegativity":0.89},
    "La": {"atomic_number":57, "atomic_weight":138.90547, "partial_charge":+3, "electronegativity":1.1},
    "Ce": {"atomic_number":58, "atomic_weight":140.116, "partial_charge":+4, "electronegativity":1.12},
    "Pr": {"atomic_number":59, "atomic_weight":140.90766, "partial_charge":+3, "electronegativity":None},
    "Nd": {"atomic_number":60, "atomic_weight":144.242, "partial_charge":+3, "electronegativity":1.14},
    "Pm": {"atomic_number":61, "atomic_weight":"145", "partial_charge":"+3", "electronegativity":"None"},
    "Sm": {"atomic_number":62, "atomic_weight":150.36, "partial_charge":"+3", "electronegativity":"None"},
    "Eu": {"atomic_number":63, "atomic_weight":151.964," partial_charge":"+3", "electronegativity":"None"},
    "Gd": {"atomic_number":64, "atomic_weight":157.25,"partial_charge":"+3","electronegativity":"None"}, 
    "Tb": {"atomic_number":65, "atomic_weight":158.925,"partial_charge":"+3","electronegativity":"None"},
    "Dy": {"atomic_number":66, "atomic_weight":162.500, "partial_charge":+3, "electronegativity":1.22},
    "Ho": {"atomic_number":67, "atomic_weight":164.930, "partial_charge":+3, "electronegativity":1.23},
    "Er": {"atomic_number":68, "atomic_weight":167.259, "partial_charge":+3, "electronegativity":1.24},
    "Tm": {"atomic_number":69, "atomic_weight":168.934, "partial_charge":+3, "electronegativity":1.25},
    "Yb": {"atomic_number":70, "atomic_weight":173.04, "partial_charge":+3, "electronegativity":1.10},
    "Lu": {"atomic_number":71, "atomic_weight":174.9668, "partial_charge":+3, "electronegativity":1.27},
    "Hf": {"atomic_number":72, "atomic_weight":178.49, "partial_charge":+4, "electronegativity":1.30},
    "Ta": {"atomic_number":73, "atomic_weight":180.94788, "partial_charge":+5, "electronegativity":1.50},
    "W": {"atomic_number":74, "atomic_weight":183.84, "partial_charge":+6, "electronegativity":2.36},
    "Re": {"atomic_number":75, "atomic_weight":186.207, "partial_charge":"+7", "electronegativity":"1 .9"},
    "Os": {"atomic_number":76, "atomic_weight":190.23," partial_charge":"+4", "electronegativity":"2 .2"},
    "Ir":{" atomic_number ":77," atomic_weight ":192.217," partial_charge ":"+3"," electronegativity ":"2.20"}, 
    'Pt': {' atomic_number ':78,' atomic_weight':195.084,' partial_charge ':'+2',' electronegativity ':2.28}, 
    'Au': {' atomic_number ':79,' atomic_weight ':196.96657,' partial_charge ':'+1 or +3',' electronegativity ':2.54}, 
    'Hg': {' atomic_number ':80,' atomic_weight ':200.592,' partial_charge ':'+1 or +2',' electronegativity ':2.00},
    "Tl": {"atomic_number":81, "atomic_weight":204.3833, "partial_charge":+1, "electronegativity":1.62},
    "Pb": {"atomic_number":82, "atomic_weight":207.2, "partial_charge":+2, "electronegativity":2.33},
    "Bi": {"atomic_number":83, "atomic_weight":208.9804, "partial_charge":+3, "electronegativity":2.02},
    "Po": {"atomic_number":84, "atomic_weight":"[209]", "partial_charge":"+2", "electronegativity":"2.0"},
    "At": {"atomic_number":85, "atomic_weight":"[210]", "partial_charge":-1, "electronegativity":2.2},
    "Rn": {"atomic_number":86, "atomic_weight":"[222]", "partial_charge":0, "electronegativity":None},
    "Fr": {"atomic_number":87, "atomic_weight":"[223]", "partial_charge":"+1", "electronegativity":0.7},
    "Ra": {"atomic_number":88, "atomic_weight":"[226]", "partial_charge":"+2", "electronegativity":0.9},
    "Ac": {"atomic_number":89, "atomic_weight":"[227]", "partial_charge":"+3", "electronegativity":1.1},
    "Th": {"atomic_number":90, "atomic_weight":232.03806, "partial_charge":"+4", "electronegativity":1.3},
    "Pa": {"atomic_number":91, "atomic_weight":231.03588, "partial_charge":"+5", "electronegativity":1.5},
    "U": {"atomic_number":92, "atomic_weight":238.02891, "partial_charge":"+6", "electronegativity":1.38},
    "Np": {"atomic_number":93, "atomic_weight":"[237]", "partial_charge":"+5", "electronegativity":1.36},
    "Pu": {"atomic_number":94, "atomic_weight":"[244]", "partial_charge":"+4", "electronegativity":"None"},
    "Am": {"atomic_number":95, "atomic_weight":"[243]", "partial_charge":"+3", "electronegativity":"None"},
    "Cm": {"atomic_number":96, "atomic_weight":"[247]", "partial_charge":"+3", "electronegativity":"None"},
    'Bk': {' atomic_number ':97,' atomic_weight ':'[247]',' partial_charge ':'+3',' electronegativity ':'None'}, 
    'Cf': {' atomic_number ':98,' atomic_weight ':'[251]',' partial_charge ':'+3',' electronegativity ':'None'}, 
    'Es': {' atomic_number ':99,' atomic_weight ':'[252]',' partial_charge ':'+3',' electronegativity ':'None'}, 
    'Fm': {' atomic_number ':100,' atomic_weight ':'[257]',' partial_charge ':'+3',' electronegativity ':'None'},
    "Md": {"atomic_number":101, "atomic_weight":258, "partial_charge":+3, "electronegativity":None},
    "No": {"atomic_number":102, "atomic_weight":259, "partial_charge":+2, "electronegativity":None},
    "Lr": {"atomic_number":103, "atomic_weight":262, "partial_charge":+3, "electronegativity":None},
    "Rf": {"atomic_number":104, "atomic_weight":261, "partial_charge":+4, "electronegativity":None},
    "Db": {"atomic_number":105, "atomic_weight":262, "partial_charge":+5, "electronegativity":None},
    "Sg": {"atomic_number":106, "atomic_weight":266, "partial_charge":+6, "electronegativity":None},
    "Bh": {"atomic_number":107, "atomic_weight":264, "partial_charge":+7, "electronegativity":None},
    "Hs": {"atomic_number":108, "atomic_weight":277, "partial_charge":+4, "electronegativity":None},
    "Mt": {"atomic_number":109, "atomic_weight":268, "partial_charge":"+1", "electronegativity":"None"},
    "Ds": {"atomic_number":110, "atomic_weight":"271.9", "partial_charge":"+5", "electronegativity":"None"},
    "Rg": {"atomic_number":111, "atomic_weight":"271.8", "partial_charge":"+1", "electronegativity":"None"},
    "Cn": {"atomic_number":112, "atomic_weight":"285", "partial_charge":"+2", "electronegativity":"None"},
    "Nh": {"atomic_number":113," atomic_weight ":"286"," partial_charge ":"+3"," electronegativity ":"None"}, 
    'Fl': {' atomic_number ':114,' atomic_weight ':289,' partial_charge ':'+4',' electronegativity ':'None'}, 
    'Mc': {' atomic_number ':115,' atomic_weight ':290.196,' partial_charge ':'+3',' electronegativity ':'None'}, 
    'Lv': {' atomic_number ':116,' atomic_weight ':293,' partial_charge ':'+3',' electronegativity ':'None'}, 
    'Ts': {' atomic_number ':117,' atomic_weight ':294,' partial_charge ':'-1',' electronegativity ':'Unknown'}, 
    'Og': {' atomic_number ':118,' atomic_weight ':'[294]',' partial_charge ':'Unknown',' electronegativity ':'Unknown'}
           # Add other elements as needed
}

# Function to get molecule input from the user
def get_molecule_input():
    atoms = input("Enter the atoms in the molecule (e.g., H O H for H2O): ").split()
    coordinates = []
    for atom in atoms:
        coord = input(f"Enter the 3D coordinates for {atom} (x y z): ").split()
        coordinates.append([float(coord[0]), float(coord[1]), float(coord[2])])
    coordinates = np.array(coordinates)
    return {"atoms": atoms, "coordinates": coordinates}

# Function to visualize the molecule in 3D
def visualize_molecule(atoms, coordinates):
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=coordinates[:, 0], y=coordinates[:, 1], z=coordinates[:, 2],
        mode='markers+text',
        marker=dict(size=10, color='blue'),
        text=atoms,
        textposition="top center"))
    fig.update_layout(title="Molecule Visualization", scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z'))
    fig.show()

# Function to calculate bond length between two atoms
def bond_length(atom1, atom2):
    return np.linalg.norm(atom1 - atom2)

# Function to calculate bond angle between three atoms
def bond_angle(atom1, atom2, atom3):
    vector1 = atom1 - atom2
    vector2 = atom3 - atom2
    cosine_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    return np.arccos(cosine_angle) * (180 / np.pi)

# Function to calculate dipole moment of a molecule
def dipole_moment(atoms, coordinates):
    dipole = np.zeros(3)
    for i, atom in enumerate(atoms):
        charge = element_properties.get(atom.upper(), {}).get("partial_charge", 0)
        dipole += charge * np.array(coordinates[i])
    return dipole

# Improved electrostatic potential mapping function
def electrostatic_potential(atoms):
    potentials = [element_properties.get(atom.upper(), {}).get("electronegativity", np.random.random()) for atom in atoms]
    return potentials

# Function to visualize partial charge distribution using color mapping
def partial_charge_distribution(atoms, coordinates):
    charges = [element_properties.get(atom.upper(), {}).get("partial_charge", 0) for atom in atoms]
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2], c=charges,
                         cmap='coolwarm', s=100)
    
    ax.set_title('Partial Charge Distribution')
    
    # Add color bar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Partial Charge')
    
    plt.show()

# Function to calculate basic thermodynamic properties based on molecular weight
def thermodynamic_properties(atoms):
    # Example: Calculate based on average atomic weight of the molecule
    molecular_weight = sum(element_properties.get(atom.upper(), {}).get("atomic_weight", 0) for atom in atoms)
    
    # Simplified calculation: Enthalpy and entropy based on molecular weight
    temperature = 298 # in Kelvin (standard conditions)
    
    # Using a simple relation: Enthalpy (J/mol) ~ R * T * MW/100 (arbitrary scaling)
    R = 8.314 # J/(mol*K)
    
    enthalpy = R * temperature * (molecular_weight / 100) # Arbitrary scaling factor for demonstration
    entropy = enthalpy / temperature # Simplified relation
    
    return {"Molecular Weight (g/mol)": molecular_weight,
            "Enthalpy (J/mol)": enthalpy,
            "Entropy (J/mol·K)": entropy}

# Improved molecular dynamics simulation function
def molecular_dynamics_simulation(coordinates):
    # A simple random walk simulation as an example of molecular dynamics
    displacement = np.random.normal(0, 0.01, coordinates.shape)
    new_coordinates = coordinates + displacement
    return new_coordinates

# Main function to run the entire program
def main():
   molecule = get_molecule_input()
   
   atoms = molecule["atoms"]
   coordinates = molecule["coordinates"]

   # Visualize Molecule
   visualize_molecule(atoms, coordinates)

   # Bond Length and Angle Calculations
   for i in range(len(coordinates)-1):
       length = bond_length(coordinates[i], coordinates[i+1])
       print(f"Bond Length between Atom {i+1} and Atom {i+2}: {length:.2f} Å")
   
   for i in range(1, len(coordinates)-1):
       angle = bond_angle(coordinates[i-1], coordinates[i], coordinates[i+1])
       print(f"Bond Angle between Atom {i}: {angle:.2f}°")

   # Dipole Moment Calculation
   dipole = dipole_moment(atoms, coordinates)
   print(f"Dipole Moment: {dipole}")

   # Electrostatic Potential Mapping
   potentials = electrostatic_potential(atoms)
   print(f"Electrostatic Potentials: {potentials}")

   # Partial Charge Distribution Visualization
   partial_charge_distribution(atoms, coordinates)

   # Thermodynamic Properties Calculation
   thermo_props = thermodynamic_properties(atoms)
   print(f"Thermodynamic Properties: {thermo_props}")

   # Simulate Molecular Dynamics (optional)
   new_coordinates = molecular_dynamics_simulation(coordinates)
   visualize_molecule(atoms, new_coordinates) # Visualize the new positions after simulation

# Run the program if this is the main module
if __name__ == "__main__":
   main()