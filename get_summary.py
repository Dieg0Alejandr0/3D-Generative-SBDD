import os
from rdkit import Chem
from explore_utils import *
import torch

def get_folder_pairs(folder_dir):
    
    folder_files = os.listdir(folder_dir)
    
    pairs = []
    for file in folder_files:
        if file[-3:] == 'pdb':
            pairs.append(file[0:-13])
        else:
            pairs.append(file[0:-4])
    
    return list(set(pairs))


get_mol_stats = lambda mol: {'atoms':mol.GetNumAtoms(), 'bonds':mol.GetNumBonds(), 'conformers':mol.GetNumConformers(), 'heavy':mol.GetNumHeavyAtoms()}
    


def get_summary(data_dir):
    
    print("We're off!")
    folders = os.listdir(data_dir)
    protein_stats = {}
    lig_stats = {}
    bad_proteins = []
    bad_ligands = []
    for folder in folders:
        
        print(f"READING FOLDER:{folder}")
        
        try: 
            folder_dir = f"{data_dir}/{folder}"
            folder_pairs = get_folder_pairs(folder_dir)
            for pair in folder_pairs:

                print(f"READING PAIR:{pair}")
                current_dir = f"{folder_dir}/{pair}"

                try:
                    ligand_mol = read_molecule(f"{current_dir}.sdf")
                    ligand_smiles = Chem.MolToSmiles(ligand_mol)

                    if ligand_smiles not in lig_stats:
                        lig_stats[ligand_smiles] = get_mol_stats(ligand_mol)

                    try:

                        protein_mol = read_molecule(f"{current_dir}_pocket10.pdb")
                        protein_smiles = Chem.MolToSmiles(protein_mol)

                        if protein_smiles not in protein_stats:
                            protein_dict = get_mol_stats(protein_mol)
                            receptor = get_receptor(f"{current_dir}_pocket10.pdb", ligand_mol, 5)
                            protein_dict['residues'] = receptor[2].shape[0]
                            protein_stats[protein_smiles] = protein_dict

                    except:
                        bad_proteins.append(pair)

                except:
                    bad_ligands.append(pair)
        
        except:
            print('BAD FOLDER')
                            
    
    print(f"SAVING RESULTS")
    results = {"pockets":protein_stats, "ligands":lig_stats, "bad_proteins":bad_proteins, "bad_ligands":bad_ligands}     
    torch.save(results, '/data/scratch/draygoza/3D-Generative-SBDD/summary_stats.pt')
    print(f"We're done!")
    
if __name__ == '__main__':
    get_summary("/data/rsg/nlp/xiangfu/sbdd_data/crossdocked_pocket10")