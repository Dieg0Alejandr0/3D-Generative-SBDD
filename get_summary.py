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


def get_big_pockets(ligands_dict, threshold):
    
    big_pocket_smiles = []
    for ligand in ligands:
        if ligands[ligand]['atoms'] >= threshold:
            for protein_smile in ligands[ligand]['pockets']:
                big_pocket_smiles.append(protein_smile)
    
    return set(big_pocket_smiles)
    


def get_summary_CD(data_dir):
    
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
                            
                        if 'pockets' not in lig_stats[ligand_smiles]:
                            lig_stats[ligand_smiles]['pockets'] = [protein_smiles]
                        else:
                            lig_stats[ligand_smiles]['pockets'].append(protein_smiles)
                            
                        if 'ligands' not in protein_stats[protein_smiles]:
                            protein_stats[protein_smiles]['ligands'] = [ligand_smiles]
                        else:
                            protein_stats[protein_smiles]['ligands'].append(ligand_smiles)

                            
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
    
    
def get_pathological_stats():
    
    print("WE'RE OFF!")
    samples_stats = {}
    bad_ligands = []
    
    print("GETTING RELEVANT POCKETS")
    pockets = os.listdir('outputs')
    print(pockets)
    
    print("GETTING RELEVANT LIGANDS")
    for pocket in pockets:
        
        print(f"READING PAIR")
        samples = os.listdir(f'outputs/{pocket}/SDF/')
        for sample in samples:

            try:
                supplier = Chem.SDMolSupplier(f'outputs/{pocket}/SDF/{sample}', sanitize=False, removeHs=False)
                mol = supplier[0]
                samples_stats[Chem.MolToSmiles(mol)] = get_mol_stats(mol)

            except:
                bad_ligands.append(sample)

    
    print(f"SAVING RESULTS")
    results = {"stats":samples_stats, "bad_ligands":bad_ligands}  
    print(len(samples_stats))
    torch.save(results, '/data/scratch/draygoza/3D-Generative-SBDD/pathological_stats.pt')
    print(f"WE'RE DONE!")
    
if __name__ == '__main__':
    #get_summary_CD("/data/rsg/nlp/xiangfu/sbdd_data/crossdocked_pocket10")
    
    summary = torch.load('summary_stats.pt')
    ligands = summary['ligands']
    get_pathological_stats()