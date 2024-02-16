from moleculekit.molecule import Molecule
from moleculekit.util import boundingBox
from moleculekit.tools.preparation import proteinPrepare
from moleculekit.vmdviewer import viewer
import mdtraj as md
# from graph_tool.all import *
import numpy as np
import sys, getopt
# from protlego.structural.clusters import postprocess_session
import logging
import os
import sys, math
import wget
import torch
import os
from Bio.PDB import PDBParser
import logging

# 設定日誌等級為 ERROR，這將只輸出 ERROR 級別的日誌
logging.basicConfig(level=logging.CRITICAL)

group_dict = {}
group_index = 1
print('Beginning Processing ...')
import csv

# if os.path.exists('/hdd/yishan0128/700001/raw/700001.pt'):
#     graph= torch.load('/hdd/yishan0128/700001/raw/700001.pt')
#     print("graph",graph['edge_A'])

proteinNames_ = []
fileList_ = []
broken_file = 0
# Load the file with the list of functions.
classes_ = {}
with open("class_map.txt", 'r') as mFile:
    for line in mFile:
        lineList = line.rstrip().split('\t')
        classes_[lineList[0]] = int(lineList[1])
# print("lineList",lineList)
# print("classes_",classes_)
# Get the file list.
fileList_ = []
cathegories_ = []
p = PDBParser()
name_ = []
debu_dict = {}
molecule_dict = {
    'MET': {'index': 0, 'description': '蛋白質中的甲硫胺酸（Methionine）'},
    'GLY': {'index': 1, 'description': '甘氨酸（Glycine）'},
    'PRO': {'index': 2, 'description': '脯氨酸（Proline）'},
    'ILE': {'index': 3, 'description': '非極性氨基酸（Isoleucine）'},
    'THR': {'index': 4, 'description': '極性氨基酸（Threonine）'},
    'SER': {'index': 5, 'description': '極性氨基酸（Serine）'},
    'TYR': {'index': 6, 'description': '極性氨基酸（Tyrosine）'},
    'VAL': {'index': 7, 'description': '非極性氨基酸（Valine）'},
    'ARG': {'index': 8, 'description': '帶正電氨基酸（Arginine）'},
    'CYS': {'index': 9, 'description': '胱氨酸（Cysteine）'},
    'LEU': {'index': 10, 'description': '非極性氨基酸（Leucine）'},
    'ASN': {'index': 11, 'description': '極性氨基酸（Asparagine）'},
    'LYS': {'index': 12, 'description': '帶正電氨基酸（Lysine）'},
    'ASP': {'index': 13, 'description': '帶負電氨基酸（Aspartic Acid）'},
    'PHE': {'index': 14, 'description': '芳香族氨基酸（Phenylalanine）'},
    'GLN': {'index': 15, 'description': '極性氨基酸（Glutamine）'},
    'GLU': {'index': 16, 'description': '帶負電氨基酸（Glutamic Acid）'},
    'TRP': {'index': 17, 'description': '芳香族氨基酸（Tryptophan）'},
    'ALA': {'index': 18, 'description': '非極性氨基酸（Alanine）'},
    'HIS': {'index': 19, 'description': '組氨酸（Histidine）'},
    'ZN': {'index': 20, 'description': '鋅離子（Zinc Ion）'},
    'SO4': {'index': 21, 'description': '硫酸根離子（Sulfate Ion）'},
    'NA': {'index': 22, 'description': '鈉離子（Sodium Ion）'},
    'CL': {'index': 23, 'description': '氯離子（Chloride Ion）'},
    'CA': {'index': 24, 'description': '鈣離子（Calcium Ion）'},
    'PO4': {'index': 25, 'description': '磷酸根離子（Phosphate Ion）'},
    'K': {'index': 26, 'description': '鉀離子（Potassium Ion）'},
    'ADP': {'index': 27, 'description': '腺苷二磷酸（Adenosine Diphosphate）'},
    'FE': {'index': 28, 'description': '鐵離子（Iron Ion）'},
    'PTR': {'index': 29, 'description': '轉位核糖核酸（Phosphotriester RNA）'},
    'MG': {'index': 30, 'description': ' 鎂離子'},
    'MN': {'index': 31, 'description': ' 錳離子'}
}

# 建立一個索引計數器
index111 = 0




def BondAngle(vector1, vector2):


    dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1] + vector1[2] * vector2[2]
    length1 = (vector1[0]**2+vector1[1]**2+vector1[2]**2)**0.5

    length2 = (vector2[0]**2+vector2[1]**2+vector2[2]**2)**0.5
    angle = 180 - (math.acos(dot_product / (length1 * length2)) * 180/ math.pi)
    try:
        angle = 180 - (math.acos(dot_product / (length1 * length2)) * 180/ math.pi)
    except ZeroDivisionError:
        angle = 180
    # print("angle",angle)
    return angle


from tempfile import gettempdir
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import biotite.sequence as seq
import biotite.structure as struc
import biotite.structure.io as strucio
import biotite.structure.io.mmtf as mmtf
import biotite.database.rcsb as rcsb
import os.path
import biotite.structure.io.pdb as pdb

meow_flag = 0

def detect_disulfide_bonds(structure, distance=2.05, distance_tol=0.05,
                           dihedral=90, dihedral_tol=20):

    disulfide_bonds = []

    sulfide_mask = (structure.res_name == "CYS") & \
                   (structure.atom_name == "SG")

    cell_list = struc.CellList(
        structure,
        cell_size=distance+distance_tol,
        selection=sulfide_mask
    )

    for sulfide_i in np.where(sulfide_mask)[0]:

        potential_bond_partner_indices = cell_list.get_atoms_in_cells(
            coord=structure.coord[sulfide_i]
        )

        for sulfide_j in potential_bond_partner_indices:
            if sulfide_i == sulfide_j:

                continue

            sg1 = structure[sulfide_i]
            sg2 = structure[sulfide_j]

            cb1 = structure[
                (structure.chain_id == sg1.chain_id) &
                (structure.res_id == sg1.res_id) &
                (structure.atom_name == "CB")
            ]
            cb2 = structure[
                (structure.chain_id == sg2.chain_id) &
                (structure.res_id == sg2.res_id) &
                (structure.atom_name == "CB")
            ]
            # Measure distance and dihedral angle and check criteria
            bond_dist = struc.distance(sg1, sg2)

            bond_dihed = np.abs(np.rad2deg(struc.dihedral(cb1[0], sg1, sg2, cb2[0])))

            if np.all(bond_dist > distance - distance_tol) and \
                np.all(bond_dist < distance + distance_tol) and \
                np.all(bond_dihed > dihedral - dihedral_tol) and \
                np.all(bond_dihed < dihedral + dihedral_tol):

                    bond_tuple = sorted((np.array(cb1[0].res_id), np.array(cb2[0].res_id)))

                    # print(sg1.res_id)
                    # Add bond to list of bonds, but each bond only once
                    if bond_tuple not in disulfide_bonds:
                        disulfide_bonds.append(bond_tuple)
        # print("sulfide_mask",sulfide_mask)
    return np.array(disulfide_bonds, dtype=int)
def label(hbond):
    # hbond_label = '%s -%s- %s' % (t.topology.atom(hbond[0]), t.topology.atom(hbond[1]), t.topology.atom(hbond[2]))
    # print("topo",hbond[0],hbond[2])
    # print(t.topology.atom(hbond[0]).residue,t.topology.atom(hbond[2]).residue)
    hbond_label = t.topology.atom(hbond[0]).residue, t.topology.atom(hbond[2]).residue

    return hbond_label
def get_protein_charged(mol):
    lys_n = (mol.resname == "LYS") & (mol.name == "NZ")
    arg_c = (mol.resname == "ARG") & (mol.name == "CZ")
    hip_c = (mol.resname == "HIP") & (mol.name == "CE1")
    pos = lys_n | arg_c | hip_c

    asp_c = (mol.resname == "ASP") & (mol.name == "CG")
    glu_c = (mol.resname == "GLU") & (mol.name == "CD")
    neg = asp_c | glu_c

    return list(np.where(pos)[0]), list(np.where(neg)[0])

def pi_interaction(structure, distance=4.5, distance_tol=1,
                           dihedral=500, dihedral_tol=500):

    disulfide_bonds = []
    sulfide_mask = ((structure.res_name  == "TRP") | 
                (structure.res_name  == "PHE") |
                (structure.res_name == "TYR") |
                (structure.res_name  == "PRO")|
                (structure.res_name  == "LYS")|
                (structure.res_name  == "ARG")) & \
                ((structure.atom_name == "CG") | 
                (structure.atom_name == "CD1") |
                (structure.atom_name == "CD2") |
                (structure.atom_name == "CE1") |
                (structure.atom_name == "CE2") |
                (structure.atom_name == "CZ") |
                (structure.atom_name == "CD")
                )

    pi_indices = structure[sulfide_mask]
    # sulfide_mask = (structure.res_name == "CYS") & \

    cell_list = struc.CellList(
        structure,
        cell_size=distance+distance_tol,
        selection=sulfide_mask
    )
    # sulfide_mask = pi_indices
    ss = structure[sulfide_mask]
    residue_centers = {} 
    for residue in ss:
        # 過濾出當前殘基的所有原子
        atoms_in_residue = ss[ss.res_id == residue.res_id]

        coordinates = atoms_in_residue.coord

        # 計算平均值
        center = np.mean(coordinates, axis=0)

        residue_centers[residue.res_id] = center
        # print("residue.res_id",residue.res_id)


    # print(hoho)
    for sulfide_i in np.where(sulfide_mask)[0]:
        # print(sulfide_i)

        potential_bond_partner_indices = cell_list.get_atoms_in_cells(
            coord=structure.coord[sulfide_i]
        )

        for sulfide_j in potential_bond_partner_indices:

            sg1 = structure[sulfide_i]
            sg2 = structure[sulfide_j]
            if sg1.res_id == sg2.res_id:
                continue            

            cb1 = structure[
                (structure.chain_id == sg1.chain_id) &
                (structure.res_id == sg1.res_id) 

            ]
            cb2 = structure[
                (structure.chain_id == sg2.chain_id) &
                (structure.res_id == sg2.res_id) 
                # ((structure.atom_name == "CZ")| (structure.atom_name == "CZ2")| (structure.atom_name == "CZ3")|(structure.atom_name == "CE2"))
            ]

            bond_dist = struc.distance(residue_centers[sg1.res_id], residue_centers[sg2.res_id])
            # print('c1: ',cb1[0])

            if np.all(bond_dist > distance - distance_tol) and np.all(bond_dist < distance + distance_tol):

                bond_tuple = sorted((np.array(cb1[0].res_id), np.array(cb2[0].res_id)))

                # print(sg1.res_id)
                # Add bond to list of bonds, but each bond only once
                if bond_tuple not in disulfide_bonds:
                    # print(bond_tuple)
                    disulfide_bonds.append(bond_tuple)

    return np.array(disulfide_bonds, dtype=int)
    


def protonate_mol(inputmolfile: str) -> Molecule:

    mol1 = Molecule(inputmolfile)
    # print("mol1: ",mol1)
    # mol2 = mol1

    chains = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P','Q','R','S','T','U','V','W','X','Y','Z']
    # chains = ['D']
    # mol2 = mol1
    mol22 = None
    # print('123chains', chains)
    mol2 = mol1.copy()
    # mol2.filter(f"chain {chain}")  
    mol22,df = proteinPrepare(mol2,store_site=inputmolfile )
    # mol22,df = proteinPrepare(mol2,store_site=inputmolfile )
    df.to_csv(f"{inputmolfile[:-4]}_protonated.csv")

    from moleculekit.atomselect_utils import get_bonded_groups
    from moleculekit.molecule import calculateUniqueBonds
    from moleculekit.bondguesser import guess_bonds_rdkit
    # mol22 = proteinPrepare(mol2)

    # for chain in chains:
    #     try:
    #         # print('123chain1',chain)
    #         mol2 = mol1.copy()
    #         # print('123chain2',chain)
    #         # mol2.filter(f"protein and chain {chain}")

    #         mol22 = proteinPrepare(mol2)
    #         # print('123chain4',chain)
    #         break  # 如果成功处理了某个链，就跳出循环
    #     except:
    #         continue

    if mol22 is None:

        print("无法处理任何链")
        print(filename)
        meow_flag = 1
        return None

    # print(mol1)s
    # mol = proteinPrepare(mol1)
    outputfile = f"{inputmolfile[:-4]}-chainA_protonated.pdb"
    mol22.write(outputfile)
    return mol22



from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP
from Bio.PDB import PDBList
import os
import numpy as np
from collections import deque

from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Polypeptide import three_to_index

## before executed, user should download a version of pdb data from SCOP
## If you would not download pdb file beforehand, then you can use   pdb_dl.retrieve_pdb_file(i, pdir='./', file_format='pdb', overwrite=True)
root = '/hdd/yishan0128/pdbstyle-2.07-2/'
# root ='/hdd/yishan0128/'
# root = '/hdd/yishan0128/pdbstyle-2.07-2/fx/'
from os import path
import warnings
warnings.filterwarnings("ignore")


import pandas as pd
from tqdm import tqdm

bug = 0
terminal=0

n = 1   #used to trace the number of processed data
v = 0   #used to trace the number of vertices
v2_1 = 0
v2_2 = 0

node_l = [] #for storing the node label
node_attr = []
node_attr_2 = []
graph_i = [] #for storing the graph indicator
graph_l = [] #for storing the graph label
edge_A = [] #for storing the edge
edge_A_1 = []
edge_A_2 = []
edge_attr = []
edge_l = []
CA_list = []
N_list = []
C_list = []
C1_list = []
C2_list = []
C3_list = []
C4_list = []
C5_list = []
C6_list = []
dir = ""
v2_x = 1
six = 0
seven = 0
jj = 0
ma = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

cnnnt = 0
flag_2 = 0
file_damage = 0
if meow_flag == 1:
    print(filename)
    print("meow_flag is 1, stopping program execution.")
else:
    with open("training.txt", 'r') as mFile:
        lines = mFile.readlines()
    for curLine in tqdm(lines):
        
        if meow_flag == 1:
            print(filename)
            print(ss)
            break
        splitLine = curLine.rstrip().split('\t')
        curClass = classes_[splitLine[-1]] ###y
        # print("splitLine",splitLine)


        ss = "/hdd/yishan0128/pdbstyle-2.07-2"+"/"+splitLine[0][2]+splitLine[0][3]+"/"+splitLine[0]+".ent"
        if os.path.exists(ss): 
            # print(ss)
            cnnnt+=1
            filename = ss
            if filename.endswith(".ent"): # and splitLine[0] ==  "d1f1oa_":
                
            
                flag = 0

                p = PDBParser()


                structure = p.get_structure(splitLine[0],ss) ###
                # structure = p.get_structure(i, '%s/%s.ent' %(dir, i))
                dir = "/hdd/yishan0128/pdbstyle-2.07-2"+"/"+splitLine[0][2]+splitLine[0][3]
                ik = splitLine[0]
                file1 = pdb.PDBFile.read(os.path.join(dir, f"{ik}.ent")) ###for s bond
                knottin1 = file1.get_structure() #####



                sulfide_indices = np.where(
                    (knottin1.res_name == "CYS") & (knottin1.atom_name == "SG")
                )[0]

                disulfide_bonds = detect_disulfide_bonds(knottin1)
                pi_interaction1 = pi_interaction(knottin1)

                res_name = []
                res_coor = []
                res_coor_CA = []
                res_coor_N = []
                res_coor_C = []
                res_coor_C1 = []
                res_coor_C2 = []
                res_coor_C3 = []
                res_coor_C4 = []
                res_coor_C5 = []
                res_coor_C6 = []
                res_dssp = []
            
                v = 0
                new_adjust = 0
                res_list_b = []
                res_list_a = []
                CA_cnt = 0
                print("ss",ss)
                try:
                    dssp = DSSP(structure[0],ss,file_type='PDB')
                except:
                    pass
                # dssp_dict = dssp[0]

                for chain in structure[0]:

                    for cnt, res in enumerate(chain):
                        ZZ = res.get_id()[2]
                        residue_id = res.get_full_id()
                        # res_list_b.append(int(cnn))
                        # res_list_a.append(int(residue_id[3][1]))
                        if cnt ==0:
                            resseq1 = residue_id[3][1]

                        resseq = residue_id[3][1]

                        # CA_cnt = 0
                        processed_chains = []  
                        CA_processed = []
                        # print("res",res)
                        flag_coor = 0
                        flag_coor_CA = 0
                        flag_coor_C = 0
                        flag_coor_N = 0
                        flag_coor_C1 = 0
                        flag_coor_C2 = 0
                        flag_coor_C3 = 0
                        flag_coor_C4 = 0
                        flag_coor_C5 = 0
                        flag_coor_C6 = 0
                        store_coor = 0
                        for cnn,atom in enumerate(res.get_atoms()):
                        
                            # if(atom.get_id()== 'CA' ):
                            v = v + 1      # record the accumulated number of vertices
                            #     # CA_cnt =CA_cnt+1
                            #     # print("residue_id[3][1]",residue_id)
                            #     # processed_chains.append(int(residue_id[3][1]))
                            #     if(int(residue_id[3][1]) not in processed_chains):
                            #         # res_list_b.append(int(cnn))
                            if res.get_resname()!='HOH':
                                if (int(residue_id[3][1]) not in processed_chains):
                                    # print(chain[0].get_id())
                                    # dssp1 =dssp[chain.get_id(),residue_id[3][1]]

                                    try:
                                        # print("dssp",dssp[splitLine[0][5],residue_id[3][1]])
                                        dssp1 =dssp[chain.get_id(),residue_id[3][1]]
                                    except:
                                        # print()
                                        dssp1 = (0, '>', '>', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
                                    # print(atom)
                                    # if(atom.get_id()== 'CA' ):
                                    #     list_coor_CA = atom.get_coord().tolist()
                                    #     res_coor_CA.append([list_coor_CA],int(residue_id[3][1]))
                                    residue_id = res.get_full_id()
                                    res_name.append(res.get_resname())
                                    list_coor = atom.get_coord().tolist()
                                    list_coor_2 = list_coor
                                    res_list_b.append(int(CA_cnt))
                                    
                                    res_list_a.append(int(residue_id[3][1]))
                                    res_coor.append([list_coor_2,resseq,res.get_resname()])###
                                    store_coor =list_coor
                                    res_dssp.append(dssp1)
                                    new_adjust = len(res_coor)
                                    CA_cnt =CA_cnt+1     
                                    processed_chains.append(int(residue_id[3][1]))
                                    flag_coor =1
                                if(atom.get_id()== 'CA' ):
                                    list_coor_CA = atom.get_coord().tolist()
                                    res_coor_CA.append(list_coor_CA)
                                    flag_coor_CA =1
                                # print(len(res),res.get_atoms())
                                if flag_coor_CA == 0 and cnn == len(res)-1:
                                    # print("store_coor",store_coor)
                                    res_coor_CA.append(store_coor)

                                if(atom.get_id()== 'C' ):
                                    list_coor_C = atom.get_coord().tolist()
                                    res_coor_C.append(list_coor_C)
                                    flag_coor_C =1
                                # print(len(res),res.get_atoms())
                                if flag_coor_C == 0 and cnn == len(res)-1:
                                    res_coor_C.append(store_coor)
                                if(atom.get_id()== 'N' ):
                                    list_coor_N = atom.get_coord().tolist()
                                    res_coor_N.append(list_coor_N)
                                    flag_coor_N =1
                                # print(len(res),res.get_atoms())
                                if flag_coor_N == 0 and cnn == len(res)-1:
                                    res_coor_N.append(store_coor)

                                if((atom.get_id()== 'CG' or atom.get_id()== 'SG' or atom.get_id()== 'OG' or atom.get_id()== 'CG1' or atom.get_id()== 'OG1') and flag_coor_C1 ==0):
                                    list_coor_C1 = atom.get_coord().tolist()
                                    res_coor_C1.append(list_coor_C1)
                                    # print(residue_id)
                                    
                                    flag_coor_C1 =1
                                # print(len(res),res.get_atoms())
                                if flag_coor_C1 == 0 and cnn == len(res)-1:
                                    res_coor_C1.append(store_coor)

                                if((atom.get_id()== 'CD' or atom.get_id()== 'SD' or atom.get_id()== 'CD1' or atom.get_id()== 'OD1' or atom.get_id()== 'ND1') and flag_coor_C2 ==0):
                                    list_coor_C2 = atom.get_coord().tolist()
                                    res_coor_C2.append(list_coor_C2)
                                    flag_coor_C2 =1
                                    # print(residue_id)
                                # print(len(res),res.get_atoms())
                                # print(len(res)-1)
                                if flag_coor_C2 == 0 and cnn == len(res)-1:
                                    res_coor_C2.append(store_coor)
                                if((atom.get_id()== 'CE' or atom.get_id()== 'NE' or atom.get_id()== 'OE1') and flag_coor_C3 ==0):
                                    list_coor_C3 = atom.get_coord().tolist()
                                    res_coor_C3.append(list_coor_C3)
                                    flag_coor_C3 =1
                                if flag_coor_C3 == 0 and cnn == len(res)-1:
                                    res_coor_C3.append(store_coor)
                                if((atom.get_id()== 'CZ' or atom.get_id()== 'NZ') and flag_coor_C4 ==0):
                                    list_coor_C4 = atom.get_coord().tolist()
                                    res_coor_C4.append(list_coor_C4)
                                    flag_coor_C4 =1
                                if flag_coor_C4 == 0 and cnn == len(res)-1:
                                    res_coor_C4.append(store_coor)
                                if((atom.get_id()== 'NH1' ) and flag_coor_C5 ==0):
                                    list_coor_C5 = atom.get_coord().tolist()
                                    res_coor_C5.append(list_coor_C5)
                                    flag_coor_C5 =1
                                if flag_coor_C5 == 0 and cnn == len(res)-1:
                                    res_coor_C5.append(store_coor)

                                if((atom.get_id()== 'CB' ) and flag_coor_C6 ==0):
                                    list_coor_C6 = atom.get_coord().tolist()
                                    res_coor_C6.append(list_coor_C6)
                                    flag_coor_C6 =1
                                if flag_coor_C6 == 0 and cnn == len(res)-1:
                                    res_coor_C6.append(store_coor)

                assert len(res_coor_C) == len(res_coor_CA) == len(res_coor) ==len(res_coor_N) ==len(res_coor_C1) ==len(res_coor_C2) ==len(res_coor_C3) ==len(res_coor_C4) ==len(res_coor_C5) ==len(res_coor_C6)
                for ls in res_coor_CA:
                    # print(ls)
                    CA_list.append(ls)
                for ls in res_coor_N:
                    # print(ls)
                    N_list.append(ls)
                for ls in res_coor_C:
                    # print(ls)
                    C_list.append(ls)
                for ls in res_coor_C1:
                    # print(ls)
                    C1_list.append(ls)
                for ls in res_coor_C2:
                    # print(ls)
                    C2_list.append(ls)
                for ls in res_coor_C3:
                    # print(ls)
                    C3_list.append(ls)
                for ls in res_coor_C4:
                    # print(ls)
                    C4_list.append(ls)

                for ls in res_coor_C5:
                    # print(ls)
                    C5_list.append(ls)
                for ls in res_coor_C6:
                    # print(ls)
                    C6_list.append(ls)

                model = structure[0]

                print("filename",filename + '\n')
                
                # inputfile = dir+'/'+filename
                inputfile = filename
                print("file_dir",inputfile)
                # try:
                hbonds = []
                # _ = protonate_mol(inputfile)
                inputfile_protonated = f"{inputfile[:-4]}-chainA_protonated.pdb"
                result_list = [[0] * 7 for _ in range(len(res_list_b))]
                # t = md.load(inputfile_protonated)
                # hbonds = md.baker_hubbard(t, sidechain_only=False)
                # inputfile_protonated = f"{inputfile[:-4]}-chainA_protonated.pdb"
                # if os.path.exists(inputfile_protonated):
                #     t = md.load(inputfile_protonated)
                #     table, bonds = t.topology.to_dataframe()

                #     hbonds = md.baker_hubbard(t, sidechain_only=False)
                


                # else:
                    # print("#####")
                    # _ = protonate_mol(inputfile,splitLine[0][5])
                    # t = md.load(inputfile_protonated)
                    # # print("tttt",t.topology)
                    # hbonds = md.baker_hubbard(t, sidechain_only=False)
                try:
                    print("#####")
                    _ = protonate_mol(inputfile)
                    t = md.load(inputfile_protonated)
                    # print("tttt",t.topology)
                    hbonds = md.baker_hubbard(t, sidechain_only=False)
                except ValueError as err:
                    if " Too few atoms present" in str(err):
                        print("ValueError")
                        broken_file +=1
                        # 忽略特定错误
                        pass
                print("45345",f'{ss[:-4]}_{ss[-6].upper()}.csv')
                if os.path.exists(f'{ss[:-4]}_{ss[-6].upper()}.csv'):
                    print("yes")
                    with open(f'{ss[:-4]}_{ss[-6].upper()}.csv', 'r') as csv_file:              
                        csv_reader = csv.reader(csv_file)
                        # print("len",len(csv_reader))
                        for row in csv_reader:
                            try:
                                index = res_list_a.index(int(row[1]))
                                index = res_list_b[index]
                                if row[6] not in group_dict:
                                    group_dict[row[6]] = group_index
                                    group_index += 1
                                # if
                                # print(row[3])
                                if row[3] in molecule_dict:
                                    result_list[index][0] = group_dict[row[6]]
                                    result_list[index][1] = row[7]
                                    result_list[index][2] = row[11]
                                    result_list[index][3] = row[12]
                                    result_list[index][4] = row[13]
                                    result_list[index][5] = row[14]
                                    result_list[index][6] = row[15]
                                # print("{} found at index: {}".format(row[1], index))
                            except ValueError:
                                pass
                # inputfile_protonated = f"{inputfile[:-4]}-chainA_protonated.pdb"
                # t = md.load(inputfile_protonated)
                # hbonds = md.baker_hubbard(t, sidechain_only=False)

                # print


                graph_l.append(int(curClass))

                # pattern = r'(\w+)(?\d+)\-w+ = \w+(?\d+)\-w+'
# 
                # pattern = r'(\d+)-\w+\s*=\s*(\d+)-\w+'

                list2 = []

                migu_flag = 0
                # print("hbond",hbonds)
                try:
                    meow = hbonds
                except NameError as err:
                    migu_flag = 1
                if migu_flag==0:
                    for hbond in hbonds:
                        if migu_flag ==1:
                            break
                        k =label(hbond)

                        Num,Num2 =k

                        Num=str(Num)
                        Num2=str(Num2)
                        # print("num",int(Num[3:]),int(Num2[3:]))
                        list2.append([int(Num[3:]),int(Num2[3:])])
                        # print("hbond",hbond) 


                ca_atoms = [res for res in structure.get_residues()]

                add = 0

                num1 = 0

                value = add

                adj_mtr = [[0 for x in range(new_adjust)] for y in range(new_adjust)]       
                adj_mtr_1 = [[0 for x in range(new_adjust)] for y in range(new_adjust)]
                adj_mtr_2 = [[0 for x in range(new_adjust)] for y in range(new_adjust)]
                
                ###################
                ver1 = []
                ver2 = []

                
                for i_d, res_list in enumerate(res_coor):
                    kkk,res_num_2, res_n_2 = res_list
                    left = kkk
                    num2 = 0
                    nn = 0
                    for j_d, res_list_2 in enumerate(res_coor):
                        l, res_num, res_name_1 = res_list_2
                        right = l

                        vector = [float(x) - float(y) for x, y in zip(kkk, l)]

                        dist = (vector[0]**2+vector[1]**2+vector[2]**2)**0.5
                        adj_mtr_2[i_d][j_d] = dist
                        if dist < 8 and dist != 0:


                            adj_mtr[num1][num2] = 3
                            adj_mtr[num2][num1] = 3



                    
                        num2= num2+1
                    num1= num1+1
                v2_x = v2_x + num1



                ver1 = []
                ver2 = []
                # print("res_list_a",res_list_a)
                # print("res_list_b",res_list_b)
                for ii in list2:
                    if (ii[0] == (res_list_a[-1]+1))|(ii[1] == (res_list_a[-1]+1))|(ii[0] == 1)|(ii[1] == 1): ##terminal is easy to lost or first residue sometimes missed
                        continue
                    ver1.append(ii[0]) 
                    ver2.append(ii[1]) 
                    # print("23232",ii[0],ii[1])
                    try:
                        id_1 = res_list_a.index(ii[0])
                        id_1 = res_list_b[id_1]

                        id_2 = res_list_a.index(ii[1])
                        id_2 = res_list_b[id_2]
                    except ValueError as err:
                        if "not in list" in str(err):
                            pass


                    # print("121",id_1,id_2)
                    # adj_mtr[id_1][id_2] = 1
                    try:
                        adj_mtr[id_2][id_1] = 1
                    except:
                        print("23232",ii[1],ii[0])
                    try:
                        adj_mtr[id_1][id_2] = 1
                    except:
                        print("23232",ii[0],ii[1])

                        pass
                    # except:
                    #     print()
                    #     print("error")
                    #     print(ii[0],ii[1])
                    #     continue


                v = 0
                start = 0

                node_angle = [0 for x in range(new_adjust)] 
                for nnnn_a, list_in_list_a in enumerate(res_coor):
                    if nnnn_a + 2 >= len(res_coor):
                        continue
                    vector_a1 = [float(x) - float(y) for x, y in zip(res_coor[nnnn_a+1][0], res_coor[nnnn_a][0])]

                    vector_a2 = [float(x) - float(y) for x, y in zip(res_coor[nnnn_a+2][0], res_coor[nnnn_a+1][0])]
                    new_angle = BondAngle(vector_a1,vector_a2)
                    node_angle[nnnn_a] = new_angle


                for nnnn, list_in_list in enumerate(res_coor):
                    list_in_list_2_ =[]
                    v+=1

                    adj_mtr[v-1][v-2] = 2
                    adj_mtr[v-2][v-1] = 2

                    v2_1 = v2_1 + 1

                    list_in_list_2 = ("{:.2f}".format(node_angle[nnnn]))
                    list_in_list_2 = float(list_in_list_2)
                    list_in_list_2_.append(list_in_list_2)

                    # res_dssp[nnnn]
                    
                    for item in res_dssp[nnnn]:
                        # print("item",nnnn,item)
                        list_in_list_2_.append(item)

                    # print("list_in_list_2_",list_in_list_2_)

                    node_attr.append(list_in_list_2_)
                    # print("compar",res_dssp[nnnn])
                    # print("result_list",result_list[nnnn])
                    node_attr_2.append(result_list[nnnn])



                    debu1 = list_in_list[2]
                    if debu1 in molecule_dict:
                        index111 = molecule_dict[debu1]['index']
                        description = molecule_dict[debu1]['description']
                    # iiii = list_in_list[0]
                    else:
                        # print(f"找不到分子名稱: {debu1}")
                        index111 = 32
                    node_l.append(index111)


                    graph_i.append(n)     #第幾個graph的意思


                
                for sg1_index, sg2_index in disulfide_bonds:
                    # print("sg1_index_s",sg1_index)
                    sg1_index = res_list_a.index(sg1_index)
                    sg1_index = res_list_b[sg1_index]
                    sg2_index = res_list_a.index(sg2_index)
                    sg2_index = res_list_b[sg2_index]
                    # try:
                    adj_mtr[sg1_index][sg2_index] = 4
                    adj_mtr[sg2_index][sg1_index] = 4
                    # except:
                        # print
                        # print(sg1_index,sg2_index,"new_adjust",new_adjust,resseq1,'wrong_sul')
                        # meow_flag =1
                        # continue
                for sg1_index, sg2_index in pi_interaction1:
                    # print("sg1_index",sg1_index)
                    # print("res_list_a",res_list_a)
                    sg1_index = res_list_a.index(sg1_index)
                    sg1_index = res_list_b[sg1_index]
                    # print("res_list_a",res_list_a,sg2_index)
                    sg2_index = res_list_a.index(sg2_index)
                    sg2_index = res_list_b[sg2_index]
                    # try:
                    adj_mtr[sg1_index][sg2_index] = 5
                    adj_mtr[sg2_index][sg1_index] = 5
                    # except:
                    #     print(sg1_index,sg2_index,"new_adjust",new_adjust,resseq1,'wrong_pi')
                    #     meow_flag =1
                        # print('wrong')
                        # continue
    
                n += 1
                
                # print(num1)
                check = 0
                for xi,x in enumerate(adj_mtr):
                    for yi,y in enumerate(x):
                        if y != 0:

                            edge_A_1.append(xi+v2_2+1)
                            edge_A_2.append(yi+v2_2+1)
                            # edge_attr.append(y)
                            edge_attr.append(adj_mtr_2[xi][yi])
                            edge_l.append(y)
                            # print("edge_l",edge_l)
                            check_1 = xi+v2_2+1    #xi這一輪 v2_2從以前到現在
                            ckeck = yi+v2_2+1
                v2_2 = v2_1

                if (len(node_l) == check_1) & (len(node_attr) == check_1) & (len(node_attr_2) == check_1) &(len(C_list) == check_1):
                    continue
                else:
                    meow_flag = 1
                    print('unequal')
                    print(len(node_l))
                    print(check_1)
                    print(filename)
                    print(len(C_list))
                    print(ss)
                    break


        jj= jj+1
        # print("jj: ", jj)
        # if jj > 100:
        #     meow_flag = 1
        #     break; 

edge_A = np.column_stack([ edge_A_1, edge_A_2])
import torch
torch.save({
            'graph_i': graph_i,
            'graph_l': graph_l,
            'edge_l': edge_l,
            'edge_attr': edge_attr,
            'edge_A': edge_A,
            'node_l': node_l,
            'node_attr': node_attr,
            'node_attr_2': node_attr_2,
            }, '/hdd/yishan0128/Fold_class/test_fam/raw/700001.pt')
    
# np.savetxt('7001_node_labels.txt', node_l, delimiter='\n', comments='', fmt='%s')
np.savetxt('test_fam_graph_indicator.txt', graph_i, delimiter='\n', comments='', fmt='%s')
np.savetxt('test_fam_graph_labels.txt', graph_l, delimiter='\n', comments='', fmt='%s')
# np.savetxt('test1w_node1_edge_labels.txt', edge_attr, delimiter='\n', comments='', fmt='%s')


with open('test_fam_edge_labels.txt', 'w') as fp:
    fp.write('\n'.join('{}'.format(x) for x in edge_l))
with open('test_fam_edge_attributes.txt', 'w') as fp:
    fp.write('\n'.join('{}'.format(x) for x in edge_attr))


with open('test_fam_A.txt', 'w') as fp:
    fp.write('\n'.join('{},{}'.format(x[0],x[1]) for x in edge_A))
    
with open('test_fam_node_labels.txt', 'w') as fp:
    fp.write('\n'.join('{}'.format(x) for x in node_l))    
for nnn2,x in enumerate(node_attr):
    for nnn,h in enumerate(x):
        if h == 'NA':
            node_attr[nnn2][nnn] = 0
for nnn3,xm in enumerate(node_attr_2):
    for nnn,hm in enumerate(xm):
        if hm == 'NA':
            node_attr_2[nnn3][nnn] = 0
with open('7000011_node_attributes.txt', 'w') as fp:
    for nnn,x in enumerate(node_attr):
        # print(x)
        fp.write(''.join('{},{},{},{:.2f},{},{},{},{},{},{},{},{},{},{}').format(x[0],ord(x[2]),ord(x[3]),x[4],x[5],x[6],x[7],x[8],x[9],x[10],x[11],x[12],x[13],x[14]))

        if nnn != len(node_attr)-1:
            fp.write('\n')
with open('7000011_CA.txt', 'w') as fp:
    for nnn,x in enumerate(CA_list):
        # print(x) 

        fp.write(''.join('{:.2f},{:.2f},{:.2f}').format(x[0],x[1],x[2]))

        if nnn != len(CA_list)-1:
            fp.write('\n')
with open('7000011_N.txt', 'w') as fp:
    for nnn,x in enumerate(N_list):
        # print(x) 

        fp.write(''.join('{:.2f},{:.2f},{:.2f}').format(x[0],x[1],x[2]))

        if nnn != len(N_list)-1:
            fp.write('\n')
with open('7000011_C.txt', 'w') as fp:
    for nnn,x in enumerate(C_list):
        # print(x) 

        fp.write(''.join('{:.2f},{:.2f},{:.2f}').format(x[0],x[1],x[2]))

        if nnn != len(C_list)-1:
            fp.write('\n')
with open('7000011_C1.txt', 'w') as fp:
    for nnn,x in enumerate(C1_list):
        # print(x) 

        fp.write(''.join('{:.2f},{:.2f},{:.2f}').format(x[0],x[1],x[2]))

        if nnn != len(C1_list)-1:
            fp.write('\n')
with open('7000011_C2.txt', 'w') as fp:
    for nnn,x in enumerate(C2_list):
        # print(x) 

        fp.write(''.join('{:.2f},{:.2f},{:.2f}').format(x[0],x[1],x[2]))

        if nnn != len(C2_list)-1:
            fp.write('\n')
with open('7000011_C3.txt', 'w') as fp:
    for nnn,x in enumerate(C3_list):
        # print(x) 

        fp.write(''.join('{:.2f},{:.2f},{:.2f}').format(x[0],x[1],x[2]))

        if nnn != len(C3_list)-1:
            fp.write('\n')
with open('7000011_C4.txt', 'w') as fp:
    for nnn,x in enumerate(C4_list):
        # print(x) 

        fp.write(''.join('{:.2f},{:.2f},{:.2f}').format(x[0],x[1],x[2]))

        if nnn != len(C4_list)-1:
            fp.write('\n')
with open('7000011_C5.txt', 'w') as fp:
    for nnn,x in enumerate(C5_list):
        # print(x) 

        fp.write(''.join('{:.2f},{:.2f},{:.2f}').format(x[0],x[1],x[2]))

        if nnn != len(C5_list)-1:
            fp.write('\n')
with open('7000011_C6.txt', 'w') as fp:
    for nnn,x in enumerate(C6_list):
        # print(x) 

        fp.write(''.join('{:.2f},{:.2f},{:.2f}').format(x[0],x[1],x[2]))

        if nnn != len(C6_list)-1:
            fp.write('\n')

with open('7000011_node_attributes_2.txt', 'w') as fp:
    for nnn,x in enumerate(node_attr_2):
        # print(x)
        fp.write(''.join('{},{},{},{},{},{},{}').format(x[0],x[1],x[2],x[3],x[4],x[5],x[6]))

        if nnn != len(node_attr_2)-1:
            fp.write('\n')
print(n)
