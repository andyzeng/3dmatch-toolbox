// Include file for PDB distance utility



// Distance functions

RNLength PDBDistance(PDBAtom *atom1, PDBAtom *atom2);
RNLength PDBDistance(PDBAtom *atom, PDBResidue *residue);
RNLength PDBDistance(PDBAtom *atom, PDBChain *chain);

RNLength PDBDistance(PDBResidue *residue, PDBAtom *atom);
RNLength PDBDistance(PDBResidue *residue1, PDBResidue *residue2);
RNLength PDBDistance(PDBResidue *residue, PDBChain *chain);

RNLength PDBDistance(PDBChain *chain, PDBAtom *atom);
RNLength PDBDistance(PDBChain *chain, PDBResidue *residue);
RNLength PDBDistance(PDBChain *chain1, PDBChain *chain2);



// Inline functions

inline RNLength
PDBDistance(PDBResidue *residue, PDBAtom *atom)
{
  // Distance is commutative
  return PDBDistance(atom, residue);
}



inline RNLength
PDBDistance(PDBChain *chain, PDBAtom *atom)
{
  // Distance is commutative
  return PDBDistance(atom, chain);
}



inline RNLength
PDBDistance(PDBChain *chain, PDBResidue *residue)
{
  // Distance is commutative
  return PDBDistance(residue, chain);
}



