"""
Builds the (contig, start, end) list that `BiStro preprocess` splits work
over, from the reference FASTA's contig list filtered by --region / --exclude.
"""

import pysam
import numpy as np

def get_contig_list(reference, region, exclude):
    with pysam.FastaFile(reference) as genome:
        y_contigs = process_input(genome, region)
        no_contigs = process_input(genome, exclude)
        contigs = genome.references
        lengths = genome.lengths

    # print(genome.get_reference_length("NT_166280.1"))
        if y_contigs:
            return [(contig,0,genome.get_reference_length(contig)) for contig in y_contigs]
        
        final_list = [(contig, 0, length) for contig, length in zip(contigs, lengths)]
        final_list = [item for item in final_list if item[0] not in no_contigs]
        return final_list



def process_input(genome, input) -> list:
    output=[]
    refs = genome.references
    if not input:
        return output
    if isinstance(input,list):
        contigs = [c.strip() for c in input]
        for c in contigs:
            if c in refs:
                output.append(c)
            else:
                raise ValueError(f"WRONG CONTIG: '{c}' not in FASTA file")
    else:
        raise ValueError(f"WRONG FORMAT: '{input}' must be a list seprated by whitespaces e.g. 'Chr1 Chr2'")
    ## mantain the original order of contigs as in the FASTA
    output_sorted = np.array(refs)[np.isin(refs,output)]
    return output_sorted.tolist()
#                 "NW_023337853.1,NT_187063.1, NT_166452.1")