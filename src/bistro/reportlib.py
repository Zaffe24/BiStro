"""
`METRICS`: per-contig read-filtering and coverage counters accumulated while
`BiStro preprocess` processes a contig, plus helpers to merge per-contig
reports into a run-wide summary and derive genome coverage / the max-depth
cutoff.
"""

from dataclasses import dataclass, fields
#import os
import numpy as np
## save stats about reads processed
@dataclass
class METRICS:
    num_reads: int = 0 ## total number of single-dtrand reads processed
    num_zmw: int = 0 ## total number of ZMW-duplexes processed
    num_discarded: int = 0 ## total number of reads failing to pass filtering
    disc_no_primary: int = 0 ## read is flagged as secondary / supplemetary
    disc_min_qlen: int = 0 ## read length shorter than min threshold
    disc_max_qlen: int = 0 ## read length higher tha nmax threshold
    disc_mean_qual: int = 0 ## read mean quality too low
    disc_mapq_qual: int = 0 ## read mapq quality too low
    disc_min_ec: int = 0 ## read not having enough full passes
    disc_max_soft:int = 0 ## proprotion of softclipped read is too high
    disc_min_identity:int = 0 ##seq identity to template is too low

    disc_lack_compl: int = 0 ## sister read of the sam ZMW did not pass filtering

    callable_bps: int = 0 ## base-pairs that were investigated
    total_length_ref: int = 0 ## total length of reference genome
    germ_positions: int = 0
    num_som_ds: int = 0
    num_som_ss: int = 0

    filtering_params: tuple = ()
    chrom_name : str = "" ## for coverage report, to be used in merge_report_coverage
    
    def gp(self,numerator,denominator=1):
        if denominator >0:
            return round(numerator/denominator * 100 , 2)
        return 0

    def compute_tot_bps(self):
        if isinstance(self.callable_bps, int):
            return self.callable_bps
        return sum(self.callable_bps)
    
    def get_chr_name(self):
        if isinstance(self.chrom_name, str):
            return self.chrom_name
        elif isinstance(self.chrom_name, list):
            if len(self.chrom_name) >= 10:
                return f"{self.chrom_name[0]} ... {self.chrom_name[-1]}"
            return " ".join(self.chrom_name)

    def get_coverage(self):
        if isinstance(self.callable_bps, list):
            tot_bps = sum(self.callable_bps)
            tot_length = sum(self.total_length_ref)
            return round(tot_bps / tot_length, 2)
        else:
            return round(self.callable_bps / self.total_length_ref,2)


    def __str__(self):
        
        lines = [f"#STATS FROM MUTATION CALLING STEP:\t{self.get_chr_name()}",
                 f"Total reads processed:\t\t{self.num_reads}",
                 f"Total ZMW duplexes:\t\t{self.num_zmw} ({2 * self.gp(self.num_zmw,self.num_reads)}%)", #each duplex is formed by two reads only
                 "",
                 f"Total reads discarded:\t\t{self.num_discarded} ({self.gp(self.num_discarded,self.num_reads)}%)",
                 f"\tNot-primary reads:\t{self.disc_no_primary} ({self.gp(self.disc_no_primary, self.num_discarded)}%)",
                 f"\tLow Mapping Qual < {self.filtering_params[0]}:\t{self.disc_mapq_qual} ({self.gp(self.disc_mapq_qual,self.num_discarded)}%)",
                 f"\tMean Read Qual < {self.filtering_params[9]}:\t{self.disc_mean_qual} ({self.gp(self.disc_mean_qual, self.num_discarded)}%)",
                 f"\tMean subpasses < {self.filtering_params[8]}:\t{self.disc_min_ec} ({self.gp(self.disc_min_ec, self.num_discarded)}%)",
                 f"\tRead length < {self.filtering_params[3]}:\t{self.disc_min_qlen} ({self.gp(self.disc_min_qlen, self.num_discarded)}%)",
                 f"\tRead length > {self.filtering_params[4]}:\t{self.disc_max_qlen} ({self.gp(self.disc_max_qlen, self.num_discarded)}%)",
                 f"\tSoftclipped > {round(100*self.filtering_params[10])}%:\t{self.disc_max_soft} ({self.gp(self.disc_max_soft, self.num_discarded)}%)",
                 f"\tRead divergence > {round(100* (1 - self.filtering_params[1]))}%:\t{self.disc_min_identity} ({self.gp(self.disc_min_identity, self.num_discarded)}%)",
                 f"\tLack ZMW complement:\t{self.disc_lack_compl} ({self.gp(self.disc_lack_compl, self.num_discarded)}%)",
                 "",
                 f"Total bps called:\t{self.compute_tot_bps()}",
                 f"Genome Coverage:\t{self.get_coverage()}"
                ]
        return "\n".join(lines)



def merge_reports(reports):
    merged = METRICS()
    if not reports:
        return merged
    for f in fields(METRICS):
        if f.name == "filtering_params":      # same for all workers, not summable
            continue
        elif f.name in ("chrom_name", "total_length_ref", "callable_bps"):          # not summable, to be used in coverage report
            setattr(merged, f.name, [getattr(r, f.name) for r in reports])
        else:
            setattr(merged, f.name, sum(getattr(r, f.name) for r in reports))
            
    merged.filtering_params = reports[0].filtering_params
    return merged


def get_coverage(callable_bps, total_length_ref):
    if isinstance(callable_bps, list):
        tot_bps = sum(callable_bps)
        tot_length = sum(total_length_ref)
        return round(tot_bps / tot_length, 2)
    else:
        return round(callable_bps / total_length_ref,2)

def get_maximum_coverage(callable_bps, total_length_ref):
    mean_cov = get_coverage(callable_bps, total_length_ref)
        
    return round( mean_cov * 4, 2)
    #return round( mean_cov + (4 * np.sqrt(mean_cov)), 2)

    ##debugging 
    #return 1000