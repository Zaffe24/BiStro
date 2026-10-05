#!/usr/bin/env python3
"""
CLI entry point (`python -m bistro`). Parses the requested subcommand
(preprocess, somatic, sbs96, cosmic) via arglib, validates its inputs via
utilib, and dispatches to that subcommand's own main().
"""
__version__ = "1.0.0"
__author__ = "Pietro Zafferani"


from . import caller, arglib, utilib, sharelib, normcountlib, cosmiclib

def main():
    parser,options = arglib.parse_args(program_version=__version__)
    if options.sub == "preprocess":  # extract candidate mutations from CCS reads
        utilib.check_num_threads(options.threads)
        utilib.check_inputs_preprocess(
            options.bam,
            options.ref,
            options.out_dir,
            options.low_complexity_regions,
        )
        caller.main(
            options.bam,
            options.ref,
            options.out_dir,
            options.sample,
            options.region,
            options.exclude,
            options.threads,
            options.min_mapq,
            options.min_sequence_identity,
            options.min_bq,
            options.min_qlen,
            options.max_qlen,
            options.trim_ends,
            options.indels_window,
            options.mismatch_window_len,
            options.min_ec,
            options.max_softclipping,
            options.z_prob,
            options.check_mem_usage,
            options.do_not_collapse,
            options.min_depth,
            options.low_complexity_regions,
            options.bq_profile_contig,
        )
    elif options.sub == "somatic":  # call true somatic mutations across samples
        utilib.check_num_threads(options.threads)
        utilib.check_inputs_somatic(options.i, options.bams)
        sharelib.main(
            options.i,
            options.thr,
            options.max_muts_per_duplex,
            options.threads,
            options.ref,
            options.bams,
            options.min_mapq,
            options.min_baseq,
            options.min_alt_support,
        )
    elif options.sub == "sbs96":
        utilib.check_num_threads(options.threads)
        utilib.check_inputs_sbs96(options.context, options.muts, options.ref, options.preset_genome)
        normcountlib.main(
            options.context,
            options.muts,
            options.ref,
            options.out,
            options.plot,
            options.sample,
            options.no_plot,
            options.threads,
            options.preset_genome,
        )

    elif options.sub == "cosmic":  # two-tier COSMIC signature attribution across samples
        utilib.check_num_threads(options.threads)
        utilib.check_inputs_cosmic(options.input, options.groups)
        cosmiclib.main(
            options.input,
            options.groups,
            options.out_dir,
            options.genome,
            options.cosmic_version,
            options.column,
            options.nboot,
            options.min_samples,
            options.merge_groups_tier1,
            options.bootstrap_tier2,
            options.poisson,
            options.seed,
            options.threads,
        )

    else:
        print(f"Command 'BiStro {options.sub}' does not exist!\n")
        parser.print_help()
        utilib.exit(1)


if __name__ == "__main__":
    main()
    utilib.exit()