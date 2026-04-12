import os
import argparse
import pathlib
import shutil
import sys
import subprocess
import math
import warnings
from collections import defaultdict
from multiprocessing import Pool, Manager
from itertools import repeat
warnings.filterwarnings("ignore")

import pandas as pd
from ete3 import NCBITaxa
from Bio import SeqIO

from utils.reference_finder import prepare_reference_genomes
from utils.alignment import run_minimap2, run_bwa, sort_samfile
from utils.summary import merge_reference_fasta, call_present_absent
from utils.ani import ani_summary
from utils.input_parsing import parsing_input_f, filter_input_df, get_seq2assembly_dict

import numpy as np
import pyskani
from sklearn.cluster import AgglomerativeClustering
from ast import literal_eval
    
def samtools_calculate_coverage(output_dir, include_supp=False):
    coverage_files = os.path.join(output_dir, "coverage_files")
    bam_files = os.path.join(output_dir, "bam_files")

    command = ["samtools",
               "coverage", "--no-header",
               os.path.join(bam_files, f"merged.sorted.bam")]
    
    if include_supp:
        # samtools coverage --ff UNMAP,QCFAIL,DUP -q 0 merged.sorted.bam > secondary_coverage.tsv
        coverage_file = os.path.join(coverage_files, f"secondary_coverage.tsv")
        command += ['--ff', 'UNMAP,QCFAIL,DUP', '-q', str(1)]
    else:
        # samtools coverage merged.sorted.bam -q 20 > primary_coverage.tsv
        coverage_file = os.path.join(coverage_files, f"primary_coverage.tsv")
        command += ['-q', str(20)]
        # command += ['--ff', 'UNMAP,QCFAIL,DUP', '-q', str(20)]

    subprocess.run(command,
                    check=True,
                    stdout=open(coverage_file, "w"))
    
def _parse_skani_triangle_matrix_file(matrix_path, assemblies):
    """Read skani ``triangle`` Phylip-style full matrix; reorder rows/cols to ``assemblies``."""
    with open(matrix_path) as f:
        lines = f.readlines()
    n = int(lines[0].strip())
    names_order = []
    rows = []
    for line in lines[1:]:
        parts = line.rstrip('\n').split('\t')
        path_or_name = parts[0]
        acc = os.path.splitext(os.path.basename(path_or_name))[0]
        vals = [float(x) for x in parts[1:]]
        if len(vals) != n:
            raise ValueError(
                f'Unexpected skani matrix row width: expected {n}, got {len(vals)}'
            )
        names_order.append(acc)
        rows.append(vals)
    mat = np.array(rows, dtype=np.float64)
    pos = {acc: i for i, acc in enumerate(names_order)}
    missing = set(assemblies) - set(pos.keys())
    if missing:
        raise ValueError(
            f'Skani matrix missing assemblies (check FASTA paths): {missing}'
        )
    idx = [pos[acc] for acc in assemblies]
    return mat[np.ix_(idx, idx)]


def compute_skani_pairwise_matrix_cli_mag(
    reference_genome_path, assemblies, skani_workdir, threads
):
    """
    MAG mode: run ``skani triangle --full-matrix --min-af 0`` (pyskani cannot set min AF).
    Requires the ``skani`` executable on PATH (e.g. conda install -c bioconda skani).
    """
    skani_bin = shutil.which('skani')
    if not skani_bin:
        raise RuntimeError(
            'MAG mode (--include-mag) requires the `skani` executable on PATH so '
            'pairwise ANI can use `--min-af 0`. Install e.g. `conda install -c bioconda skani`.'
        )
    n = len(assemblies)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float64)

    fasta_paths = [
        os.path.abspath(os.path.join(reference_genome_path, f'{acc}.fasta'))
        for acc in assemblies
    ]
    out_base = os.path.join(skani_workdir, 'pairwise_ani')
    cmd = [
        skani_bin,
        'triangle',
        '--full-matrix',
        '--min-af',
        '0',
        '-t',
        str(threads),
        '-o',
        out_base,
    ] + fasta_paths
    stdout_path = os.path.join(skani_workdir, 'skani_cli.stdout')
    stderr_path = os.path.join(skani_workdir, 'skani_cli.stderr')
    with open(stdout_path, 'w') as out, open(stderr_path, 'w') as err:
        subprocess.run(cmd, check=True, stdout=out, stderr=err)
    matrix_path = out_base
    if not os.path.isfile(matrix_path):
        raise FileNotFoundError(
            f'Expected skani ANI matrix at {matrix_path}; check skani_cli.stderr in {skani_workdir}'
        )
    return _parse_skani_triangle_matrix_file(matrix_path, assemblies)


def compute_skani_pairwise_matrix_pyskani(reference_genome_path, assemblies):
    """Build symmetric ANI matrix on a 0–100 scale using skani via pyskani (in-process)."""
    n = len(assemblies)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float64)

    contigs_by_acc = {}
    database = pyskani.Database()
    for acc in assemblies:
        fasta_path = os.path.join(reference_genome_path, f'{acc}.fasta')
        contigs = tuple(bytes(r.seq) for r in SeqIO.parse(fasta_path, 'fasta'))
        contigs_by_acc[acc] = contigs
        database.sketch(acc, *contigs)

    acc_to_idx = {a: i for i, a in enumerate(assemblies)}
    directional = np.zeros((n, n), dtype=np.float64)
    np.fill_diagonal(directional, 100.0)

    for i, acc in enumerate(assemblies):
        hits = database.query(acc, *contigs_by_acc[acc])
        for hit in hits:
            j = acc_to_idx.get(hit.reference_name)
            if j is None or i == j:
                continue
            directional[i, j] = hit.identity

    if n > 1:
        mask = ~np.eye(n, dtype=bool)
        mx = directional[mask].max()
        if mx > 0 and mx <= 1.0:
            directional[mask] *= 100.0

    ani_sym = (directional + directional.T) / 2.0
    np.fill_diagonal(ani_sym, 100.0)
    return ani_sym


def compute_skani_pairwise_matrix(
    reference_genome_path, assemblies, mag_mode=False, threads=1, skani_workdir=None
):
    """
    Pairwise ANI on a 0–100 scale. MAG mode uses ``skani triangle --min-af 0``; otherwise pyskani.
    """
    if mag_mode:
        return compute_skani_pairwise_matrix_cli_mag(
            reference_genome_path, assemblies, skani_workdir, threads
        )
    return compute_skani_pairwise_matrix_pyskani(reference_genome_path, assemblies)


def find_representative_genome(ani_matrix, assembly_ids, downloaded_assemblies):
    dist_nparray = 1 - np.asarray(ani_matrix) / 100.0
    dist_df = pd.DataFrame(dist_nparray.T + dist_nparray,
                 columns=assembly_ids,
                 index=assembly_ids)

    model = AgglomerativeClustering(metric='precomputed', n_clusters=None, compute_full_tree=True,
                                    linkage='complete', 
                                    distance_threshold=0.05).fit(dist_df.values)
    cluster_df = dist_df.copy()
    #print(cluster_df)
    cluster_df['Cluster Label'] = model.labels_
    
    representative_genomes = defaultdict(list)
    member2representative = dict()
    for cluster_idx in cluster_df['Cluster Label'].unique():
        cluster_members = cluster_df[cluster_df['Cluster Label'] == cluster_idx].index.to_list()
        selected_df = downloaded_assemblies[downloaded_assemblies['Assembly Accession ID'].isin(cluster_members)].copy()
        if selected_df[selected_df['Assembly Level'] == 'Complete Genome'].shape[0] == 1:
            representative_genome = selected_df[selected_df['Assembly Level'] == 'Complete Genome']['Assembly Accession ID'].values[0]
        elif selected_df[selected_df['Assembly Level'] == 'Complete Genome'].shape[0] > 1:
            complete_genomes = list(selected_df[selected_df['Assembly Level'] == 'Complete Genome']['Assembly Accession ID'].values)
            representative_genome = dist_df.loc[complete_genomes].sum().idxmin()
        else:
            representative_genome = dist_df.loc[cluster_members].sum().idxmin()
        
        representative_genomes[representative_genome] = cluster_members
        for member in cluster_members:
            member2representative[member] = representative_genome
            
    return representative_genomes, member2representative

def samtools_merged_consensus(output_directory, threads):
    merged_bam = os.path.join(output_directory, 'bam_files', 'merged.sorted.bam')
    subprocess.run(['samtools', 'consensus', 
                    '--show-ins', 'no', 
                    '--show-del', 'yes',
                    '--min-MQ', str(20),
                    '-a',
                    '--mode', "simple",
                    '--threads', str(threads),
                    merged_bam, 
                    '-o', os.path.join(output_directory, 'merged_consensus.fasta')],
                  check=True)
    
    consensus_record_dict = SeqIO.to_dict(SeqIO.parse(os.path.join(output_directory, 'merged_consensus.fasta'), "fasta"))
    return consensus_record_dict

def alignment_summary(downloaded_assemblies, output_directory, seq2assembly_dict, include_supp=True):
    if include_supp:
        coverage_file_name = 'secondary_coverage.tsv'
        column_prefix = 'Secondary'
    else:
        coverage_file_name = 'primary_coverage.tsv'
        column_prefix = 'Primary'
    
    columns = [f'{column_prefix} Breadth',
               f'{column_prefix} Expected',
               f'{column_prefix} Score',
               f'{column_prefix} Depth']
    
    coverage_df = pd.read_csv(os.path.join(output_directory,
                                           'coverage_files',
                                           coverage_file_name),
                              sep='\t',
                              header=None,
                              names=['rname','startpos','endpos','numreads','covbases','coverage','meandepth','meanbaseq','meanmapq'])
    
    taxa_records = defaultdict(lambda: defaultdict(int))
    for idx, row in coverage_df.iterrows():
        taxa_reference = seq2assembly_dict[row['rname']]
        taxa_records[taxa_reference]['genome_length'] += row['endpos']
        taxa_records[taxa_reference]['reads_mapped'] += row['numreads']
        taxa_records[taxa_reference]['genome_totol_count'] += int(row['meandepth'] * row['endpos'])
        taxa_records[taxa_reference]['covbases'] += row['covbases']
    
    breadth_coverage_list = []
    depth_coverage_list = []
    expected_breadth_coverage_list = []
    coverage_score = []
    for assembly_id in downloaded_assemblies['Assembly Accession ID']:
        breadth_coverage, depth_coverage, expected_breadth_coverage = calculate_depth(assembly_id, taxa_records)
        breadth_coverage_list.append(breadth_coverage)
        depth_coverage_list.append(depth_coverage)
        expected_breadth_coverage_list.append(expected_breadth_coverage)
        if expected_breadth_coverage != 0:
            coverage_score.append(min(breadth_coverage/expected_breadth_coverage, 1))
        else:
            coverage_score.append(0)
        
    downloaded_assemblies[columns[0]] = breadth_coverage_list
    downloaded_assemblies[columns[1]] = expected_breadth_coverage_list
    downloaded_assemblies[columns[2]] = coverage_score
    downloaded_assemblies[columns[3]] = depth_coverage_list
    
    downloaded_assemblies.to_csv(os.path.join(output_directory, 'alignment.csv'), index=False)
    
    return downloaded_assemblies

def run_magnet(args):
    
    # Input arguments. 
    input_tsv = args.classification
    input_fastq = args.fastq
    input_fastq2 = args.fastq2
    mode = args.mode
    working_directory = args.output
    taxid_col_idx = args.taxid_idx
    abundance_col_idx = args.abundance_idx
    min_abundance = args.min_abundance
    min_mapq = args.min_mapq
    min_coverage_score = args.min_covscore
    threads = args.threads
    valid_kingdom_str = args.kingdom
    valid_kingdom = set()
    for i in valid_kingdom_str.split(','):
        valid_kingdom.add(int(i))

    if args.include_mag:
        mag_flag = 'all'
    else:
        mag_flag = 'exclude'

    accession_flag = args.accession
    call_subspecies = args.subspecies

    sep = '\t'
    if str(input_tsv)[-3:] == 'csv':
        sep = ','

    ncbi_taxa_db = NCBITaxa()
    if not os.path.exists(working_directory):
        os.mkdir(working_directory)

    if accession_flag:
        abundance_col_idx = None
        min_abundance = 0
        input_df, min_abundance = parsing_input_f(input_tsv, sep, taxid_col_idx, abundance_col_idx, min_abundance)
        valid_taxids = list(input_df['tax_id'].values)
    else:
        input_df, min_abundance = parsing_input_f(input_tsv, sep, taxid_col_idx, abundance_col_idx, min_abundance)
        # make valid_kingdom a variable?
        valid_taxids = filter_input_df(input_df, min_abundance, ncbi_taxa_db, valid_kingdom=valid_kingdom, ret_subspecies=call_subspecies)

    # Get reference genomes
    reference_metadata = prepare_reference_genomes(valid_taxids, working_directory, ncbi_taxa_db, accession_flag=accession_flag, mag_flag=mag_flag)
    downloaded_assemblies = reference_metadata[reference_metadata['Downloaded']]
    reference_genome_path = os.path.join(working_directory, 'reference_genomes')

    # Make skani path if it doesn't exist. 
    skani_path = os.path.join(working_directory, 'skani')
    if not os.path.exists(skani_path):
        os.mkdir(skani_path)

    assemblies = downloaded_assemblies['Assembly Accession ID'].values
    mag_mode = args.include_mag
    # If mag mode need to run skani with a custom flag not supported through the PyWheel package
    # because a fragmented mag could have its ANI surpressed against its own species
    with open(os.path.join(skani_path, "skani.log"), "w") as log_f:
        if mag_mode:
            log_f.write(
                "Computing pairwise ANI with skani CLI: triangle --full-matrix --min-af 0.\n"
            )
        else:
            log_f.write("Computing pairwise ANI with skani (pyskani).\n")
    ani_sym = compute_skani_pairwise_matrix(
        reference_genome_path,
        assemblies,
        mag_mode=mag_mode,
        threads=threads,
        skani_workdir=skani_path,
    )
    with open(os.path.join(skani_path, "skani.log"), "a") as log_f:
        log_f.write(f"Done. Matrix shape {ani_sym.shape}.\n")

    representative_genomes, member2representative = find_representative_genome(
        ani_sym, assemblies, downloaded_assemblies)

    representative_labels = []
    cluster_members = []
    for idx, row in downloaded_assemblies.iterrows():
        accession = row['Assembly Accession ID']
        if accession in representative_genomes.keys():
            representative_labels.append(True)
            cluster_members.append(','.join(representative_genomes[accession]))
        else:
            representative_labels.append(False)
            cluster_members.append(','.join(representative_genomes[member2representative[accession]]))
    downloaded_assemblies['Cluster Representative'] = representative_labels
    downloaded_assemblies['Cluster Members'] = cluster_members

    representative_df = downloaded_assemblies[downloaded_assemblies['Cluster Representative']]

    seq2assembly_dict = get_seq2assembly_dict(working_directory, representative_df)
    reference_fasta = merge_reference_fasta(list(representative_df['Assembly Accession ID']), working_directory)

    if mode == 'ont':
        aligner_output = run_minimap2(input_fastq, reference_fasta, 'merged', working_directory, threads=threads)
    if mode == 'illumina':
        aligner_output = run_bowtie2(input_fastq, input_fastq2, reference_fasta, 'merged', working_directory, threads=threads)

    sort_samfile('merged', aligner_output, working_directory, min_mapq=0, threads=threads)
    coverage_files = os.path.join(working_directory, "coverage_files")
    if not os.path.exists(coverage_files):
        os.mkdir(coverage_files)

    pool = Pool(processes=threads)
    pool.starmap(samtools_calculate_coverage, zip(repeat(working_directory), [True, False]))
    pool.close()
    pool.join()


    representative_df = alignment_summary(representative_df,
                                          working_directory,
                                          seq2assembly_dict,
                                          include_supp=True)

    representative_df = alignment_summary(representative_df,
                                          working_directory,
                                          seq2assembly_dict,
                                          include_supp=False)

    consensus_record_dict = samtools_merged_consensus(working_directory, threads)
    representative_df = ani_summary(representative_df, consensus_record_dict, working_directory, threads)
    representative_df = call_present_absent(representative_df, min_coverage_score)
    representative_df.sort_values(['Primary Score'], ascending=False).to_csv(os.path.join(working_directory, 
                                                                                          'cluster_representative.csv'), 
                                                                             index=False)
    
def get_expected_coverage(genome_length, reads_mapped, genome_totol_count):
    mean_mapping_length = genome_totol_count/reads_mapped
    
    N = genome_length/mean_mapping_length
    x = reads_mapped
    
    expected_M = N*(1-((1-1/N)**x))
    variance = N*((1-1/N)**x) + (N**2)*(1-1/N)*((1-2/N)**x)-(N**2)*((1-1/N)**(2*x))
    
    expected_coverage = expected_M/N
    try:
        std = math.sqrt(variance)
    except ValueError:
        std = 0
    return expected_coverage, std

def calculate_depth(assembly_id, taxa_records):
    genome_length = taxa_records[assembly_id]['genome_length']
    covbases = taxa_records[assembly_id]['covbases']
    genome_totol_count = taxa_records[assembly_id]['genome_totol_count']
    reads_mapped = taxa_records[assembly_id]['reads_mapped']
    
    if genome_totol_count == 0 or reads_mapped == 0:
        breadth_coverage = 0
        depth_coverage = 0
        expected_breadth_coverage = 0
    else:
        breadth_coverage = covbases/genome_length
        depth_coverage = genome_totol_count/covbases
        expected_breadth_coverage, std = get_expected_coverage(genome_length, reads_mapped, genome_totol_count)
    
    return breadth_coverage, depth_coverage, expected_breadth_coverage
def main():
    parser = argparse.ArgumentParser(description="Universal Taxonomic Classification Verifier.")
    parser.add_argument(
        "-c",
        "--classification",
        type=pathlib.Path,
        required=True,
        help="Path to the Taxonomic Classification Report. Accepting csv/tsv file format, other text formats are treated as tsv.",
    )
    parser.add_argument(
        "-i",
        "--fastq",
        type=pathlib.Path,
        required=True,
        help="Path to the first fastq file.",
    )
    parser.add_argument(
        "-I",
        "--fastq2",
        type=pathlib.Path,
        required=False,
        help="Path to the second fastq file for paired-end reads.",
    )
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        required=False,
        choices=["ont", "illumina"],
        help="Modes for different sequencing platforms [ont, illumina]. Default:[ont]",
        default="ont",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=pathlib.Path,
        required=True,
        help="Path to the output directory.",
    )
    parser.add_argument(
        "-t",
        "--taxid-idx",
        type=int,
        required=False,
        help="The column index (0-based) of the taxids. Default:[0]",
        default=0,
    )
    parser.add_argument(
        "-a",
        "--abundance-idx",
        type=int,
        required=False,
        help="The column index (0-based) of the abundance. Default:[None]",
    )
    parser.add_argument(
        "--min-abundance",
        type=float,
        required=False,
        help="Minimum abundance (0-1) for pre-filtering, exclude taxa below the threshold.",
        default=0,
    )
    parser.add_argument(
        "--min-mapq",
        type=int,
        required=False,
        help="Minimum MAPQ for primary alignments. Default:[20]",
        default=20,
    )
    parser.add_argument(
        "--min-covscore",
        type=float,
        required=False,
        help="Minimum Coverage Score for supplementary alignments. Default:[0.7]",
        default=0.7,
    )
    parser.add_argument(
        "--threads",
        type=int,
        required=False,
        help="Number of threads for Multi-threading. Default:[1]",
        default=1,
    )
    parser.add_argument(
        "--kingdom",
        type=str,
        help="A comma separated list of taxids of valid kingdoms. Default:[2,4751,2157,10239]",
        default="2,4751,2157,10239",
    )
    parser.add_argument(
        "--include-mag",
        action="store_true",
        required=False,
        help="Include metagenomic assemble genomes. Default:[off]",
    )
    parser.set_defaults(include_mag=False)
    parser.add_argument(
        "--subspecies",
        action="store_true",
        required=False,
        help="Verify taxonomic classification at subspecies rank. Default:[off]",
    )
    parser.set_defaults(subspecies=False)
    parser.add_argument(
        "--accession",
        action="store_true",
        required=False,
        help="Take accession ids as taxids. Does not work with min-abundance. Default:[off]",
    )
    parser.set_defaults(accession=False)

    args = parser.parse_args()

    run_magnet(args)


if __name__ == "__main__":
    # Make run as a console entry point: `magnet ...`
    main()

