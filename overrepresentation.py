import logging
from golib.core.gene_ontology import GeneOntology
from rich.progress import track
import scipy.stats as stats
import pandas as pd

logger = logging.getLogger("overrepresentation")
logger.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S"
)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
logger.addHandler(ch)


def read_complexes(path, max_group_size):
    complexes = {}
    with open(path) as f:
        for line in f:
            if not line.startswith("Clu"):
                fields = line.strip().split(",")
                if int(fields[1]) <= max_group_size:
                    complexes[int(fields[0])] = (
                        fields[len(fields) - 1].replace('"', "").split()
                    )
    return complexes


def get_proteins_from_fasta_file(path):
    proteins = []
    with open(path) as f:
        for line in f:
            if line[0] == ">":
                proteins.append(line.strip().split("|")[1])
    return proteins


def get_proteins_in_complexes(complexes):
    proteins = set()
    for _, prots in complexes.items():
        proteins |= set(prots)
    return list(proteins)


def run(
    proteome_file,
    complexes_file,
    goa_file,
    obo_file,
    out_file,
    pvalue_tau=0.05,
    min_group_count=1,
    max_group_size=100,
):
    logger.info(f"Parsing proteome fasta file {proteome_file}...")
    background = get_proteins_from_fasta_file(proteome_file)
    total_background = len(background)

    logger.info("Building Ontology in memory...")
    go = GeneOntology(obo=obo_file)
    go.build_ontology()

    logger.info("Processing Complexes file...")
    complexes = read_complexes(complexes_file, max_group_size)
    complexes_prots = get_proteins_in_complexes(complexes)
    num_complexes = len(complexes)

    logger.info("Loading GO annotations for this proteome...")
    go.load_gaf_file(goa_file, "overrep")
    go.up_propagate_annotations("overrep")
    annotations = go.annotations("overrep")

    logger.info("Building unified annotations matrix...")
    all_prots = set(background) | set(complexes_prots)
    bg_cond = annotations["Protein"].isin(all_prots)
    table = (
        annotations[bg_cond]
        .pivot(index="GO ID", columns="Protein", values="Score")
        .fillna(0)
    )
    table_prots = table.columns.values
    num_hypotheses = table.shape[0]

    # the background is shared for all complexes,
    # so we can pre-calculate the counts
    annotated_bg = list(set(background) & set(table_prots))
    bg_counts = table[annotated_bg].sum(axis=1)
    # this is more than anything a sanity check, should not change the value
    bg_counts = bg_counts[bg_counts > 0]
    tot_minus_bg_counts = total_background - bg_counts

    logger.info(f"Found {num_complexes} complexes," " analyzing overrepresentation")
    overrepresented_goterms = []
    for i, (complex_id, proteins) in track(
        enumerate(complexes.items()), description="Analyzing...", total=num_complexes
    ):
        perc = i / len(complexes)
        logger.info(
            f"Analyzing complex {i}/{len(complexes)}" f" ({perc * 100.0:.2f}%)) ..."
        )
        total_group = len(proteins)
        annotated_gr_prots = list(set(proteins) & set(table_prots))
        group_counts = table[annotated_gr_prots].sum(axis=1)
        group_counts_idx = group_counts > 0
        group_counts = group_counts[group_counts_idx]
        if group_counts.shape[0] < 1:
            continue
        counts = pd.concat(
            [
                group_counts,
                bg_counts[group_counts_idx],
                total_group - group_counts,
                tot_minus_bg_counts[group_counts_idx],
            ],
            axis=1,
        ).reset_index()
        counts.columns = [
            "GO ID",
            "group_count",
            "bg_count",
            "gr_tot-gr_count",
            "bg_tot-bg_count",
        ]
        # calculate the pvalues
        counts["pvalue"] = counts.apply(
            lambda x: stats.fisher_exact(
                table=[
                    [x["group_count"], x["bg_count"]],
                    [x["gr_tot-gr_count"], x["bg_tot-bg_count"]],
                ],
                alternative="greater",
            )[1],
            axis=1,
        )
        # correct pvalues with the Bonferroni correction
        counts["corrected_pvalue"] = counts["pvalue"] * num_hypotheses
        for _, r in counts[counts["corrected_pvalue"] < pvalue_tau].iterrows():
            overrepresented_goterms.append(
                (complex_id, r["GO ID"], r["corrected_pvalue"])
            )

    logger.info("Writing overrepresentation file...")
    with open(out_file, "w") as out:
        for complex_id, goterm, pvalue in overrepresented_goterms:
            out.write(f"{complex_id}\t{goterm}\t{pvalue}\n")

    logger.info("Done")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="formats the raw output of blast into "
        "the homolog information for ProteoBOOSTER"
    )
    parser.add_argument("proteome_file", help="Path to the proteome fasta file")
    parser.add_argument("complexes_file", help="File with complexes to analyze")
    parser.add_argument("goa_file", help="path to GOA file")
    parser.add_argument("obo_file", help="path to go.obo file")
    parser.add_argument("output_file", help="path to write the results")
    args = parser.parse_args()
    run(
        args.proteome_file,
        args.complexes_file,
        args.goa_file,
        args.obo_file,
        args.output_file,
    )
