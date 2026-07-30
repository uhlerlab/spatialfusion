import argparse
import gc
import gzip
import logging
import pathlib as pl
import warnings

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

numba_logger = logging.getLogger("numba")
numba_logger.setLevel(logging.WARNING)


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_symbol_to_ensembl_map(gtf_path: pl.Path) -> dict[str, str]:
    logging.info("Parsing GTF gene annotations from %s", gtf_path)
    records = []

    with gzip.open(gtf_path, "rt") as gtf:
        for line in gtf:
            if line.startswith("#"):
                continue

            fields = line.strip().split("\t")
            if len(fields) < 9 or fields[2] != "gene":
                continue

            attr_dict = {}
            for item in fields[8].split(";"):
                item = item.strip()
                if not item:
                    continue
                key, value = item.split(" ", 1)
                attr_dict[key] = value.strip('"')

            gene_id = attr_dict.get("gene_id")
            gene_name = attr_dict.get("gene_name")
            if gene_id and gene_name:
                records.append((gene_name.upper(), gene_id.split(".")[0]))

    mapping_df = pd.DataFrame(records, columns=["gene_symbol", "ensembl_id"]).drop_duplicates()
    symbol_to_ens = mapping_df.set_index("gene_symbol")["ensembl_id"].to_dict()
    logging.info("Parsed %s gene symbol to Ensembl ID mappings", len(symbol_to_ens))
    return symbol_to_ens


def remove_control_probes(adata):
    mask = ~adata.var_names.str.upper().str.startswith(
        (
            "BLANK_",
            "NEGCONTROLCODEWORD",
            "NEGCONTROLPROBE",
            "ANTISENSE_",
        )
    )
    return adata[:, mask].copy()


def map_genes_to_ensembl(adata, symbol_to_ens: dict[str, str]):
    adata.var_names = adata.var_names.str.strip().str.upper()
    adata.var_names_make_unique()
    adata.var["ensembl_id"] = [symbol_to_ens.get(gene, None) for gene in adata.var_names]

    mapped = adata.var["ensembl_id"].notnull().sum()
    logging.info("Mapped genes: %s/%s", mapped, adata.n_vars)

    adata = adata[:, adata.var["ensembl_id"].notnull()].copy()
    adata.var_names = adata.var["ensembl_id"].astype(str)
    adata.var_names_make_unique()
    return adata


def align_to_vocab(adata, vocab):
    vocab = vocab[:, vocab.var_names.str.startswith("ENSG")].copy()
    common_genes = vocab.var_names.intersection(adata.var_names)
    logging.info("Common genes: %s", len(common_genes))

    adata = adata[:, common_genes].copy()
    ordered_genes = [gene for gene in vocab.var_names if gene in adata.var_names]
    adata = adata[:, ordered_genes].copy()
    return adata, ordered_genes, vocab


def align_technology_mean(
    ordered_genes: list[str],
    vocab,
    technology_mean_path: pl.Path,
) -> np.ndarray:
    technology_mean_full = np.load(technology_mean_path)
    technology_mean_map = dict(zip(vocab.var_names, technology_mean_full))
    technology_mean = np.array([technology_mean_map[gene] for gene in ordered_genes])
    return technology_mean.astype(np.float32)


def add_nicheformer_metadata(
    adata,
    split: str,
    modality: int,
    species: int,
    assay: int,
):
    adata.obs["modality"] = modality
    adata.obs["species"] = species
    adata.obs["assay"] = assay

    if "nicheformer_split" not in adata.obs.columns:
        adata.obs["nicheformer_split"] = split

    return adata


def run_embed_nicheformer(
    dataset_path: str,
    output_dir: str,
    model_path: str,
    vocab_path: str,
    technology_mean_path: str,
    gtf_path: str,
    output_name: str = "nicheformer.parquet",
    batch_size: int = 16,
    max_seq_len: int = 1500,
    aux_tokens: int = 30,
    chunk_size: int = 1000,
    num_workers: int = 0,
    embedding_layer: int = -1,
    seed: int = 42,
    split: str = "train",
    modality: int = 4,
    species: int = 5,
    assay: int = 9,
    device: str = "cuda",
) -> None:
    import nicheformer

    set_seed(seed)
    symbol_to_ens = build_symbol_to_ensembl_map(pl.Path(gtf_path))

    logging.info("Loading AnnData from %s", dataset_path)
    adata = ad.read_h5ad(dataset_path)
    logging.info("Original shape: %s", adata.shape)

    cell_ids = (
        adata.obs["cell_id"].astype(str)
        if "cell_id" in adata.obs.columns
        else adata.obs_names.astype(str)
    )

    adata = remove_control_probes(adata)
    logging.info("After removing controls: %s", adata.shape)

    adata = map_genes_to_ensembl(adata, symbol_to_ens)
    logging.info("After mapping: %s", adata.shape)

    vocab = sc.read_h5ad(vocab_path)
    adata, ordered_genes, vocab = align_to_vocab(adata, vocab)
    logging.info("Final aligned shape: %s", adata.shape)

    technology_mean = align_technology_mean(ordered_genes, vocab, pl.Path(technology_mean_path))
    adata.X = adata.X.astype(np.float32)
    adata = add_nicheformer_metadata(
        adata,
        split=split,
        modality=modality,
        species=species,
        assay=assay,
    )

    dataset = nicheformer.data.NicheformerDataset(
        adata=adata,
        technology_mean=technology_mean,
        split=split,
        max_seq_len=max_seq_len,
        aux_tokens=aux_tokens,
        chunk_size=chunk_size,
        metadata_fields={"obs": ["modality", "species", "assay"]},
    )
    logging.info("Token shape: %s", dataset.tokens.shape)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    model = nicheformer.models.Nicheformer.load_from_checkpoint(
        checkpoint_path=model_path,
        strict=False,
    )
    model.eval().to(device)

    embeddings = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            emb = model.get_embeddings(batch=batch, layer=embedding_layer)
            embeddings.append(emb.cpu().numpy())
            gc.collect()

    embeddings = np.concatenate(embeddings, axis=0)
    out = pl.Path(output_dir) / output_name
    pd.DataFrame(embeddings, index=cell_ids).to_parquet(out)
    logging.info("Saved embeddings to %s", out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Nicheformer embedding extraction for SpatialFusion.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--adata", type=pl.Path, required=True, help="Input AnnData (.h5ad).")
    parser.add_argument(
        "--output-dir",
        type=pl.Path,
        default=pl.Path.cwd(),
        help="Directory where the output parquet file is written.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output parquet file if it exists.",
    )
    parser.add_argument(
        "--nicheformer-weights",
        type=pl.Path,
        required=True,
        help="Path to the Nicheformer checkpoint file, usually `nicheformer.ckpt`.",
    )
    parser.add_argument(
        "--nicheformer-vocab",
        type=pl.Path,
        required=True,
        help="Path to the Nicheformer vocabulary AnnData file, usually `model.h5ad`.",
    )
    parser.add_argument(
        "--nicheformer-technology-mean",
        type=pl.Path,
        required=True,
        help="Path to the Nicheformer technology mean NumPy file.",
    )
    parser.add_argument(
        "--gtf",
        type=pl.Path,
        required=True,
        help="GTF annotation file used to map gene symbols to Ensembl IDs.",
    )
    parser.add_argument(
        "--nicheformer-output-name",
        type=str,
        default="nicheformer.parquet",
        help="Output filename for Nicheformer embeddings.",
    )
    parser.add_argument(
        "--nicheformer-batch-size",
        type=int,
        default=16,
        help="Batch size passed to Nicheformer inference.",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=1500,
        help="Maximum sequence length passed to NicheformerDataset.",
    )
    parser.add_argument(
        "--aux-tokens",
        type=int,
        default=30,
        help="Number of auxiliary tokens passed to NicheformerDataset.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Chunk size passed to NicheformerDataset.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of worker processes for the Nicheformer dataloader.",
    )
    parser.add_argument(
        "--embedding-layer",
        type=int,
        default=-1,
        help="Nicheformer layer used for embedding extraction.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split label used by NicheformerDataset.",
    )
    parser.add_argument(
        "--modality",
        type=int,
        default=4,
        help="Nicheformer metadata code for modality.",
    )
    parser.add_argument(
        "--species",
        type=int,
        default=5,
        help="Nicheformer metadata code for species.",
    )
    parser.add_argument(
        "--assay",
        type=int,
        default=9,
        help="Nicheformer metadata code for assay.",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device used for model inference.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    out = args.output_dir / args.nicheformer_output_name
    if out.exists() and not args.overwrite:
        raise FileExistsError(f"{out} exists. Use --overwrite to replace it.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    run_embed_nicheformer(
        dataset_path=str(args.adata),
        output_dir=str(args.output_dir),
        model_path=str(args.nicheformer_weights),
        vocab_path=str(args.nicheformer_vocab),
        technology_mean_path=str(args.nicheformer_technology_mean),
        gtf_path=str(args.gtf),
        output_name=args.nicheformer_output_name,
        batch_size=args.nicheformer_batch_size,
        max_seq_len=args.max_seq_len,
        aux_tokens=args.aux_tokens,
        chunk_size=args.chunk_size,
        num_workers=args.num_workers,
        embedding_layer=args.embedding_layer,
        seed=args.seed,
        split=args.split,
        modality=args.modality,
        species=args.species,
        assay=args.assay,
        device=args.device,
    )


if __name__ == "__main__":
    main()
