import argparse
import logging
import pathlib as pl
import sys
import warnings

import pandas as pd

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

numba_logger = logging.getLogger("numba")
numba_logger.setLevel(logging.WARNING)

DEFAULT_SCFOUNDATION_DIR = pl.Path("/app/zero-shot-scfoundation")


def run_embed_scgpt(
    dataset_path: str,
    model_dir: str,
    output_dir: str,
    n_hvg: int,
    gene_col: str = "index",
    layer_key: str = "X",
    log_norm: bool = False,
    seed: int = 42,
    max_seq_len: int = 1200,
    batch_size: int = 32,
    input_bins: int = 51,
    model_run: str = "pretrained",
    num_workers: int = 0,
    output_name: str = "scGPT.parquet",
    scfoundation_dir: str | None = None,
) -> None:
    if scfoundation_dir:
        sys.path.append(scfoundation_dir)
    else:
        sys.path.append(str(DEFAULT_SCFOUNDATION_DIR))

    from sc_foundation_evals import data, scgpt_forward
    from sc_foundation_evals.helpers.custom_logging import log

    log.setLevel(logging.INFO)

    scgpt_model = scgpt_forward.scGPT_instance(
        saved_model_path=model_dir,
        model_run=model_run,
        batch_size=batch_size,
        save_dir=output_dir,
        num_workers=num_workers,
        explicit_save_dir=True,
    )

    scgpt_model.create_configs(seed=seed, max_seq_len=max_seq_len, n_bins=input_bins)
    scgpt_model.load_pretrained_model()

    input_data = data.InputData(adata_dataset_path=dataset_path)
    vocab_list = scgpt_model.vocab.get_stoi().keys()

    adata = input_data.adata
    genes_in_vocab = adata.var_names.intersection(vocab_list)
    if len(genes_in_vocab) / len(adata.var_names) < 0.5:
        log.warning("Fewer than 50% of genes are found in the model vocab; continuing anyway.")

    adata._inplace_subset_var(genes_in_vocab)
    input_data.adata = adata

    input_data.preprocess_data(
        gene_vocab=vocab_list,
        model_type="scGPT",
        gene_col=gene_col,
        data_is_raw=not log_norm,
        counts_layer=layer_key,
        n_bins=input_bins,
        n_hvg=n_hvg,
    )

    scgpt_model.tokenize_data(
        data=input_data, input_layer_key="X_binned", include_zero_genes=False
    )
    scgpt_model.extract_embeddings(data=input_data)

    index = (
        input_data.adata.obs["cell_id"]
        if "cell_id" in input_data.adata.obs.columns
        else input_data.adata.obs.index
    )
    pd.DataFrame(input_data.adata.obsm["X_scGPT"], index=index).to_parquet(
        pl.Path(output_dir) / output_name
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run scGPT embedding extraction for SpatialFusion.",
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
        "--scgpt-weights",
        type=pl.Path,
        required=True,
        help="Path to scGPT model directory (expects args/vocab/weights files).",
    )
    parser.add_argument(
        "--scfoundation-dir",
        type=pl.Path,
        default=None,
        help="Path added to PYTHONPATH so `sc_foundation_evals` can be imported.",
    )
    parser.add_argument(
        "--input-is-log-normalized",
        choices=["True", "False"],
        required=True,
        help="Whether the selected layer is already log-normalized.",
    )
    parser.add_argument(
        "--n-hvg",
        type=int,
        default=1200,
        help="Number of highly variable genes used during preprocessing.",
    )
    parser.add_argument(
        "--scgpt-batch-size",
        type=int,
        default=16,
        help="Batch size passed to scGPT inference.",
    )
    parser.add_argument(
        "--gene-col",
        type=str,
        default="index",
        help="Column in `adata.var` containing gene names (or `index`).",
    )
    parser.add_argument(
        "--layer-key",
        type=str,
        default="X",
        help="AnnData layer containing counts used by scGPT preprocessing.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed passed to scGPT config creation.")
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=1200,
        help="Max sequence length passed to scGPT config.",
    )
    parser.add_argument(
        "--input-bins",
        type=int,
        default=51,
        help="Number of bins used when discretizing expression values.",
    )
    parser.add_argument(
        "--model-run",
        type=str,
        default="pretrained",
        help="Value forwarded to `scGPT_instance(model_run=...)`.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of worker processes for scGPT dataloading/tokenization.",
    )
    parser.add_argument(
        "--scgpt-output-name",
        type=str,
        default="scGPT.parquet",
        help="Output filename for scGPT embeddings.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    scfoundation_dir = args.scfoundation_dir
    if scfoundation_dir is None:
        if DEFAULT_SCFOUNDATION_DIR.exists():
            scfoundation_dir = DEFAULT_SCFOUNDATION_DIR
        else:
            raise ValueError("--scfoundation-dir is required unless /app/zero-shot-scfoundation exists.")

    scgpt_out = args.output_dir / args.scgpt_output_name
    if scgpt_out.exists() and not args.overwrite:
        raise FileExistsError(f"{scgpt_out} exists. Use --overwrite to replace it.")

    run_embed_scgpt(
        dataset_path=str(args.adata),
        model_dir=str(args.scgpt_weights),
        output_dir=str(args.output_dir),
        n_hvg=args.n_hvg,
        gene_col=args.gene_col,
        layer_key=args.layer_key,
        log_norm=args.input_is_log_normalized == "True",
        seed=args.seed,
        max_seq_len=args.max_seq_len,
        batch_size=args.scgpt_batch_size,
        input_bins=args.input_bins,
        model_run=args.model_run,
        num_workers=args.num_workers,
        output_name=args.scgpt_output_name,
        scfoundation_dir=str(scfoundation_dir),
    )


if __name__ == "__main__":
    main()
