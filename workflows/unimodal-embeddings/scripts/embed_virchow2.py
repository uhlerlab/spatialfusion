import argparse
import logging
import pathlib as pl
import warnings

import numpy as np
import pandas as pd
import scanpy as sc
import tifffile
import timm
import torch
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from tqdm.auto import tqdm

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

numba_logger = logging.getLogger("numba")
numba_logger.setLevel(logging.WARNING)

DEFAULT_VIRCHOW2_MODEL_NAME = "hf-hub:paige-ai/Virchow2"


def load_wsi(path: pl.Path):
    try:
        logging.info("Loading WSI with tifffile.imread")
        return tifffile.imread(path)
    except Exception as exc:
        logging.warning(
            "tifffile.imread failed (%s: %s); falling back to the first TIFF page",
            type(exc).__name__,
            exc,
        )
        with tifffile.TiffFile(path) as tif:
            return tif.pages[0].asarray()


def load_virchow2_model(
    model_name: str = DEFAULT_VIRCHOW2_MODEL_NAME,
    device: str = "cuda",
):
    model = timm.create_model(
        model_name,
        pretrained=True,
        mlp_layer=timm.layers.SwiGLUPacked,
        act_layer=torch.nn.SiLU,
    )
    model.eval().to(device)

    transform = create_transform(
        **resolve_data_config(model.pretrained_cfg, model=model)
    )
    return model, transform


def extract_centered_patch(wsi, x: int, y: int, patch_size: int = 256):
    half_size = patch_size // 2
    x0, x1 = x - half_size, x + half_size
    y0, y1 = y - half_size, y + half_size

    pad_x0 = max(0, -x0)
    pad_x1 = max(0, x1 - wsi.shape[1])
    pad_y0 = max(0, -y0)
    pad_y1 = max(0, y1 - wsi.shape[0])

    patch = np.pad(
        wsi[max(0, y0):min(wsi.shape[0], y1), max(0, x0):min(wsi.shape[1], x1)],
        ((pad_y0, pad_y1), (pad_x0, pad_x1), (0, 0)),
        mode="constant",
    )

    if patch.shape[:2] != (patch_size, patch_size):
        return None
    return patch


def virchow2_forward(model, img_tensor):
    output = model(img_tensor)
    class_token = output[:, 0]
    patch_tokens = output[:, 5:]
    return torch.cat([class_token, patch_tokens.mean(1)], dim=-1)


def embed_virchow2(
    wsi,
    cell_names,
    he_coords,
    output_file: pl.Path,
    model_name: str = DEFAULT_VIRCHOW2_MODEL_NAME,
    batch_size: int = 512,
    device: str = "cuda",
):
    logging.info("Loading Virchow2 model")
    model, transform = load_virchow2_model(model_name=model_name, device=device)

    embeddings = []
    cell_ids = []
    batch_imgs = []
    batch_ids = []

    logging.info("Embedding %s image patches in batches of %s", len(he_coords), batch_size)
    for cid, (x, y) in tqdm(zip(cell_names, he_coords), total=len(cell_names)):
        patch = extract_centered_patch(wsi, int(x), int(y))
        if patch is None:
            continue

        batch_imgs.append(transform(Image.fromarray(patch)))
        batch_ids.append(cid)

        if len(batch_imgs) == batch_size:
            img_tensor = torch.stack(batch_imgs).to(device)
            with torch.inference_mode(), torch.autocast(device_type=device, dtype=torch.float16):
                batch_embs = virchow2_forward(model, img_tensor).cpu().numpy()

            embeddings.extend(batch_embs)
            cell_ids.extend(batch_ids)
            batch_imgs.clear()
            batch_ids.clear()

    if batch_imgs:
        img_tensor = torch.stack(batch_imgs).to(device)
        with torch.inference_mode(), torch.autocast(device_type=device, dtype=torch.float16):
            batch_embs = virchow2_forward(model, img_tensor).cpu().numpy()
        embeddings.extend(batch_embs)
        cell_ids.extend(batch_ids)

    pd.DataFrame(embeddings, index=cell_ids).to_parquet(output_file)
    logging.info("Saved %s embeddings to %s", len(cell_ids), output_file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Virchow2 embedding extraction for SpatialFusion.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--adata",
        type=pl.Path,
        required=True,
        help="Input AnnData (.h5ad). Spatial coordinates are read from `obsm`.",
    )
    parser.add_argument(
        "--wsi",
        type=pl.Path,
        required=True,
        help="Input H&E/WSI TIFF file.",
    )
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
        "--spatial-key",
        type=str,
        default="spatial",
        help="Key in `adata.obsm` containing spot or cell pixel coordinates.",
    )
    parser.add_argument(
        "--virchow2-model-name",
        type=str,
        default=DEFAULT_VIRCHOW2_MODEL_NAME,
        help="timm model name or Hugging Face Hub reference for Virchow2.",
    )
    parser.add_argument(
        "--virchow2-batch-size",
        type=int,
        default=512,
        help="Batch size for Virchow2 patch inference.",
    )
    parser.add_argument(
        "--virchow2-output-name",
        type=str,
        default="Virchow2.parquet",
        help="Output filename for Virchow2 embeddings.",
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

    out = args.output_dir / args.virchow2_output_name
    if out.exists() and not args.overwrite:
        raise FileExistsError(f"{out} exists. Use --overwrite to replace it.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    adata = sc.read_h5ad(args.adata)
    wsi = load_wsi(args.wsi)
    logging.info("Loaded WSI with shape %s", wsi.shape)

    index = (
        adata.obs["cell_id"].astype(str).values
        if "cell_id" in adata.obs.columns
        else adata.obs_names.astype(str)
    )

    embed_virchow2(
        wsi=wsi,
        cell_names=index,
        he_coords=adata.obsm[args.spatial_key],
        output_file=out,
        model_name=args.virchow2_model_name,
        batch_size=args.virchow2_batch_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
