import argparse
import logging
import os
import pathlib as pl
import warnings

import numpy as np
import pandas as pd
import scanpy as sc
import tifffile
import timm
import torch
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

numba_logger = logging.getLogger("numba")
numba_logger.setLevel(logging.WARNING)


def load_uni_model(model_path: str, device: str = "cuda"):
    timm_kwargs = {
        "model_name": "vit_giant_patch14_224",
        "img_size": 224,
        "patch_size": 14,
        "depth": 24,
        "num_heads": 24,
        "init_values": 1e-5,
        "embed_dim": 1536,
        "mlp_ratio": 2.66667 * 2,
        "num_classes": 0,
        "no_embed_class": True,
        "mlp_layer": timm.layers.SwiGLUPacked,
        "act_layer": torch.nn.SiLU,
        "reg_tokens": 8,
        "dynamic_img_size": True,
    }

    model = timm.create_model(pretrained=False, **timm_kwargs)
    model.load_state_dict(torch.load(model_path, map_location="cpu"), strict=True)
    model.eval().to(device)

    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    return model, transform


def embed_uni(
    wsi,
    adata,
    he_coords,
    output_dir: str,
    model_path: str,
    output_name: str = "UNI.parquet",
    batch_size: int = 128,
    device: str = "cuda",
):
    print("Load UNI model")
    model, transform = load_uni_model(model_path, device)

    os.makedirs(output_dir, exist_ok=True)

    embeddings = []
    cell_ids = []
    batch_imgs = []
    batch_ids = []

    print(f"Embedding {len(he_coords)} image patches in batches of {batch_size}...")
    for cid, (x, y) in tqdm(zip(adata.obs_names, he_coords), total=len(adata)):
        x, y = int(x), int(y)
        x0, x1 = x - 128, x + 128
        y0, y1 = y - 128, y + 128

        pad_x0 = max(0, -x0)
        pad_x1 = max(0, x1 - wsi.shape[1])
        pad_y0 = max(0, -y0)
        pad_y1 = max(0, y1 - wsi.shape[0])

        patch = np.pad(
            wsi[max(0, y0):min(wsi.shape[0], y1), max(0, x0):min(wsi.shape[1], x1)],
            ((pad_y0, pad_y1), (pad_x0, pad_x1), (0, 0)),
            mode="constant",
        )

        if patch.shape[:2] != (256, 256):
            continue

        batch_imgs.append(transform(Image.fromarray(patch)))
        batch_ids.append(cid)

        if len(batch_imgs) == batch_size:
            img_tensor = torch.stack(batch_imgs).to(device)
            with torch.inference_mode(), torch.autocast(device_type=device, dtype=torch.float16):
                batch_embs = model(img_tensor).to(torch.float16).cpu().numpy()

            embeddings.extend(batch_embs)
            cell_ids.extend(batch_ids)
            batch_imgs.clear()
            batch_ids.clear()

    if batch_imgs:
        img_tensor = torch.stack(batch_imgs).to(device)
        with torch.inference_mode(), torch.autocast(device_type=device, dtype=torch.float16):
            batch_embs = model(img_tensor).to(torch.float16).cpu().numpy()
        embeddings.extend(batch_embs)
        cell_ids.extend(batch_ids)

    out = pl.Path(output_dir) / output_name
    embedding_columns = [str(i) for i in range(len(embeddings[0]))]
    pd.DataFrame(embeddings, index=cell_ids, columns=embedding_columns).to_parquet(out)
    print(f"Saved {len(cell_ids)} embeddings to {out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run UNI embedding extraction for SpatialFusion.",
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
        "--uni-weights",
        type=pl.Path,
        required=True,
        help="Path to UNI model weights file.",
    )
    parser.add_argument(
        "--uni-batch-size",
        type=int,
        default=512,
        help="Batch size for UNI patch inference.",
    )
    parser.add_argument(
        "--spatial-key",
        type=str,
        default="spatial",
        help="Key in `adata.obsm` containing spot pixel coordinates.",
    )
    parser.add_argument(
        "--uni-output-name",
        type=str,
        default="UNI.parquet",
        help="Output filename for UNI embeddings.",
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

    out = args.output_dir / args.uni_output_name
    if out.exists() and not args.overwrite:
        raise FileExistsError(f"{out} exists. Use --overwrite to replace it.")

    adata = sc.read_h5ad(args.adata)
    with tifffile.TiffFile(args.wsi) as tif:
        wsi = tif.series[0].asarray()

    embed_uni(
        wsi=wsi,
        adata=adata,
        he_coords=adata.obsm[args.spatial_key],
        output_dir=str(args.output_dir),
        model_path=str(args.uni_weights),
        output_name=args.uni_output_name,
        batch_size=args.uni_batch_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
