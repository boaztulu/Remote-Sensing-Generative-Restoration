import argparse, os, glob, sys, subprocess, cv2, torch
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

def collect(img_or_dir):
    if os.path.isdir(img_or_dir):
        exts = ("*.png","*.jpg","*.jpeg","*.bmp","*.tif","*.tiff","*.webp")
        files = []
        for e in exts:
            files += glob.glob(os.path.join(img_or_dir, e))
        return sorted(files)
    return [img_or_dir]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",    required=True)
    ap.add_argument("--input",   required=True)
    ap.add_argument("--outdir",  required=True)
    ap.add_argument("--tile",    type=int, default=512)
    ap.add_argument("--tile-pad",type=int, default=10)
    ap.add_argument("--pre-pad", type=int, default=0)
    ap.add_argument("--outscale",type=float, default=4.0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-idx",  type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    use_half = torch.cuda.is_available()
    upsampler = RealESRGANer(
        scale=4, model_path=args.ckpt, model=model,
        tile=args.tile, tile_pad=args.tile_pad, pre_pad=args.pre_pad, half=use_half
    )

    files = collect(args.input)
    if not files:
        raise FileNotFoundError(f"No images found at: {args.input}")

    # shard the list for multi-GPU/task runs
    shard_files = [f for i, f in enumerate(files) if (i % args.num_shards) == args.shard_idx]
    print(f"[shard {args.shard_idx}/{args.num_shards}] processing {len(shard_files)} file(s).")

    for ipath in shard_files:
        img = cv2.imread(ipath, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"[skip] cannot read {ipath}")
            continue
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.ndim == 3 and img.shape[2] == 4:  # drop alpha
            img = img[:, :, :3]

        try:
            out, _ = upsampler.enhance(img, outscale=args.outscale)
        except RuntimeError as e:
            msg = str(e).lower()
            if "out of memory" in msg or "cuda" in msg:
                print(f"[warn] OOM at tile={args.tile}; retry smaller.")
                upsampler.tile = max(128, args.tile // 2)
                out, _ = upsampler.enhance(img, outscale=args.outscale)
            else:
                raise

        name, _ = os.path.splitext(os.path.basename(ipath))
        opath = os.path.join(args.outdir, f"{name}_x4.png")
        cv2.imwrite(opath, out)
        print(f"[ok] {ipath} -> {opath}")

    print(f"Done. Results in: {args.outdir}")

if __name__ == "__main__":
    main()
