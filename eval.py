import argparse
import pathlib
import os
from datetime import datetime
import numpy as np
from tqdm import tqdm
import shutil

from cwvae import build_model
from data_loader import *
import tools
from loggers.checkpoint import Checkpoint


if __name__ == "__main__":
    tf.disable_v2_behavior()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--logdir",
        default=None,
        type=str,
        required=True,
        help="path to dir containing model checkpoints (with config in the parent dir)",
    )
    parser.add_argument(
        "--num-examples", default=100, type=int, help="number of examples to eval on"
    )
    parser.add_argument(
        "--eval-seq-len",
        default=None,
        type=int,
        help="total length of evaluation sequences",
    )
    parser.add_argument("--datadir", default=None, type=str)
    parser.add_argument(
        "--num-samples", default=1, type=int, help="samples to generate per example"
    )
    parser.add_argument(
        "--open-loop-ctx", default=36, type=int, help="number of context frames"
    )
    parser.add_argument(
        "--use-obs",
        default=None,
        type=str,
        help="string of T/Fs per level, e.g. TTF to skip obs at the top level",
    )

    args = parser.parse_args()

    copy_datasets = False
    if args.datadir is None and "DATA_ROOT" in os.environ:
        copy_datasets = True
        args.datadir = os.path.join(os.environ["DATA_ROOT"], "datasets")
    assert args.datadir is not None
    print("***", args.datadir)

    assert os.path.exists(args.logdir)

    # Set directories.
    args.logdir = pathlib.Path(args.logdir).resolve()
    exp_rootdir = str(args.logdir.parent)

    # args.logdir format: ..../exp_name/dataset_name/model_desc/model_iter
    eval_name = f"{args.logdir.parent.parent.parent.stem}-{args.logdir.stem.split('_')[-1]}"
    eval_logdir = os.path.join(exp_rootdir, eval_name)
    print(f"Eval directory: {eval_logdir}")

    # Load config.
    cfg = tools.read_configs(os.path.join(exp_rootdir, "config.yml"))
    cfg.batch_size = 1
    cfg.open_loop_ctx = args.open_loop_ctx
    if args.eval_seq_len is not None:
        cfg.eval_seq_len = args.eval_seq_len
    if args.datadir:
        cfg.datadir = args.datadir
    if args.use_obs is not None:
        assert len(args.use_obs) == cfg.levels
        args.use_obs = args.use_obs.upper()
        cfg.use_obs = [dict(T=True, F=False)[c] for c in args.use_obs]
    else:
        cfg.use_obs = True
    tools.validate_config(cfg)

    if copy_datasets:
        if cfg.dataset == "minerl":
            src = "datasets/minerl_navigate"
            dst = os.path.join(cfg.datadir, "minerl_navigate")
        elif cfg.dataset == "mazes":
            src = "datasets/gqn_mazes"
            dst = os.path.join(cfg.datadir, "gqn_mazes")
        if not os.path.exists(dst):
            print(f"Copying the dataset from {src} to {dst}")
            shutil.copytree(src, dst)
            print("Copying done.")

    # Load dataset.
    _, val_data_batch = load_dataset(cfg)

    # Define session
    session_config = tf.ConfigProto(device_count={"GPU": 1}, log_device_placement=False)
    session = tf.Session(config=session_config)

    # Build model.
    model_components = build_model(cfg)

    # Define checkpoint saver for variables currently in session.
    checkpoint = Checkpoint(exp_rootdir)

    print("Restoring model from {}".format(args.logdir))
    checkpoint.restore(session, os.path.basename(os.path.normpath(args.logdir)))

    os.makedirs(eval_logdir, exist_ok=False)

    # Evaluating.
    ssim_best = []
    psnr_best = []
    ssim_all = []
    psnr_all = []
    for i_ex in tqdm(range(args.num_examples)):
        try:
            gts = np.tile(
                get_single_batch(val_data_batch, session),
                [args.num_samples, 1, 1, 1, 1],
            )
            preds = session.run(
                model_components["open_loop_obs_decoded"]["prior_multistep"],
                feed_dict={model_components["training"]["obs"]: gts},
            )

            # Computing metrics.
            ssim, psnr = tools.compute_metrics(gts[:, args.open_loop_ctx :], preds)

            # Getting arrays save-ready
            gts = np.uint8(np.clip(gts, 0, 1) * 255)
            preds = np.uint8(np.clip(preds, 0, 1) * 255)

            # Finding the order within samples wrt avg metric across time.
            order_ssim = np.argsort(np.mean(ssim, -1))
            order_psnr = np.argsort(np.mean(psnr, -1))

            # Setting aside the best metrics among all samples for plotting.
            ssim_best.append(np.expand_dims(ssim[order_ssim[-1]], 0))
            psnr_best.append(np.expand_dims(psnr[order_psnr[-1]], 0))

            ssim_all.append(ssim)
            psnr_all.append(psnr)

            for i_smp, pred in enumerate(preds):
                gt = gts[0].transpose([0, 3, 1, 2])
                pred = pred.transpose([0, 3, 1, 2])
                sample = np.concatenate([gt[:args.open_loop_ctx], pred])
                np.save(os.path.join(eval_logdir, f"sample_{i_ex:04d}-{i_smp}.npy"), sample)

        except tf.errors.OutOfRangeError:
            break

    psnr_all = np.stack(psnr_all)
    ssim_all = np.stack(ssim_all)
    np.save(os.path.join(eval_logdir, "psnr.npy"), psnr_all)
    np.save(os.path.join(eval_logdir, "ssim.npy"), ssim_all)
    # Plotting.
    tools.plot_metrics(ssim_best, eval_logdir, "ssim")
    tools.plot_metrics(psnr_best, eval_logdir, "psnr")
