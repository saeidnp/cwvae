import numpy as np
import argparse
import yaml
import os

import tensorflow.compat.v1 as tf

from cwvae import build_model
from loggers.summary import Summary
from loggers.checkpoint import Checkpoint
from data_loader import *
import tools
import wandb
import shutil
import time


def train_setup(cfg, loss):
    session_config = tf.ConfigProto(device_count={"GPU": 1}, log_device_placement=False)
    session = tf.Session(config=session_config)
    step = tools.Step(session)

    with tf.name_scope("optimizer"):
        # Getting all trainable variables.
        weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

        # Creating optimizer.
        optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.lr, epsilon=1e-04)

        # Computing gradients.
        grads = optimizer.get_gradients(loss, weights)
        grad_norm = tf.global_norm(grads)

        # Clipping gradients by global norm, and applying gradient.
        if cfg.clip_grad_norm_by is not None:
            capped_grads = tf.clip_by_global_norm(grads, cfg.clip_grad_norm_by)[0]
            capped_gvs = [
                tuple((capped_grads[i], weights[i])) for i in range(len(weights))
            ]
            apply_grads = optimizer.apply_gradients(capped_gvs)
        else:
            gvs = zip(grads, weights)
            apply_grads = optimizer.apply_gradients(gvs)
    return apply_grads, grad_norm, session, step


if __name__ == "__main__":
    tf.disable_v2_behavior()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--logdir",
        default=None,
        type=str,
        help="path to root log directory",
    )
    parser.add_argument(
        "--datadir",
        default=None,
        type=str,
        help="path to root data directory",
    )
    parser.add_argument(
        "--config",
        default=None,
        type=str,
        help="path to config yaml file",
        required=True,
    )
    parser.add_argument(
        "--base-config",
        default="./configs/base_config.yml",
        type=str,
        help="path to base config yaml file",
    )

    args = parser.parse_args()
    copy_datasets = False
    if args.datadir is None and "DATA_ROOT" in os.environ:
        copy_datasets = True
        args.datadir = os.path.join(os.environ["DATA_ROOT"], "datasets")
    print("***", args.datadir)

    cfg = tools.read_configs(
        args.config, args.base_config, datadir=args.datadir, logdir=args.logdir
    )

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

    # Creating model dir with experiment name.
    exp_rootdir = os.path.join(cfg.logdir, cfg.dataset, tools.exp_name(cfg))
    os.makedirs(exp_rootdir, exist_ok=True)

    # Dumping config.
    print(cfg)
    with open(os.path.join(exp_rootdir, "config.yml"), "w") as f:
        yaml.dump(dict(cfg), f, default_flow_style=False)
    # wandb.tensorboard.patch(root_logdir=exp_rootdir)
    wandb.init(entity=os.environ["WANDB_ENTITY"],
                project=os.environ["WANDB_PROJECT"],
                config=cfg,
                sync_tensorboard=True,
                tags=["cwvae"])
    print("wandb run id:", wandb.run.id)

    # Load dataset.
    train_data_batch, val_data_batch = load_dataset(cfg)

    # Build model.
    model_components = build_model(cfg)
    model = model_components["meta"]["model"]

    # Setting up training.
    apply_grads, grad_norm, session, step = train_setup(cfg, model.loss)

    # Define summaries.
    summary = Summary(exp_rootdir, save_gifs=cfg.save_gifs)
    summary.build_summary(cfg, model_components, grad_norm=grad_norm)

    # Define checkpoint saver for variables currently in session.
    checkpoint = Checkpoint(exp_rootdir)

    # Restore model (if exists).
    if os.path.exists(checkpoint.log_dir_model):
        print("Restoring model from {}".format(checkpoint.log_dir_model))
        checkpoint.restore(session)
        print("Will start training from step {}".format(step()))
    else:
        # Initialize all variables.
        session.run(tf.global_variables_initializer())

    # Start training.
    print("Getting validation batches.")
    val_batches = get_multiple_batches(val_data_batch, cfg.num_val_batches, session)
    print("Training.")
    t_0 = time.time()
    while True:
        try:
            train_batch = get_single_batch(train_data_batch, session)
            print(f"Step: {step()}, {(time.time() - t_0) / (step() + 1):.2f} s/step")
            feed_dict_train = {model_components["training"]["obs"]: train_batch}
            feed_dict_val = {model_components["training"]["obs"]: val_batches}

            # Train one step.
            session.run(fetches=apply_grads, feed_dict=feed_dict_train)

            # Saving scalar summaries.
            if step() % cfg.save_scalars_every == 0:
                summaries = session.run(
                    summary.scalar_summary, feed_dict=feed_dict_train
                )
                summary.save(summaries, step(), True)
                summaries = session.run(summary.scalar_summary, feed_dict=feed_dict_val)
                summary.save(summaries, step(), False)

            # Saving gif summaries.
            if step() % cfg.save_gifs_every == 0:
                summaries = session.run(summary.gif_summary, feed_dict=feed_dict_train)
                summary.save(summaries, step(), True)
                summaries = session.run(summary.gif_summary, feed_dict=feed_dict_val)
                summary.save(summaries, step(), False)

            # Saving model.
            if step() % cfg.save_model_every == 0:
                checkpoint.save(session)

            if cfg.save_named_model_every and step() % cfg.save_named_model_every == 0:
                checkpoint.save(session, save_dir="model_{}".format(step()))

            step.increment()
        except tf.errors.OutOfRangeError:
            break

    print("Training complete.")
    wandb.finish()