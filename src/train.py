"""
Training routine, extracted from the notebook. Fixes the two bugs found in the
original: (1) fit_generator -> model.fit with a tf.data.Dataset directly, so
there's no steps_per_epoch to accidentally miscalculate; (2) validation always
uses the real held-out validation set, never the (mislabeled, and in Kaggle's
case unlabeled) test folder.

CLI usage:
    python -m src.train --data-dir /path/to/kaggle/download --model efficientnet --epochs 15
"""
import argparse
import os
import time

from tensorflow.keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight

from src.dataset import load_labels, stratified_split
from src.models import build_ensemble_model, build_efficientnet_model
from src.preprocessing import make_dataset


def train_model(model, train_ds, val_ds, name, output_dir, model_dir, epochs=15, class_weight=None):
    callbacks = [
        CSVLogger(os.path.join(output_dir, "metrics", f"{name}_training_log.csv")),
        ModelCheckpoint(
            os.path.join(model_dir, f"{name}_best.keras"),
            monitor="val_auc", mode="max", save_best_only=True,
        ),
        EarlyStopping(monitor="val_auc", mode="max", patience=3, restore_best_weights=True),
    ]
    start = time.time()
    history = model.fit(
        train_ds, validation_data=val_ds, epochs=epochs,
        class_weight=class_weight, callbacks=callbacks, verbose=1,
    )
    train_time_sec = time.time() - start
    print(f"[{name}] training wall-clock time: {train_time_sec:.1f}s for {epochs} epoch(s)")
    return history, train_time_sec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--model-dir", default="./models")
    parser.add_argument("--output-dir", default="./outputs")
    parser.add_argument("--model", choices=["ensemble", "efficientnet"], default="efficientnet")
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-class-weight", action="store_true",
                         help="Use class-weighted loss instead of undersampling.")
    parser.add_argument("--sample-size-per-class", type=int, default=None,
                         help="Reproduce the original undersampling strategy, e.g. 85000.")
    args = parser.parse_args()

    for d in (args.model_dir, args.output_dir,
              os.path.join(args.output_dir, "figures"), os.path.join(args.output_dir, "metrics")):
        os.makedirs(d, exist_ok=True)

    df = load_labels(os.path.join(args.data_dir, "train_labels.csv"))
    df_train, df_val = stratified_split(
        df, val_split=0.10, seed=args.seed, sample_size_per_class=args.sample_size_per_class
    )

    train_img_dir = os.path.join(args.data_dir, "train")
    train_ds = make_dataset(df_train, train_img_dir, training=True,
                             batch_size=args.batch_size, image_size=args.image_size, seed=args.seed)
    val_ds = make_dataset(df_val, train_img_dir, training=False,
                           batch_size=args.batch_size, image_size=args.image_size, seed=args.seed)

    class_weight = None
    if args.use_class_weight:
        weights = compute_class_weight("balanced", classes=[0, 1], y=df_train["label"].values)
        class_weight = {0: weights[0], 1: weights[1]}
        print("Using class weights:", class_weight)

    if args.model == "ensemble":
        model = build_ensemble_model(args.image_size, args.learning_rate)
    else:
        model = build_efficientnet_model(args.image_size, args.learning_rate)

    train_model(model, train_ds, val_ds, args.model, args.output_dir, args.model_dir,
                epochs=args.epochs, class_weight=class_weight)


if __name__ == "__main__":
    main()
