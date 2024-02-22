"""Plugen4faces training script."""
import click
from plugen4faces.flow import train


@click.command()
@click.option("--sigma", default=0.7, type=float)
@click.option("--decay", default=0.9995, type=float)
@click.option("--epochs", default=1500, type=int)
@click.option("--num_layers", "-L", default=10, type=int)
@click.option("--flow_divisor", default=400, type=float)
@click.option("--attr_divisor", default=4, type=float)
@click.option("--batch_size", "-bs", default=512, type=int)
@click.option("--dropout", default=0.0, type=float)
@click.option(
    "--batch_norm_between_layers", is_flag=True, show_default=True, default=True
)
@click.option("--snapshot_interval", default=100, type=int)
@click.option("--remove_partials", "-R", is_flag=True, show_default=True, default=True)
def main(
    sigma: float,
    decay: float,
    epochs: int,
    num_layers: int,
    flow_divisor: float,
    attr_divisor: float,
    batch_size: int,
    dropout: float,
    batch_norm_between_layers: bool,
    snapshot_interval: int,
    remove_partials: bool,
):
    train(
        sigma,
        decay,
        epochs,
        num_layers,
        flow_divisor,
        attr_divisor,
        batch_size,
        dropout,
        batch_norm_between_layers,
        snapshot_interval,
        remove_partials,
    )


if __name__ == "__main__":
    main()
