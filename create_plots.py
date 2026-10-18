import pandas as pd
import matplotlib.pyplot as plt

def plot_quantization_error(csv_path, output_path):
    data = pd.read_csv(csv_path)

    required_columns = {"size", "sigma", "quantization_error"}
    missing = required_columns - set(data.columns)
    if missing:
        raise ValueError(f"CSV is missing required column(s): {', '.join(sorted(missing))}")

    data = data.dropna(subset=["size", "sigma", "quantization_error"])

    # aggregate repeated runs by taking the median quantization error.
    summary = (
        data.groupby(["sigma", "size"], as_index=False)["quantization_error"]
        .median()
        .sort_values(["sigma", "size"])
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    for sigma, group in summary.groupby("sigma", sort=True):
        ax.plot(
            group["size"],
            group["quantization_error"],
            marker="o",
            label=f"sigma = {sigma:g}",
        )

    ax.set_xlabel("Map size")
    ax.set_ylabel("Median quantization error")
    ax.set_title("Median Quantization Error vs. Map Size")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Sigma")
    fig.tight_layout()

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()

    plt.close(fig)


def main():
    plot_quantization_error("results/som_results.csv", "report/figs/map_side.pdf")


if __name__ == "__main__":
    main()
