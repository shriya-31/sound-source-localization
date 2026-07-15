"""
Visualization utilities
"""

import matplotlib.pyplot as plt

class ResultPlotter:

    @staticmethod
    def plot_neural_ild(angles, neural_ild, broadband, cfs, channels=(0,7,15,23,31,39,47), title="Neural ILD Representation"):

        plt.figure(figsize=(12,7))

        for ch in channels:

            plt.plot(
                angles,
                neural_ild[ch],
                marker="o",
                linewidth=2,
                alpha=0.5,
                label=f"{cfs[ch]/1000:.1f} kHz (Ch {ch})",
            )

        plt.plot(
            angles,
            broadband,
            color="red",
            linewidth=4,
            linestyle="--",
            label="Broadband Mean",
        )

        plt.axhline(
            0,
            color="black",
            linestyle="--",
            alpha=0.5,
        )

        plt.axvline(
            0,
            color="black",
            linestyle="--",
            alpha=0.5,
        )

        plt.grid(alpha=0.6)

        plt.xlabel("Azimuth (degrees)")
        plt.ylabel("Neural ILD")

        plt.title(title)

        plt.legend(
            bbox_to_anchor=(1.02,1),
            loc="upper left",
        )

        plt.tight_layout()

        plt.show()