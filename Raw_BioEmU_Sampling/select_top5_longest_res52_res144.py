from pathlib import Path

import numpy as np


INPUT_PATH = Path("ca_distance_res52_res144.dat")
OUTPUT_PATH = Path("top5_longest_res52_res144.csv")


def main() -> None:
    data = np.loadtxt(INPUT_PATH, comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)

    top = data[np.argsort(data[:, 1])[::-1][:5]]
    top = top[np.argsort(top[:, 1])[::-1]]

    np.savetxt(
        OUTPUT_PATH,
        top,
        delimiter=",",
        header="frame_number,distance_angstrom",
        comments="",
        fmt=["%d", "%.6f"],
    )

    for frame, distance in top:
        print(f"{int(frame)},{distance:.6f}")


if __name__ == "__main__":
    main()
