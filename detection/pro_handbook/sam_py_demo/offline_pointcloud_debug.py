from .get_book_points import run_capture_and_pca_offline
import numpy as np

def main():
    theta_rad, p_min, book_width, shot_dir = run_capture_and_pca_offline(
        query="WMS導入と運用のための99の極意[第2版]",
        shot_dir="captures/20260409_165212",
        sam_device="gpu",
    )

    print("\n=== OFFLINE Summary ===")
    print(f"book width [mm] = {book_width}")
    print(f"roll (deg)      = {np.degrees(theta_rad):.6f}")
    print(f"p_min           = {p_min}")
    print(f"shot_dir        = {shot_dir}")

if __name__ == "__main__":
    main()