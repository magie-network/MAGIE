from magie.k_index_magpy import daily_K_full_archive
if __name__ == "__main__":
    kvals, errors = daily_K_full_archive(site_code="val", archive_path_builder=lambda date: "/magnetometer_archive/{}/{}/{}/iaga2002/".format(
        *date
    ),
    output_path_builder=lambda date: "/magnetometer_archive/{}/{}/{}/k_index/".format(
        *date
    ), start="2025-01-01", end="2026-01-01", max_workers=8, error_log_path="./daily_k_errors.log")
