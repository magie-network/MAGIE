from magie.k_index_magpy import daily_K_full_archive, daily_K_plots_full_archive
if __name__ == "__main__": # Full archive run
    for site_code in ["val", "dun"]:
        kvals, errors = daily_K_full_archive(site_code=site_code, archive_path_builder=lambda date: "/magnetometer_archive/{}/{}/{}/iaga2002/".format(
            *date
        ),
        output_path_builder=lambda date: "/magnetometer_archive/{}/{}/{}/k_index/".format(
            *date
        ), start="2025-01-01", end="2026-01-01", max_workers=8, error_log_path="./daily_k_errors.log")
        
        results, errors = daily_K_plots_full_archive(site_code=site_code, archive_path_builder=lambda date: "/home/simon/Documents/magnetometer_archive/{}/{}/{}/k_index/".format(
            *date), output_path_builder=lambda date: "/home/simon/Documents/magnetometer_archive/{}/{}/{}/k_index/".format(
            *date), start="2025-01-01", end="2026-01-01", max_workers=8, error_log_path="./daily_k_errors.log")

if __name__ == "__main__": # daily update of archive
    import pandas as pd
    now_time= pd.Timestamp("2026-01-01").floor("1D")
    for site_code in ["val", "dun"]:
        kvals, errors = daily_K_full_archive(site_code=site_code, archive_path_builder=lambda date: "/home/simon/Documents/magnetometer_archive/{}/{}/{}/iaga2002/".format(
            *date
        ),
        output_path_builder=lambda date: "/home/simon/Documents/magnetometer_archive/{}/{}/{}/k_index/".format(
            *date
        ), start=now_time-pd.Timedelta(3, 'D'), end=now_time-pd.Timedelta(2, 'D'), max_workers=8, error_log_path="./daily_k_errors.log")
        print(kvals)
        print(errors)
        results, errors = daily_K_plots_full_archive(site_code=site_code, archive_path_builder=lambda date: "/home/simon/Documents/magnetometer_archive/{}/{}/{}/k_index/".format(
            *date), output_path_builder=lambda date: "/home/simon/Documents/magnetometer_archive/{}/{}/{}/k_index/".format(
            *date), start=now_time-pd.Timedelta(3, 'D'), end=now_time-pd.Timedelta(2, 'D'), max_workers=8, error_log_path="./daily_k_errors.log")
        print(results)
        print(errors)