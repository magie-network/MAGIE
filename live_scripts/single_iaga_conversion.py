from magie.file_conversions import magie_legacy2iaga2002
import pandas as pd
now_time = pd.Timestamp('2025-12-31T00:01') - pd.Timedelta(days=1)  # Use yesterday's date as we assume we run this after midnight
now_time = str(now_time.date()).split('-')
archive_path_builder=lambda date: '/home/simon/Documents/magnetometer_archive/{}/{}/{}/txt/'.format(*date)
output_dir_builder=lambda date: '/home/simon/Documents/magnetometer_archive/{}/{}/{}/iaga2002/'.format(*date)

for site in ['dun', 'val']:

    filename= archive_path_builder(now_time) + f"{site}{''.join(now_time)}.txt"
    file, filename= magie_legacy2iaga2002(filename)
    with open(output_dir_builder(now_time) + filename, 'w') as f:
        f.write(file)
