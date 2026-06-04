import os
import subprocess
from PIL import Image
import shlex
def get_image_size(image_path):
    with Image.open(image_path) as img:
        return img.size  # (width, height)
def _audio_args(audio_path, audio_volume=1.0, audio_offset=0.0, loop_audio=False):
    """
    Build audio-related ffmpeg args.
    - audio_volume: 1.0 = unchanged, 0.5 = half, 2.0 = double
    - audio_offset: seconds to delay the audio start (can be negative to start earlier)
    - loop_audio: if True, loops the audio to cover the whole video (requires -shortest too)
    """
    args = []
    if loop_audio:
        # Loop indefinitely (-1) until video ends; -stream_loop must appear before the -i it applies to
        args += ['-stream_loop', '-1']
    # Audio input
    args += ['-i', audio_path]

    # Build audio filter chain (volume + optional time offset)
    afilters = []
    if audio_volume != 1.0:
        afilters.append(f'volume={audio_volume}')
    # Offset: positive = delay start; negative = shift earlier
    # Use "adelay" for positive offset (needs ms), and "atrim" + "asetpts" trick for negative.
    if audio_offset > 0:
        delay_ms = int(audio_offset * 1000)
        # adelay expects per-channel spec; use a single value with '|'
        afilters.append(f'adelay={delay_ms}|{delay_ms}')
    elif audio_offset < 0:
        # Trim the first |offset| seconds and reset timestamps
        # atrim=start=… will drop the beginning; then reset pts
        afilters.append(f'atrim=start={abs(audio_offset)},asetpts=PTS-STARTPTS')

    if afilters:
        args += ['-af', ','.join(afilters)]

    # AAC audio, 192 kbps is a solid default
    args += ['-c:a', 'aac', '-b:a', '192k']

    return args
# def create_video_from_images(folder_path, prefix='', num_digits=2, fps=5,
#                              create_gif=True, audio_path=None, audio_volume=1.0,
#                              audio_offset=0.0, loop_audio=False):
#     # Get all PNG images in the folder
#     image_files = sorted(f for f in os.listdir(folder_path) if f.endswith('.png'))
    
#     if image_files:
#         # Use first image to determine dimensions
#         first_image_path = os.path.join(folder_path, image_files[0])
#         width, height = get_image_size(first_image_path)
#         scale_filter = f'scale={width}:{height}:flags=lanczos'
        
#         # Input file pattern
#         input_pattern = os.path.join(folder_path, prefix+f'%0{num_digits}d.png')
#         output_video = os.path.join(folder_path, 'output.mp4')
        
#         # Create video
#         ffmpeg_command = [
#             'ffmpeg',
#             '-y',
#             '-framerate', f'{fps}',
#             '-i', input_pattern,
#             '-vf', scale_filter,
#             '-c:v', 'libx264',
#             '-pix_fmt', 'yuv420p',
#             output_video
#         ]
#         # If audio provided, add audio args before output and ensure we stop at the shorter stream
#         if audio_path:
#             ffmpeg_command += _audio_args(
#                 audio_path=audio_path,
#                 audio_volume=audio_volume,
#                 audio_offset=audio_offset,
#                 loop_audio=loop_audio
#             )
#             ffmpeg_command += ['-shortest']  # end when the first stream ends
#             print('added audio')
#         ffmpeg_command += [output_video]
#         subprocess.run(ffmpeg_command)
#         print(f"Created video: {output_video}")
#         if create_gif:
#             # Convert video to GIF using two-pass palette method
#             convert_video_to_gif(output_video, folder_path, width, height, fps=fps)
#     else:
#         print(f"No images found in folder: {folder_path}")
def create_video_from_images(
    folder_path,
    prefix='',
    filename='output',
    num_digits=2,
    fps=5,
    audio_path=None,
    audio_volume=1.0,
    audio_offset=0.0,
    loop_audio=False,
    create_gif=True,
):

    # Detect image sequence
    image_files = sorted(f for f in os.listdir(folder_path) if f.lower().endswith('.png'))
    if not image_files:
        print(f"No images found in folder: {folder_path}")
        return

    first_image = os.path.join(folder_path, image_files[0])
    width, height = get_image_size(first_image)
    def even(x): return x if x % 2 == 0 else x - 1
    width  = even(width)
    height = even(height)
    scale_filter = f"scale={width}:{height}:flags=lanczos"

    # Auto-detect starting number from first matching file
    first_filename = image_files[0]
    # Remove prefix and .png extension, extract the numeric part
    numeric_str = first_filename.replace(prefix, '', 1).replace('.png', '', 1)
    try:
        start_number = int(numeric_str)
    except ValueError:
        start_number = 1  # fallback to 1 if can't parse
    
    input_pattern = os.path.join(folder_path, prefix + f"%0{num_digits}d.png")
    output_video = os.path.join(folder_path, filename+".mp4")

    cmd = [
        "ffmpeg", "-hide_banner", "-stats", "-y",

        # Apply framerate BEFORE the image input
        "-framerate", str(fps),

        # Input 0: images
        "-start_number", str(start_number),
        "-i", input_pattern,
    ]

    # Optional audio input
    if audio_path:
        if loop_audio:
            cmd += ["-stream_loop", "-1"]  # must come before audio input

        # Input 1: audio
        cmd += ["-i", audio_path]

    # Video settings
    cmd += [
        "-vf", scale_filter,
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
    ]

    # Add audio filters if present
    if audio_path:
        af = []
        if audio_volume != 1.0:
            af.append(f"volume={audio_volume}")
        if audio_offset > 0:
            af.append(f"adelay={int(audio_offset*1000)}|{int(audio_offset*1000)}")
        elif audio_offset < 0:
            af.append(f"atrim=start={abs(audio_offset)},asetpts=PTS-STARTPTS")

        if af:
            cmd += ["-af", ",".join(af)]

        # Audio codec
        cmd += ["-c:a", "aac", "-b:a", "192k"]

        # Critical: map correctly!
        cmd += ["-map", "0:v:0", "-map", "1:a:0", "-shortest"]

    cmd += [output_video]

    # Debug print
    print("FFmpeg CMD:", " ".join(shlex.quote(c) for c in cmd))

    # Run
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    print(p.stderr)
    if p.returncode != 0:
        raise RuntimeError("ffmpeg failed")

    print("✅ Created video:", output_video)
    if create_gif:
        # Convert video to GIF using two-pass palette method
        convert_video_to_gif(output_video, folder_path, width, height, fps=fps, filename=filename)

def convert_video_to_gif(video_path, folder_path, width, height, fps=5, filename='output'):
    gif_output = os.path.join(folder_path, filename+'.gif')
    palette_path = os.path.join(folder_path, 'palette.png')
    scale_filter = f'scale={width}:{height}:flags=lanczos'

    # Step 1: Generate palette
    palettegen_command = [
        'ffmpeg',
        '-y',
        '-i', video_path,
        '-vf', f'fps={fps},{scale_filter},palettegen',
        palette_path
    ]
    subprocess.run(palettegen_command)
    
    # Step 2: Create GIF using palette
    gif_command = [
        'ffmpeg',
        '-y',
        '-i', video_path,
        '-i', palette_path,
        '-filter_complex', f'fps={fps},{scale_filter}[x];[x][1:v]paletteuse',
        gif_output
    ]
    subprocess.run(gif_command)
    if os.path.exists(palette_path):
        os.remove(palette_path)
    print(f"Created GIF: {gif_output}")

def process_all_folders(base_folder, **create_video_kwargs):
    for root, dirs, files in os.walk(base_folder):
        create_video_from_images(root, **create_video_kwargs)
