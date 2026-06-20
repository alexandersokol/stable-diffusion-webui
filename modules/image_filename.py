import os


def get_next_available_filename(path, basename, file_decoration, extension, start_sequence, max_attempts=500):
    for i in range(max_attempts):
        sequence_number = start_sequence + i
        fn = f"{sequence_number:05}" if basename == '' else f"{basename}-{sequence_number:04}"
        fullfn = os.path.join(path, f"{fn}{file_decoration}.{extension}")
        if not os.path.exists(fullfn):
            return fullfn

    raise FileExistsError(f"Could not find an available filename after {max_attempts} attempts in {path}")
