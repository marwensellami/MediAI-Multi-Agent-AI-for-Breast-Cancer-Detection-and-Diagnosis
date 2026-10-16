# backend/check_pixel_spacing.py
import pydicom
from pathlib import Path

def check_pixel_spacing(file_path):
    ds = pydicom.dcmread(file_path, force=True)
    print(f"Modality : {ds.get('Modality')}")
    print(f"Rows : {ds.get('Rows')}, Columns : {ds.get('Columns')}")
    print(f"Number of Frames : {ds.get('NumberOfFrames')}")

    # Tags directs
    print(f"PixelSpacing (0028,0030) : {ds.get('PixelSpacing')}")
    print(f"ImagerPixelSpacing (0018,1164) : {ds.get('ImagerPixelSpacing')}")
    print(f"NominalScannedPixelSpacing (0018,2010) : {ds.get('NominalScannedPixelSpacing')}")

    # Shared Functional Groups
    if hasattr(ds, 'SharedFunctionalGroupsSequence'):
        for seq in ds.SharedFunctionalGroupsSequence:
            if hasattr(seq, 'PixelMeasuresSequence'):
                for item in seq.PixelMeasuresSequence:
                    print(f"PixelSpacing (dans Shared) : {item.get('PixelSpacing')}")

    # Per-frame (premier frame)
    if hasattr(ds, 'PerFrameFunctionalGroupsSequence'):
        for frame in ds.PerFrameFunctionalGroupsSequence[:1]:
            if hasattr(frame, 'PixelMeasuresSequence'):
                for item in frame.PixelMeasuresSequence:
                    print(f"PixelSpacing (dans PerFrame) : {item.get('PixelSpacing')}")

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "app/data/uploads/mammo.png"
    check_pixel_spacing(Path(path))