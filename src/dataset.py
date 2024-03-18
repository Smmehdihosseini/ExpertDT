import os
from utils.wsi_manager import SectionManager


class Dataset:
    def __init__(self,
                 src,
                 crop_size,
                 slide_extensions,
                 resize=False,
                 keep_out=()):
        self.src = src
        self.crop_size = crop_size
        self.slide_extensions = slide_extensions
        if not resize:
            self.resize = self.crop_size
        else:
            self.resize = resize
        self.keep_out = keep_out

        self._get_filepaths()
        self._make_crop_list()

    def _get_filepaths(self):
        self._filepaths = list()
        # Get all filepaths with proper extension
        for extension in self.slide_extensions:
            self._filepaths += [os.path.join(dp, f) for dp, dn, filenames in os.walk(self.src) for f in filenames if
                                f.endswith(extension)]
        # Exclude keep_out patients
        for filepath in self._filepaths:
            for element in self.keep_out:
                if element in filepath:
                    self._filepaths.remove(filepath)

    def _make_crop_list(self):
        sections = SectionManager(self.crop_size, overlap=1)
        self.crop_objs_original = [sections.crop(filepath, slide_label=None, size=self.resize, save_dir=None)
                                   for filepath in self._filepaths]
        print("Found {} files in {}".format(len(self._filepaths), self.src))


