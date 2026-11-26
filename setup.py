"""Copies config/ and ground_truth/ (at the root of the repository) into the wheel, as
pidsmaker/resources/, where `pidsmaker.config.pipeline` finds them when PIDSMaker is installed
from a wheel. In a source checkout or an editable install, they are read from the repository."""

import os

from setuptools import setup
from setuptools.command.build_py import build_py

RESOURCES = ["config", "ground_truth"]
# Not needed at runtime (the ground truth reports of DARPA, 4.5 MB)
EXCLUDED_EXTENSIONS = (".pdf",)


class BuildPyWithResources(build_py):
    def run(self):
        super().run()
        if self.editable_mode:
            return
        root = os.path.dirname(os.path.abspath(__file__))
        for name in RESOURCES:
            for folder, _, files in os.walk(os.path.join(root, name)):
                rel = os.path.relpath(folder, root)
                for f in files:
                    if f.endswith(EXCLUDED_EXTENSIONS):
                        continue
                    dst_dir = os.path.join(self.build_lib, "pidsmaker", "resources", rel)
                    self.mkpath(dst_dir)
                    self.copy_file(os.path.join(folder, f), os.path.join(dst_dir, f))


setup(cmdclass={"build_py": BuildPyWithResources})
