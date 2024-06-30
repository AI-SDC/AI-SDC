"""Base class for an attack object."""

from __future__ import annotations

import importlib
import inspect
import logging
import os
import uuid
from datetime import datetime

from fpdf import FPDF

from aisdc.attacks import report
from aisdc.attacks.target import Target

logger = logging.getLogger(__name__)


class Attack:
    """Base class to represent an attack."""

    def __init__(self, output_dir: str = "outputs", write_report: bool = True) -> None:
        """Instantiate an attack.

        Parameters
        ----------
        output_dir : str
            name of the directory where outputs are stored
        write_report : bool
            Whether to generate a JSON and PDF report.
        """
        self.output_dir: str = output_dir
        self.write_report: bool = write_report
        self.attack_metrics: dict | list = {}
        self.metadata: dict = {}
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def attack(self, target: Target) -> dict:
        """Run an attack."""
        raise NotImplementedError

    def _construct_metadata(self) -> None:
        """Generate attack metadata."""
        self.metadata = {
            "attack_name": str(self),
            "attack_params": self.get_params(),
            "global_metrics": {},
        }

    def _get_attack_metrics_instances(self) -> dict:
        """Get metrics for each individual repetition of an attack."""
        raise NotImplementedError

    def _make_pdf(self, output: dict) -> FPDF | None:
        """Create PDF report."""
        raise NotImplementedError

    def _make_report(self) -> dict:
        """Create attack report and write JSON/PDF as requested."""
        dest: str = os.path.join(self.output_dir, "report")
        logger.info("Generating reports: %s.json %s.pdf", dest, dest)
        # generate report
        self._construct_metadata()
        output: dict = {
            "log_id": str(uuid.uuid4()),
            "log_time": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
            "metadata": self.metadata,
            "attack_experiment_logger": self._get_attack_metrics_instances(),
        }
        # write JSON and PDF
        if self.write_report:
            report.write_json(output, dest)
            pdf_report = self._make_pdf(output)
            if pdf_report is not None:
                report.write_pdf(dest, pdf_report)
        # return report for programmatic access
        return output

    def __str__(self) -> str:
        """Return the string representation of an attack."""
        raise NotImplementedError

    @classmethod
    def _get_param_names(cls) -> list[str]:
        """Get parameter names."""
        init_signature = inspect.signature(cls.__init__)
        return [
            p.name
            for p in init_signature.parameters.values()
            if p.name != "self" and p.kind != p.VAR_KEYWORD
        ]

    def get_params(self) -> dict:
        """Get parameters for this attack.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        out: dict = {}
        for key in self._get_param_names():
            out[key] = getattr(self, key)
        return out


def get_class_by_name(class_path: str):
    """Return a class given its name."""
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)
