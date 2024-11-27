"""Configuration comparison base."""

import contextlib
import dataclasses
import pathlib
import typing
from dataclasses import dataclass

import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
import znjson
import zntrack
from dscribe.descriptors import SOAP
from tqdm import trange
from zntrack.config import NodeStatusEnum

from ipsuite import base

"""Use a Kernel and some initial configuration to select further configurations."""

from __future__ import annotations

import logging

import ase
import tqdm

from ipsuite.configuration_selection.base import ConfigurationSelection

if typing.TYPE_CHECKING:
    pass


log = logging.getLogger(__name__)


class KernelSelection(ConfigurationSelection):
    """Use the chosen kernel to selected configurations furthes apart.

    Attributes
    ----------
    n_configurations: int
        number of configurations to select
    kernel: ConfigurationComparison = zntrack.Nodes()
    points_per_cycle: int
        Number of configurations to add before recomputing the MMK
    correlation_time: int
        Ideally the correlation time of the data to only sample from uncorrelated data.
        This will only sample from configurations that are configuration_time apart.
        The smaller, the slower is the selection but the number of looked at
        configuration is larger giving potentially better results.
    seed: int
        seed selection in case of random picking initial configuration
    threshold: float
        threshold to stop the selection. The maximum number of configurations
        is still n_configurations. If the threshold is reached before, the
        selection stops. Typical values can be 0.995
    """

    n_configurations: int = zntrack.params()
    # kernel: "ipsuite.configuration_comparison.ConfigurationComparison" = zntrack.deps()
    initial_configurations: typing.List[ase.Atoms] = zntrack.deps(None)
    points_per_cycle: int = zntrack.params(1)
    kernel_results: typing.List[typing.List[float]] = zntrack.outs()
    seed: int = zntrack.params(1234)
    threshold: float = zntrack.params(None)

    # TODO what if the correlation time restricts the number of atoms to
    #  be less than n_configurations?
    correlation_time: int = zntrack.params(1)

    def select_atoms(self, atoms_lst: typing.List[ase.Atoms]) -> typing.List[int]:
        """Atom Selection method.

        Parameters
        ----------
        atoms_lst: typing.List[ase.Atoms]
            list of atoms objects

        Returns
        -------
        typing.List[int]:
            list containing the taken indices
        """
        if self.initial_configurations is None:
            np.random.seed(self.seed)
            initial_configurations = [atoms_lst[np.random.randint(len(atoms_lst))]]
            self.n_configurations -= 1
        else:
            initial_configurations = self.initial_configurations
        selected_atoms = []
        # we don't change the analyte, so we don't want to recompute the
        # SOAP vector every time.
        self.kernel.analyte = atoms_lst[:: self.correlation_time]
        self.kernel.remove_database = False
        self.kernel.load_analyte = False
        self.kernel.disable_tqdm = True

        self.kernel_results = []
        hist_results = []

        # TODO do not use the atoms in atoms_list but store the ids directly
        try:
            for idx in tqdm.trange(self.n_configurations, ncols=70):
                self.kernel.reference = initial_configurations + selected_atoms
                self.kernel.run()

                minimum_indices = np.argsort(self.kernel.result)[
                    : self.points_per_cycle
                ]
                selected_atoms += [
                    self.kernel.analyte[x.item()] for x in minimum_indices
                ]
                # There is currently no check in place to ensure that an atom is only
                # selected once. This should inherently be ensured by the way the
                # MMK selects configurations.
                self.kernel.load_analyte = True
                self.kernel_results.append(self.kernel.result)
                if self.threshold is not None:
                    hist, bins = np.histogram(
                        self.kernel.result, bins=np.linspace(0, 1, 10000), density=True
                    )
                    bins = bins[:-1]
                    hist[bins > self.threshold] = 0
                    hist_results.append(np.trapz(hist, bins))
                    if hist_results[-1] == 0:
                        log.warning(
                            f"Threshold {self.threshold} reached before"
                            f" {self.n_configurations} configurations were selected -"
                            f" stopping after selecting {idx + 1} configurations."
                        )
                        break
        finally:
            self.kernel.unlink_database()

        if self.initial_configurations is None:
            # include the randomly selected configuration
            selected_atoms += initial_configurations
            self.n_configurations += 1

        selected_ids = [
            idx for idx, atom in enumerate(atoms_lst) if atom in selected_atoms
        ]
        if self.threshold is None:
            if len(selected_ids) != self.n_configurations:
                print(f"{self.initial_configurations = }")
                raise ValueError(
                    f"Unable to select {self.n_configurations}. Could only select"
                    f" {len(selected_ids)}"
                )

        return selected_ids

    def plot_kernel(self, duration: int = 1000, remove: bool = True):
        """Generate an animation of the Kernel change while extending the reference.

        Raises
        ------
        ImportError: the imageio package is not shipped with mlsuite by default but is
                        required for generating the animation.
        """
        try:
            import imageio
        except ImportError as err:
            raise ImportError(
                "Package 'imageio' is required for generating a gif"
            ) from err

        import pathlib
        import shutil

        import matplotlib.pyplot as plt

        img_dir = pathlib.Path("img")

        img_dir.mkdir()
        for idx, results in enumerate(self.kernel_results):
            plt.plot(results)
            plt.title(f"Iteration {idx}")
            plt.ylabel("Kernel value")
            plt.savefig(img_dir / f"{str(idx).zfill(4)}.png")
            plt.close()

        with imageio.get_writer(
            "kernel_selection.gif", mode="I", duration=duration, loop=0
        ) as writer:
            for filename in sorted(img_dir.glob("*.png")):
                image = imageio.v2.imread(filename)
                writer.append_data(image)

        if remove:
            shutil.rmtree(img_dir)


def convert_to_df(similarities: typing.List) -> pd.DataFrame:
    """Convert similarities to pd.DataFrame to save as zntrack.plots.

    Parameters
    ----------
    similarities: typing.List
        contains similarities
    Returns
    -------
    df: pd.DataFrame
        contains a pd.Dataframe with the similarity.
    """
    df = pd.DataFrame({"similarities": similarities})
    df.index.name = "configuration_index"
    return df


@dataclass
class SOAPParameter:
    """Dataclass to store SOAP parameter used for representation.

    Attributes
    ----------
    r_cut: float
        cutoff radius of the soap descriptor in Angstrom
    n_max: int
        number of radial basis functions
    l_max: int
        maximum degree of spherical harmonics
    n_jobs: int
        number of parallel jobs to instantiate
    sigma: float
        The standard deviation of the gaussians used to expand the atomic density.
    rbf: str
        The radial basis functions to use
    weighting: dict
        Contains the options which control the weighting of the atomic density.
    """

    r_cut: float = 9.0
    n_max: int = 7
    l_max: int = 7
    n_jobs: int = -1
    sigma: float = 1.0
    rbf: str = "gto"
    weighting: dict = None


class SOAPParameterConverter(znjson.ConverterBase):
    """Converter class to encode and decode dictionaries and dataclasses."""

    level = 100
    representation = "soap_parameter_dataclass"
    instance = SOAPParameter

    def encode(self, obj: SOAPParameter) -> dict:
        """Encode dataclass to dictionary."""
        return dataclasses.asdict(obj)

    def decode(self, value: dict) -> SOAPParameter:
        """DEcode dictionary to dataclass."""
        return SOAPParameter(**value)


# znjson.config.register(SOAPParameterConverter)


def create_dataset(file: h5py.File, data, soap: SOAP, name: str):
    """Create an entry in the HDF5 dataset."""
    file.create_dataset(
        name,
        (
            len(data),
            len(data[0]),
            soap.get_number_of_features(),
        ),
    )


def write_dataset(
    file: h5py.File,
    data,
    name: str,
    soap: SOAP,
    n_jobs,
    desc: str,
    disable_tqdm: bool = False,
):
    """Write data to HDF5 dataset."""
    with trange((len(data) - 1), desc=desc, leave=True, disable=disable_tqdm) as pbar:
        for max_index, atoms in enumerate(data):
            file[name][max_index] = soap.create(atoms, n_jobs=n_jobs)
            pbar.update(1)


class ConfigurationComparison(base.IPSNode):
    """Base of comparison methods to compare atomic configurations.

    Attributes
    ----------
    reference: typing.Union[utils.helpers.UNION_ATOMS_OR_ATOMS_LST,
     utils.types.SupportsAtoms]
        reference configurations to compare analyte to
    analyte: typing.Union[
        utils.helpers.UNION_ATOMS_OR_ATOMS_LST, utils.types.SupportsAtoms
    ]
        analyte comparison to compare with reference
    similarities: zntrack.plots()
        in the end a csv file to save computed maximal similarities
    soap: typing.Union[dict, SOAPParameter]
        parameter to use for the SOAP descriptor
    result: typing.List[typing.List[float]]
        result of the comparison, all similarity computations
    node_name: str
        name of the node used within the dvc graph
    compile_with_jit: bool
        choose if kernel should be compiled with jit or not.
    memory: int
            How far back to look in the MMK vector.
    """

    reference: base.protocol.HasOrIsAtoms | None = zntrack.deps(None)
    analyte: base.protocol.HasOrIsAtoms | None = zntrack.deps(None)
    memory: int = zntrack.params(1000)
    similarities: pd.DataFrame = zntrack.plots()
    soap: typing.Union[dict, SOAPParameter] = zntrack.params(
        default_factory=SOAPParameter
    )
    result: typing.List[float] = zntrack.outs()

    _name_ = "ConfigurationComparison"
    use_jit: bool = zntrack.params(True)

    soap_file: pathlib.Path = zntrack.outs_path(
        zntrack.nwd / "soap_representation.hdf5"
    )

    def __post_init__(self):
        if self.soap is None:
            soap = {}
        if not self.state.state == NodeStatusEnum.RUNNING:
            self.soap = SOAPParameter(**soap)

    def save_representation(self):
        """Save the SOAP descriptor representation as hdf5 file to save RAM.

        It will create SOAP descriptor for each configurations
         and save them ordered in a hdf5 file.
        """
        load_analyte = False
        species = [int(x) for x in set(self.analyte[0].get_atomic_numbers())]
        _soap = SOAP(
            species=species,
            periodic=False,  # any(self.analyte[0].pbc),
            r_cut=self.soap.r_cut,
            n_max=self.soap.n_max,
            l_max=self.soap.l_max,
            sigma=self.soap.sigma,
            rbf=self.soap.rbf,
            weighting=self.soap.weighting,
        )
        if self.reference is None:
            with h5py.File(self.soap_file, "w") as representation_file:
                create_dataset(
                    file=representation_file, data=self.analyte, soap=_soap, name="soap"
                )
                write_dataset(
                    file=representation_file,
                    data=self.analyte,
                    name="soap",
                    soap=_soap,
                    n_jobs=self.soap.n_jobs,
                    desc="Writing SOAP",
                    disable_tqdm=self.disable_tqdm,
                )
        else:
            with h5py.File(self.soap_file, "a") as representation_file:
                create_dataset(
                    file=representation_file,
                    data=self.reference,
                    soap=_soap,
                    name="soap_reference",
                )
                write_dataset(
                    file=representation_file,
                    data=self.reference,
                    name="soap_reference",
                    soap=_soap,
                    n_jobs=self.soap.n_jobs,
                    desc="Writing SOAP reference",
                    disable_tqdm=self.disable_tqdm,
                )

                if not load_analyte:
                    create_dataset(
                        file=representation_file,
                        data=self.analyte,
                        soap=_soap,
                        name="soap_analyte",
                    )

                    write_dataset(
                        file=representation_file,
                        data=self.analyte,
                        name="soap_analyte",
                        soap=_soap,
                        n_jobs=self.soap.n_jobs,
                        desc="Writing SOAP analyte",
                        disable_tqdm=self.disable_tqdm,
                    )

    def _save_plots(self, max_index, interval: int = 1000):
        """Save the ZnTrack plots at regular intervals."""
        if max_index % interval == 0:
            self.similarities = convert_to_df(self.result)
            type(self).similarities.save(self)

    def unlink_database(self):
        """Remove the database."""
        if pathlib.Path(self.soap_file).is_file():
            pathlib.Path(self.soap_file).unlink()

    def run(self):
        """Run the configuration comparison.

        Use the chosen comparison method to compute the similarity between
        configurations and save the result as a csv file.
        """
        # remove "soap_reference" from HDF5, do not write "soap_analyte"
        remove_database = True
        disable_tqdm = False
        self.result = []
        self.save_representation()
        if self.reference is None:
            with h5py.File(self.soap_file, "r") as representation_file:
                with trange(
                    (len(self.analyte) - 1),
                    desc="Comparing",
                    leave=True,
                    disable=disable_tqdm,
                ) as pbar:
                    for max_index, _atoms in enumerate(self.analyte):
                        if max_index == 0:
                            continue
                        reference_soap = representation_file["soap"][:max_index]
                        # if max_index <= self.memory:
                        #     reference_soap = representation_file["soap"][:max_index]
                        # else:
                        #     reference_soap = representation_file["soap"][
                        #         max_index - self.memory : max_index
                        #     ]
                        analyte_soap = representation_file["soap"][max_index]
                        comparison = self.compare(reference_soap, analyte_soap)
                        self.result.append(float(comparison.numpy()))
                        self._save_plots(max_index)
                        pbar.update(1)
        else:
            with h5py.File(self.soap_file, "r") as representation_file:
                with trange(
                    (len(self.analyte)),
                    desc="Comparing",
                    leave=True,
                    disable=disable_tqdm,
                ) as pbar:
                    for max_index, _atoms in enumerate(self.analyte):
                        reference_soap = representation_file["soap_reference"]
                        # if max_index <= self.memory:
                        #     reference_soap = representation_file["soap_reference"][
                        #         :max_index
                        #     ]
                        # else:
                        #     reference_soap = representation_file["soap_reference"][
                        #         max_index - self.memory : max_index
                        #     ]
                        analyte_soap = representation_file["soap_analyte"][max_index]
                        comparison = self.compare(reference_soap, analyte_soap)
                        self.result.append(float(comparison.numpy()))
                        self._save_plots(max_index)
                        pbar.update(1)
        self.similarities = convert_to_df(self.result)
        with h5py.File(self.soap_file, "a") as representation_file:
            with contextlib.suppress(KeyError):
                del representation_file["soap_reference"]
        if remove_database:
            self.unlink_database()

    def compare(self, reference: np.ndarray, analyte: np.ndarray) -> tf.Tensor:
        """Actual comparison method to use for similarity computation.

        Parameters
        ----------
        reference: np.ndarray
            reference representations to compare of shape (configuration, atoms, x)
        analyte: np.ndarray
            one representation to compare with the reference of shape (atoms, x).

        Returns
        -------
        maximum: tf.Tensor
            Similarity between analyte and reference.
        """
        raise NotImplementedError
