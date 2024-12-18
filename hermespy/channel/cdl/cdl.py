# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Mapping, Set

from h5py import Group
import numpy as np

from hermespy.core import HDFSerializable, SerializableEnum
from ..channel import Channel, ChannelRealization, ChannelSampleHook, LinkState
from ..consistent import ConsistentGenerator, ConsistentUniform, ConsistentRealization
from .cluster_delay_lines import ClusterDelayLineSample, ClusterDelayLineRealization

__author__ = "Jan Adler"
__copyright__ = "Copyright 2024, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.3.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


class CDLType(SerializableEnum):
    """Type of the static cluster delay line model."""

    A = 0
    B = 1
    C = 2
    D = 3
    E = 4


# ETSI TR 138.901 v17.0.0 Table 7.7.1-1
CDL_Cluster_Parameters: Mapping[CDLType, np.ndarray] = {
    #    Delay   Power   AOD     AOA     ZOD     ZOA
    #    [ns]    [dB]    [deg]   [deg]   [deg]   [deg]
    # Table 7.7.1-1: CDL-A
    CDLType.A: np.array(
        [
            [0.0000, -13.4, -178.10, +51.30, +050.2, 125.4],
            [0.3819, +0.00, -4.2000, -152.7, +093.2, 091.3],
            [0.4025, -2.20, -4.2000, -152.7, +093.2, 091.3],
            [0.5868, -4.00, -4.2000, -152.7, +093.2, 091.3],
            [0.4610, -6.00, +90.200, +76.60, +122.0, 094.0],
            [0.5375, -8.20, +90.200, +76.60, +122.0, 094.0],
            [0.6708, -9.90, +90.200, +76.60, +122.0, 094.0],
            [0.5750, -10.5, +121.50, -1.800, +150.2, 047.1],
            [0.7618, -7.50, -81.700, -41.90, +055.2, 056.0],
            [1.5375, -15.9, +158.40, +94.20, +026.4, 030.1],
            [1.8978, -6.60, -83.000, +51.90, +126.4, 058.8],
            [2.2242, -16.7, +134.80, -115.9, +171.6, 026.0],
            [2.1718, -12.4, -153.00, +26.60, +151.4, 049.2],
            [2.4942, -15.2, -172.00, +76.60, +157.2, 143.1],
            [2.5119, -10.8, -129.90, -07.00, +047.2, 117.4],
            [3.0582, -11.3, -136.00, -23.00, +040.4, 122.7],
            [4.0810, -12.7, +165.40, -47.20, +043.3, 123.2],
            [4.4579, -16.2, +148.40, +110.4, +161.8, 032.6],
            [4.5695, -18.3, +132.70, +144.5, +010.8, 027.2],
            [4.7966, -18.9, -118.60, +155.3, +016.7, 015.2],
            [5.0066, -16.6, -154.10, +102.0, +171.7, 146.0],
            [5.3043, -19.9, +126.50, -151.8, +022.7, 150.7],
            [9.6586, -29.7, -56.200, +55.20, +144.9, 156.1],
        ],
        dtype=np.float64,
    ),
    # Table 7.7.1-2: CDL-B
    CDLType.B: np.array(
        [
            [0.0000, +0.00, +009.30, -173.3, +105.8, +78.9],
            [0.1072, -02.2, +009.30, -173.3, +105.8, +78.9],
            [0.2155, -04.0, +009.30, -173.3, +105.8, +78.9],
            [0.2095, -03.2, -034.10, +125.5, +115.3, +63.3],
            [0.2870, -09.8, -065.40, -088.0, +119.3, +59.9],
            [0.2986, -01.2, -011.40, +155.1, +103.2, +67.5],
            [0.3752, -03.4, -011.40, +155.1, +103.2, +67.5],
            [0.5055, -05.2, -011.40, +155.1, +103.2, +67.5],
            [0.3681, -07.6, -067.20, -089.8, +118.2, +82.6],
            [0.3697, -03.0, +052.50, +132.1, +102.0, +66.3],
            [0.5700, -08.9, -072.00, -083.6, +100.4, +61.6],
            [0.5283, -09.0, +074.30, +095.3, +098.3, +58.0],
            [1.1021, -4.80, -052.20, +103.7, +103.4, +78.2],
            [1.2756, -5.70, -050.50, -087.8, +102.5, +82.0],
            [1.5474, -7.50, +061.40, -092.5, +101.4, +62.4],
            [1.7842, -1.90, +030.60, -139.1, +103.0, +78.0],
            [2.0169, -7.60, -072.50, -090.6, +100.0, +60.9],
            [2.8294, -12.2, -090.60, +058.6, +115.2, +82.9],
            [3.0219, -9.80, -077.60, -079.0, +100.5, +60.8],
            [3.6187, -11.4, -082.60, +065.8, +119.6, +57.3],
            [4.1067, -14.9, -103.60, +052.7, +118.7, +59.9],
            [4.2790, -9.20, +075.60, +088.7, +117.8, +60.1],
            [4.7834, -11.3, -077.60, -060.4, +115.7, +62.3],
        ],
        dtype=np.float64,
    ),
    # Table 7.7.1-3: CDL-C
    CDLType.C: np.array(
        [
            [0.0000, -04.4, -046.6, -101.0, +097.2, +087.6],
            [0.2099, -01.2, -022.8, +120.0, +098.6, +072.1],
            [0.2219, -03.5, -022.8, +120.0, +098.6, +072.1],
            [0.2329, -05.2, -022.8, +120.0, +098.6, +072.1],
            [0.2176, -02.5, -040.7, -127.5, +100.6, +070.1],
            [0.6366, +00.0, +000.3, +170.4, +099.2, +075.3],
            [0.6448, -02.2, +000.3, +170.4, +099.2, +075.3],
            [0.6560, -03.9, +000.3, +170.4, +099.2, +075.3],
            [0.6584, -07.4, +073.1, +055.4, +105.2, +067.4],
            [0.7935, -07.1, -064.5, +066.5, +095.3, +063.8],
            [0.8213, -10.7, +080.2, -048.1, +106.1, +071.4],
            [0.9336, -11.1, -097.1, +046.9, +093.5, +060.5],
            [1.2285, -05.1, -055.3, +068.1, +103.7, +090.6],
            [1.3083, -06.8, -064.3, -068.7, +104.2, +060.1],
            [2.1704, -08.7, -078.5, +081.5, +093.0, +061.0],
            [2.7105, -13.2, +102.7, +030.7, +104.2, +100.7],
            [4.2589, -13.9, +099.2, -016.4, +094.9, +062.3],
            [4.6003, -13.9, +088.8, +003.8, +093.1, +066.7],
            [5.4902, -15.8, -101.9, -013.7, +092.2, +052.9],
            [5.6077, -17.1, +092.2, +009.7, +106.7, +061.8],
            [6.3065, -16.0, +093.3, +005.6, +093.0, +051.9],
            [6.6374, -15.7, +106.6, +000.7, +092.9, +061.7],
            [7.0427, -21.6, +119.5, -021.9, +105.2, +058.0],
            [8.6523, -22.8, -123.8, +033.6, +107.8, +057.0],
        ],
        dtype=np.float64,
    ),
    # Table 7.7.1-4: CDL-D
    CDLType.D: np.array(
        [
            [00.000, -00.2, +000.0, -180.0, +098.5, +81.5],
            [00.000, -13.5, +000.0, -180.0, +098.5, +81.5],
            [00.035, -18.8, +089.2, +089.2, +085.5, +86.9],
            [00.612, -21.0, +089.2, +089.2, +085.5, +86.9],
            [01.363, -22.8, +089.2, +089.2, +085.5, +86.9],
            [01.405, -17.9, +013.0, +163.0, +097.5, +79.4],
            [01.804, -20.1, +013.0, +163.0, +097.5, +79.4],
            [02.596, -21.9, +013.0, +163.0, +097.5, +79.4],
            [01.775, -22.9, +034.6, -137.0, +098.5, +78.2],
            [04.042, -27.8, -064.5, +074.5, +088.4, +73.6],
            [07.937, -23.6, -032.9, +127.7, +091.3, +78.3],
            [09.424, -24.8, +052.6, -119.6, +103.8, +87.0],
            [09.708, -30.0, -132.1, -009.1, +080.3, +70.6],
            [12.525, -27.7, +077.2, -083.8, +086.5, +72.9],
        ],
        dtype=np.float64,
    ),
    # Table 7.7.1-5: CDL-E
    CDLType.E: np.array(
        [
            [00.0000, -00.03, +00.0, -180.0, +099.6, +80.4],
            [00.0000, -22.03, +00.0, -180.0, +099.6, +80.4],
            [00.5133, -15.80, +57.5, +018.2, +104.2, +80.4],
            [00.5440, -18.10, +57.5, +018.2, +104.2, +80.4],
            [00.5630, -19.80, +57.5, +018.2, +104.2, +80.4],
            [00.5440, -22.90, -20.1, +101.8, +099.4, +80.8],
            [00.7112, -22.40, +16.2, +112.9, +100.8, +86.3],
            [1.90920, -18.60, +09.3, -155.5, +098.8, +82.7],
            [1.92930, -20.80, +09.3, -155.5, +098.8, +82.7],
            [1.95890, -22.60, +09.3, -155.5, +098.8, +82.7],
            [2.64260, -22.30, +19.0, -143.3, +100.8, +82.9],
            [3.71360, -25.60, +32.7, -094.7, +096.4, +88.0],
            [5.45240, -20.20, +00.5, +147.0, +098.9, +81.0],
            [12.0034, -29.80, +55.9, -036.2, +095.6, +88.6],
            [20.6419, -29.20, +57.6, -026.0, +104.6, +78.3],
        ],
        dtype=np.float64,
    ),
}


CDL_Per_Cluster_Parameters: np.ndarray = np.array(
    [
        [05.0, 11 - 0, 3.0, 3.0, 10.0, False],  # Table 7.7.1-1: CDL-A
        [10.0, 22.0, 3.0, 7.0, 08.0, False],  # Table 7.7.1-2: CDL-B
        [02.0, 15.0, 3.0, 7.0, 07.0, False],  # Table 7.7.1-2: CDL-C
        [05.0, 08.0, 3.0, 3.0, 11.0, True],  # Table 7.7.1-4: CDL-D
        [05.0, 11.0, 3.0, 7.0, 08.0, True],  # Table 7.7.1-5: CDL-E
    ]
)


class CDLRealization(ChannelRealization[ClusterDelayLineSample]):
    """Realization of a static cluster delay line model for link-level simulations.

    Generated by the :meth:`realize<CDL.realize>` method of the :class:`CDL` class.
    """

    def __init__(
        self,
        type: CDLType,
        rms_delay: float,
        rayleigh_factor: float,
        angle_coupling_indices: np.ndarray,
        consistent_realization: ConsistentRealization,
        xpr_phase: ConsistentUniform,
        sample_hooks: Set[ChannelSampleHook[ClusterDelayLineSample]],
        gain: float,
    ) -> None:
        """
        Args:

            type: Type of the cluster delay line model.
            rms_delay: Root mean square delay spread of the channel.
            rayleigh_factor: Rayleigh K-factor of the channel.
            angle_coupling_indices: Indices for the coupling of rays within a cluster.
            consistent_realization: Realization of the consistent distribution.
            xpr_phase: Realization of the cross-polarization phase.
            sample_hooks: Hooks to be called after the channel sample has been generated.
            gain: Linear channel gain factor.
        """

        # Initialize base class
        ChannelRealization.__init__(self, sample_hooks, gain)

        # Store parameters
        self.__type = type
        self.__rms_delay = rms_delay
        self.__rayleigh_factor = rayleigh_factor
        self.__angle_coupling_indices = angle_coupling_indices
        self.__consistent_realization = consistent_realization
        self.__xpr_phase = xpr_phase

    def _sample(self, state: LinkState) -> ClusterDelayLineSample:
        # Sample the consistent distribution
        consistent_sample = self.__consistent_realization.sample(
            state.transmitter.position, state.receiver.position
        )

        # Fetch the cluster parameters
        parameters = CDL_Cluster_Parameters[self.__type]
        per_cluster_parameters = CDL_Per_Cluster_Parameters[self.__type.value, :]
        normalized_cluster_delays = parameters[:, 0]
        cluster_powers = 10 ** (parameters[:, 1] / 10)
        cluster_aods = parameters[:, 2]
        cluster_aoas = parameters[:, 3]
        cluster_zods = parameters[:, 4]
        cluster_zoas = parameters[:, 5]
        rms_asd_spreads = per_cluster_parameters[0]
        rms_asa_spreads = per_cluster_parameters[1]
        rms_zsd_spreads = per_cluster_parameters[2]
        rms_zsa_spreads = per_cluster_parameters[3]
        XPR_dB = per_cluster_parameters[4]
        line_of_sight = bool(per_cluster_parameters[5])

        # Generate cluster delays
        cluster_delays = self.__rms_delay * normalized_cluster_delays

        # Step 1: Generate departure and arrival angles
        # Equation 7.7-0a in ETSI TR 138.901 v17.0.0
        ray_aods = np.add.outer(
            cluster_aods, rms_asd_spreads * ClusterDelayLineRealization._ray_offset_angles
        )
        ray_aoas = np.add.outer(
            cluster_aoas, rms_asa_spreads * ClusterDelayLineRealization._ray_offset_angles
        )
        ray_zods = np.add.outer(
            cluster_zods, rms_zsd_spreads * ClusterDelayLineRealization._ray_offset_angles
        )
        ray_zoas = np.add.outer(
            cluster_zoas, rms_zsa_spreads * ClusterDelayLineRealization._ray_offset_angles
        )

        # Step 2: Coupling of rays within a cluster for both azimuth and zenith
        # Equation 7.7-0b in ETSI TR 138.901 v17.0.0
        shuffled_ray_aods = np.take_along_axis(
            ray_aods, self.__angle_coupling_indices[0, :], axis=1
        )
        shuffled_ray_aoas = np.take_along_axis(
            ray_aoas, self.__angle_coupling_indices[1, :], axis=1
        )
        shuffled_ray_zods = np.take_along_axis(
            ray_zods, self.__angle_coupling_indices[2, :], axis=1
        )
        shuffled_ray_zoas = np.take_along_axis(
            ray_zoas, self.__angle_coupling_indices[3, :], axis=1
        )

        # Draw initial random phases (step 10)
        # A single 2x2 slice represents the jones matrix transforming the polarization of a single ray
        cross_polarization_factor = 10 ** (XPR_dB / 10)
        polarization_transformations = np.exp(
            2j * np.pi * self.__xpr_phase.sample(consistent_sample)
        )
        polarization_transformations[0, 1, ::] *= cross_polarization_factor
        polarization_transformations[1, 0, ::] *= cross_polarization_factor

        return ClusterDelayLineSample(
            line_of_sight,
            self.__rayleigh_factor,
            np.pi / 180 * shuffled_ray_aoas,
            np.pi / 180 * shuffled_ray_zoas,
            np.pi / 180 * shuffled_ray_aods,
            np.pi / 180 * shuffled_ray_zods,
            0,
            cluster_delays,
            self.__rms_delay,
            cluster_powers,
            polarization_transformations,
            state,
        )

    def _reciprocal_sample(
        self, sample: ClusterDelayLineSample, state: LinkState
    ) -> ClusterDelayLineSample:
        return sample.reciprocal(state)

    def to_HDF(self, group: Group) -> None:

        group.attrs["type"] = self.__type.value
        group.attrs["rms_delay"] = self.__rms_delay
        group.attrs["rayleigh_factor"] = self.__rayleigh_factor
        HDFSerializable._write_dataset(
            group, "angle_coupling_indices", self.__angle_coupling_indices
        )
        self.__consistent_realization.to_HDF(
            HDFSerializable._create_group(group, "consistent_realization")
        )
        group.attrs["gain"] = self.gain

    @staticmethod
    def From_HDF(
        group: Group,
        xpr_phase: ConsistentUniform,
        sample_hooks: Set[ChannelSampleHook[ClusterDelayLineSample]],
    ) -> CDLRealization:

        type = CDLType(group.attrs["type"])
        rms_delay = group.attrs["rms_delay"]
        rayleigh_factor = group.attrs["rayleigh_factor"]
        angle_coupling_indices = np.array(group["angle_coupling_indices"], dtype=np.int_)
        consistent_realization = ConsistentRealization.from_HDF(group["consistent_realization"])
        gain = group.attrs["gain"]

        return CDLRealization(
            type,
            rms_delay,
            rayleigh_factor,
            angle_coupling_indices,
            consistent_realization,
            xpr_phase,
            sample_hooks,
            gain,
        )


class CDL(Channel[CDLRealization, ClusterDelayLineSample]):
    """Static cluster delay line model for link-level simulations."""

    yaml_tag = "CDL"

    __model_type: CDLType
    __rms_delay: float
    __rayleigh_factor: float
    __decorrelation_distance: float

    def __init__(
        self,
        model_type: CDLType,
        rms_delay: float,
        rayleigh_factor: float = 0.0,
        decorrelation_distance: float = 30.0,
        **kwargs,
    ) -> None:
        """
        Args:
            model_type: Type of the cluster delay line model.
            rms_delay: Root mean square delay spread of the channel.
            rayleigh_factor: Rayleigh K-factor of the channel.
            decorrelation_distance: Decorrelation distance of the channel.
            \**kwargs: Additional parameters for the base class.
        """

        # Initialize base class
        Channel.__init__(self, **kwargs)

        # Store parameters
        self.__model_type = model_type
        self.rms_delay = rms_delay
        self.rayleigh_factor = rayleigh_factor
        self.decorrelation_distance = decorrelation_distance

        self.__consistent_generator = ConsistentGenerator(self)
        self.__xpr_phase = ConsistentUniform(
            self.__consistent_generator,
            (
                2,
                2,
                CDL_Cluster_Parameters[model_type].shape[0],
                ClusterDelayLineRealization._ray_offset_angles.size,
            ),
        )

    @property
    def model_type(self) -> CDLType:
        """Type of the cluster delay line model."""

        return self.__model_type

    @property
    def rms_delay(self) -> float:
        """Root mean square delay spread of the channel.

        Raises:

            ValuError: If the delay spread is negative.
        """

        return self.__rms_delay

    @rms_delay.setter
    def rms_delay(self, value: float) -> None:
        if value < 0:
            raise ValueError("The delay spread must be non-negative.")

        self.__rms_delay = value

    @property
    def rayleigh_factor(self) -> float:
        """Rayleigh K-factor of the channel.

        Raises:

            ValueError: If the K-factor is negative.
        """

        return self.__rayleigh_factor

    @rayleigh_factor.setter
    def rayleigh_factor(self, value: float) -> None:
        if value < 0:
            raise ValueError("The K-factor must be non-negative.")

        self.__rayleigh_factor = value

    @property
    def decorrelation_distance(self) -> float:
        """Decorrelation distance of the channel.

        Raises:

            ValueError: If the decorrelation distance is negative.
        """

        return self.__decorrelation_distance

    @decorrelation_distance.setter
    def decorrelation_distance(self, value: float) -> None:
        if value < 0:
            raise ValueError("The decorrelation distance must be non-negative.")

        self.__decorrelation_distance = value

    def _realize(self) -> CDLRealization:

        angle_candidate_indices = np.arange(ClusterDelayLineRealization._ray_offset_angles.size)
        angle_coupling_indices = np.array(
            [
                [
                    self._rng.permutation(angle_candidate_indices)
                    for _ in range(CDL_Cluster_Parameters[self.model_type].shape[0])
                ]
                for _ in range(4)
            ]
        )

        return CDLRealization(
            self.model_type,
            self.rms_delay,
            self.rayleigh_factor,
            angle_coupling_indices,
            self.__consistent_generator.realize(self.decorrelation_distance),
            self.__xpr_phase,
            self.sample_hooks,
            self.gain,
        )

    def recall_realization(self, group: Group) -> CDLRealization:
        return CDLRealization.From_HDF(group, self.__xpr_phase, self.sample_hooks)
