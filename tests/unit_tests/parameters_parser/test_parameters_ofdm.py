import unittest
from typing import List
import os

import numpy as np

from parameters_parser.parameters_ofdm import (
    ParametersOfdm, ResourceType, OfdmSymbolConfig, GuardInterval, MultipleRes,
    ResourcePattern)


class TestOfdmSymbolConfigResourcesParser(unittest.TestCase):
    def setUp(self):
        self.params = ParametersOfdm()

    def test_throwExceptionIfStringNotSupported(self) -> None:
        unsupported_character = 'g'
        ofdm_symbol_structure = '16d4g' + unsupported_character

        self.assertRaises(ValueError,
                          lambda: self.params.read_ofdm_symbol_resources(ofdm_symbol_structure))

    def test_oneRef_oneData_oneNull(self) -> None:
        ofdm_symbol_structure = 'rdn'
        res_types: List[MultipleRes] = [
            MultipleRes(ResourceType.REFERENCE, 1),
            MultipleRes(ResourceType.DATA, 1),
            MultipleRes(ResourceType.NULL, 1)
        ]
        ofdm_symbol_structure_parsed: List[ResourcePattern] = [
            ResourcePattern(MultipleRes=res_types, number=1)
        ]

        self.assertListEqual(self.params.read_ofdm_symbol_resources(ofdm_symbol_structure),
                             [ofdm_symbol_structure_parsed])

    def test_oneRef_multipleData_oneRef(self) -> None:
        ofdm_symbol_structure = 'r16dr'

        res_types: List[MultipleRes] = []
        res_types.append(
            MultipleRes(ResourceType=ResourceType.REFERENCE, number=1)
        )
        res_types.append(
            MultipleRes(ResourceType=ResourceType.DATA, number=16)
        )
        res_types.append(
            MultipleRes(ResourceType=ResourceType.REFERENCE, number=1)
        )

        pattern = ResourcePattern(res_types, 1)

        self.assertListEqual(self.params.read_ofdm_symbol_resources(ofdm_symbol_structure),
                             [[pattern]])

    def test_tooManyOfdmSymbolConfigs(self) -> None:
        ofdm_symbol_structure = '1,2,3,4,5,6,7,8,9,10,11,12'
        self.assertRaises(
            ValueError, 
            lambda: self.params.read_ofdm_symbol_resources(ofdm_symbol_structure)
        )

    def test_threeOfdmSymbolConfigs_twoRefTwoDataEach(self) -> None:
        ofdm_symbol_structure = '2(rd),2(rd),2(rd)'

        res_types: List[MultipleRes] = [
            MultipleRes(ResourceType.REFERENCE, 1),
            MultipleRes(ResourceType.DATA, 1)
        ]
        first_ofdm_symbol = [ResourcePattern(
            res_types, number=2
        )]
        ofdm_symbol_structure_parsed: List[ResourcePattern] = [
            first_ofdm_symbol, first_ofdm_symbol, first_ofdm_symbol]

        self.assertListEqual(self.params.read_ofdm_symbol_resources(ofdm_symbol_structure),
                             ofdm_symbol_structure_parsed)

    def test_mappingResourcesToSymbolConfigs(self) -> None:
        res: List[ResourcePattern] = [
            ResourcePattern(
                MultipleRes=[
                    MultipleRes(ResourceType.DATA, 1),
                    MultipleRes(ResourceType.REFERENCE, 1)],
                number=1)]
        ofdm_symbol_configs = self.params.map_resource_types_to_symbols(res)

        self.assertListEqual(
            [OfdmSymbolConfig(resource_types=res[0], no_samples=self.params.fft_size)],
            ofdm_symbol_configs
        )


class TestCpSymbolParser(unittest.TestCase):
    def setUp(self) -> None:
        self.params = ParametersOfdm()
        self.params.fft_size = 2048
        self.three_ofdm_symbols: List[OfdmSymbolConfig] = (
            [OfdmSymbolConfig(), OfdmSymbolConfig(), OfdmSymbolConfig()]
        )
        self.params.cp_ratio = np.array([0.07, 0.073])

    def test_noOfdmSymbolConfigsUnequalNoCps(self) -> None:
        ofdm_symbols: List[OfdmSymbolConfig] = [
            OfdmSymbolConfig(), OfdmSymbolConfig(), OfdmSymbolConfig()]
        cp_lengths_str = 'c1,c'

        self.assertRaises(ValueError,
                          lambda: self.params.read_cp_lengths(cp_lengths_str, ofdm_symbols))

    def test_unsupportedCharacter(self) -> None:
        cp_lengths_str = 'c,c,g'

        self.assertRaises(ValueError,
                          lambda: self.params.read_cp_lengths(cp_lengths_str, self.three_ofdm_symbols))

    def test_cpRatio_doesNotExist(self) -> None:
        cp_lengths_str = 'c,c4,c'

        self.assertRaises(ValueError,
                          lambda: self.params.read_cp_lengths(cp_lengths_str, self.three_ofdm_symbols))

    def test_cpRatio_idxNotDigit(self) -> None:
        cp_lengths_str = 'ct,c,c'

        self.assertRaises(ValueError,
                          lambda: self.params.read_cp_lengths(cp_lengths_str, self.three_ofdm_symbols))

    def test_noSamplesCalculation(self) -> None:
        cp_lengths_str = 'c1,c1,c2'
        expected_cp_samples = [
            int(np.around(self.params.cp_ratio[0]*self.params.fft_size)),
            int(np.around(self.params.cp_ratio[0]*self.params.fft_size)),
            int(np.around(self.params.cp_ratio[1]*self.params.fft_size))
        ]
        self.params.read_cp_lengths(cp_lengths_str, self.three_ofdm_symbols)

        for cp_samples, ofdm_symbol in zip(expected_cp_samples, self.three_ofdm_symbols):
            self.assertEqual(
                cp_samples,
                ofdm_symbol.cyclic_prefix_samples
            )


class TestFrameStructureParser(unittest.TestCase):
    def setUp(self) -> None:
        self.params = ParametersOfdm()
        self.params.cp_ratio = np.array([0.05, 0.07])
        self.params.fft_size = 2048
        self.params.number_occupied_subcarriers = 1200
        self.params.frame_guard_interval = .001
        self.params.subcarrier_spacing = 15e3
        self.params.ofdm_symbol_resources_mapping: List[ResourcePattern] = [
            ResourcePattern(
                MultipleRes=[
                    MultipleRes(ResourceType.DATA, 1),
                    MultipleRes(ResourceType.REFERENCE, 2)],
                number=1),
            ResourcePattern(
                MultipleRes=[
                    MultipleRes(ResourceType.DATA, 3)],
                number=1)
        ]
        self.params.ofdm_symbol_configs: List[OfdmSymbolConfig] = (
            self.params.map_resource_types_to_symbols(self.params.ofdm_symbol_resources_mapping)
        )
        self.params.read_cp_lengths('c1,c2', self.params.ofdm_symbol_configs)

    def test_firstOfdmSymbol(self) -> None:
        frame_structure = '1'

        self.assertListEqual(self.params.read_frame_structure(frame_structure),
                             [self.params.ofdm_symbol_configs[0]])

    def test_firstFirstSecond(self) -> None:
        frame_structure = '112'

        self.assertListEqual(self.params.read_frame_structure(frame_structure),
                             [self.params.ofdm_symbol_configs[0],
                              self.params.ofdm_symbol_configs[0],
                              self.params.ofdm_symbol_configs[1]])

    def test_repetitionOfmSymbols(self) -> None:
        frame_structure = '3(1)'

        self.assertListEqual(self.params.read_frame_structure(frame_structure),
                             [self.params.ofdm_symbol_configs[0],
                              self.params.ofdm_symbol_configs[0],
                              self.params.ofdm_symbol_configs[0]])

    def test_firstOfdmSymbol_guard(self) -> None:
        frame_structure = '1,g'
        no_samples_gi = int(
            np.around(
                self.params.frame_guard_interval
                * self.params.subcarrier_spacing
                * self.params.fft_size
            )
        )
        self.assertListEqual(self.params.read_frame_structure(frame_structure),
                             [self.params.ofdm_symbol_configs[0],
                              GuardInterval(no_samples=no_samples_gi)])


class TestParametersParserOfdm(unittest.TestCase):
    def _dump_params(self) -> str:
        filename = "parameters_ofdm_temp.ini"
        params_content = """[General]
technology = ofdm

[Modulation]
subcarrier_spacing = 15e3
fft_size = 2048
number_occupied_subcarriers = 1200

cp_ratio = 0.0703125, .078125
precoding = DFT
modulation_order = 16

oversampling_factor = 1
dc_suppression = False

[Receiver]
channel_estimation = ideal_postamble
equalization = ZF

[MIMO]
mimo_scheme = none

[Frame]
frame_guard_interval = .001
ofdm_symbol_resources = 100(r11d), 100(6r6d)
cp_length = c1,c2
frame_structure = 10(12),1,g 
            """
        with open(filename, 'w') as f:
            f.write(params_content)

        return filename

    def setUp(self) -> None:
        self.params_file = self._dump_params()
        self.params = ParametersOfdm()
        self.params.read_params(self.params_file)

    def tearDown(self) -> None:
        os.remove(self.params_file)

    def test_noSymbolResourcesCheck(self) -> None:
        p = ParametersOfdm(1, 1)
        p.number_occupied_subcarriers = 3
        res_pattern = [[ResourcePattern(
            MultipleRes=[MultipleRes(ResourceType.NULL, 1)],
            number=1
        )]]
        self.assertRaises(ValueError, lambda: p._check_no_ofdm_symbol_resources(res_pattern))

    def test_modulationParametersCorrectlyParsed(self) -> None:
        self.assertEqual(self.params.subcarrier_spacing, 15e3)
        self.assertEqual(self.params.fft_size, 2048)
        self.assertEqual(self.params.number_occupied_subcarriers, 1200)

        np.testing.assert_array_almost_equal(self.params.cp_ratio, np.array([.0703125, .078125]))
        self.assertEqual(self.params.precoding, "DFT")
        self.assertEqual(self.params.modulation_order, 16)

        self.assertEqual(self.params.oversampling_factor, 1)

    def test_receiverParamsCorrectlyParsed(self) -> None:
        self.assertEqual(self.params.channel_estimation, "IDEAL_POSTAMBLE")
        self.assertEqual(self.params.equalization, "ZF")

    def test_frameCorrectlyParsed(self) -> None:
        no_cp_samples_first_ofdm_symbol = int(np.around(
            self.params.cp_ratio[0] * self.params.fft_size
        ))
        no_cp_samples_second_ofdm_symbol = int(np.around(
            self.params.cp_ratio[1] * self.params.fft_size
        ))
        no_samples_gi = int(
            np.around(
                self.params.frame_guard_interval
                * self.params.subcarrier_spacing
                * self.params.fft_size
            )
        )
        res_types_pattern_first_ofdm_symbol = [
            MultipleRes(ResourceType.REFERENCE, 1),
            MultipleRes(ResourceType.DATA, 11)
        ]
        res_pattern_first_ofdm_symbol = ResourcePattern(
            MultipleRes=res_types_pattern_first_ofdm_symbol,
            number=100
        )

        res_types_pattern_second_ofdm_symbol = [
            MultipleRes(ResourceType.REFERENCE, 6),
            MultipleRes(ResourceType.DATA, 6)
        ]
        res_pattern_second_ofdm_symbol = ResourcePattern(
            MultipleRes=res_types_pattern_second_ofdm_symbol,
            number=100
        )

        first_ofdm_symbol_config = OfdmSymbolConfig(
            cyclic_prefix_samples=no_cp_samples_first_ofdm_symbol,
            no_samples=self.params.fft_size,
            resource_types=[res_pattern_first_ofdm_symbol])

        second_ofdm_symbol_config = OfdmSymbolConfig(
            cyclic_prefix_samples=no_cp_samples_second_ofdm_symbol,
            no_samples=self.params.fft_size,
            resource_types=[res_pattern_second_ofdm_symbol])

        guard_interval = GuardInterval(no_samples=no_samples_gi)
        frame_structure = (
            10*[first_ofdm_symbol_config, second_ofdm_symbol_config]
            + [first_ofdm_symbol_config] + [guard_interval])

        self.assertEqual(len(frame_structure), len(self.params.frame_structure))
        self.assertListEqual(frame_structure, self.params.frame_structure)