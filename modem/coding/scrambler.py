from typing import List
from collections import deque
import numpy as np
from .encoder import Encoder, ParametersEncoder


class PseudoRandomGenerator:
    """A generator for pseudo-random bit sequences.

    See also TS 138 211 section 5.2.1 for details.
    """

    __queue_x1: deque
    __queue_x2: deque
    __initial_queue_x1: deque
    __initial_queue_x2: deque

    def __init__(self, init_sequence: np.array, offset: int = 1600) -> None:
        """Class initialization.

        Generators with identical initialization will output identical random sequences!

        Args:
            init_sequence(np.array):
                A sequence of 31 bits initializing the generator.

            offset(int):
                Gold sequence parameter controlling the sequence offset.
        """

        # The required sequence buffer length is inferred from the length of the init sequence
        m = init_sequence.shape[0]

        # Make sure the buffer / init sequence is at least 4 bits lon
        if m < 4:
            raise ValueError("The init sequence must contain at least 4 bits")

        # Init the first fifo queue as [1 0 0 ... 0]
        self.__queue_x1 = deque(np.zeros(m, dtype=int), m)
        self.__queue_x1.append(1)

        # Init the second fifo queue by the provided init sequence
        self.__queue_x2 = deque(init_sequence, m)  # TODO: Flip queue content

        # Fast-forward the queues to compensate for the offset
        for _ in range(offset - m):

            self.__forward_x1()
            self.__forward_x2()

        # Store the initial queues in order to reset the generator to n = 0
        self.__initial_queue_x1 = self.__queue_x1
        self.__initial_queue_x2 = self.__queue_x2

    def generate(self) -> int:
        """Generate the next bit within the generator sequence.

        Returns:
            int:
                The generated bit.
        """

        return (self.__forward_x1() + self.__forward_x2()) % 2

    def generate_sequence(self, length: int) -> np.array:
        """Generate a new sequence of random numbers.

        Args:
            length(int):
                Length of the sequence to be generated.

        Returns:
            np.array:
                A numpy array of dimension length containing a sequence of pseudo-random bits.
        """

        sequence = np.empty(length, dtype=int)
        for n in range(length):
            sequence[n] = self.generate()

        return sequence

    def reset(self) -> None:
        """Resets the generator to its default state.

        This implies reverting the queues back to their original state (at generator position n = 0).
        """

        self.__queue_x1 = self.__initial_queue_x1
        self.__queue_x2 = self.__initial_queue_x2

    def __forward_x1(self) -> int:

        x1 = (self.__queue_x1[3] + self.__queue_x1[0]) % 2

        self.__queue_x1.append(x1)
        return x1

    def __forward_x2(self) -> int:

        x2 = (self.__queue_x2[3] + self.__queue_x2[2] + self.__queue_x2[1] + self.__queue_x1[0]) % 2

        self.__queue_x2.append(x2)
        return x2


class Scrambler3GPP(Encoder):
    """This class represents a scrambler in the physical up- and down-link channel of the 3GPP.
    7.3.1.1
    """

    __randomGenerator: PseudoRandomGenerator
    __codewords: List[np.array]

    def __init__(self, params: ParametersEncoder, bits_in_frame: int) -> None:

        # Init base class (Encoder)
        super(Scrambler3GPP, self).__init__(params, bits_in_frame)

        self.__randomGenerator = PseudoRandomGenerator(np.random.randint(2, size=31))
        self.__codewords = list()

    def encode(self, data_bits: List[np.array]) -> List[np.array]:
        """This method encodes the incoming bits.

        Args:
            data_bits(List[np.array]):
                List of data_bits that are contained in the current frame.
                Each list element is one block with bits created by the BitSource.

        Returns:
            List[np.array]:
                List of blocks with the encoded bits. Each list item corresponds
                to a block containing a code word.
        """

        # Warn if the encoder overwrites an unused set of codewords
        if len(self.__codewords) > 0:
            raise RuntimeWarning("Unused codewords will be overwritten since the encoder re-encodes before decoding")

        self.__codewords.clear()
        codes = list()

        for block in data_bits:

            codeword = self.__randomGenerator.generate_sequence(block.shape[0])
            code = (block + codeword) % 2

            self.__codewords.append(codeword)
            codes.append(code)

        return codes

    def decode(self, encoded_bits: List[np.array]) -> List[np.array]:
        """Decode code words.

        Args:
            encoded_bits(List[np.array]):
                List of blocks with the encoded bits. Each list item corresponds
                to a block containing a code word. The expected input is soft bits.
        Returns:
            List[np.array]:
                List of data_bits that are contained in the current frame.
                Each list element is one block with bits created by the BitsSource.
        """

        # Make sure that enough codewords have been buffered
        if len(self.__codewords) < len(encoded_bits):
            raise RuntimeError("Codes require more codewords than available")

        data = list()

        for n, block in enumerate(encoded_bits):
            data.append((block + self.__codewords[n]) % 2)

        self.__codewords.clear()

        return data
