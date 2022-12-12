from .math import db2lin, lin2db, DbConversionType
from .resampling import delay_resampling_matrix
from .tile import tile_figures

__author__ = "Jan Adler"
__copyright__ = "Copyright 2022, Barkhausen Institut gGmbH"
__credits__ = ["Jan Adler"]
__license__ = "AGPLv3"
__version__ = "1.0.0"
__maintainer__ = "Jan Adler"
__email__ = "jan.adler@barkhauseninstitut.org"
__status__ = "Prototype"


__all__ = ["db2lin", "lin2db", "DbConversionType", "delay_resampling_matrix", "tile_figures"]
