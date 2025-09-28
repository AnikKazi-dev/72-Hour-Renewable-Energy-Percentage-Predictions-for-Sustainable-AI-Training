
from scripts.season import resolve_season, months_for, data_path
SEASON = resolve_season(default='Winter').capitalize()
MONTHS = months_for(SEASON.lower())

# Expose standardized CNN_LSTM models
__all__ = [
	'CNN_LSTM_Model',
	'CNN_LSTM_Model_v2',
	'CNN_LSTM_Model_v3',
]