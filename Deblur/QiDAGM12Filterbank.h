#pragma once

class QiDAGM12Filterbank {
public:
	static const int filter_count = 16;
	static const int filter_size_y;
	static const int filter_size_x;

	static const double filter_values[16][5][5];
};