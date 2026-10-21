#pragma once

#include <Eigen/Dense>
#include <string>
#include <unordered_set>
#include <vector>

struct SpatialPartitionNode
{
	int axis = -1;
	double cut = 0.;
	int left = -1;
	int right = -1;
	int chunk_id = -1;
};

struct SpatialChunkInfo
{
	bool spatial = false;
	int chunk_count = 0;
	int chunk_id = -1;
	std::string partition_id;
	std::string selection_id;
	std::vector<SpatialPartitionNode> tree;
	// Unit route axis of an optional partition frame; the tree splits rotated XY.
	bool rotated = false;
	Eigen::Vector2d axis = Eigen::Vector2d::UnitX();
	std::unordered_set<std::string> core_frame_names;
	double overlap = 0.;
	double context_distance = 0.;
	double target_chunk_size = 0.;
	int context_frame_stride = 1;
	std::string metadata;
};

// Missing metadata is legacy only when neither source nor training has spatial markers.
bool loadSpatialChunkInfo(const std::string& path, SpatialChunkInfo& info, std::string& error);
bool validateSpatialTraining(
	const std::string& source_directory, const std::string& trained_directory,
	SpatialChunkInfo& info, std::string& error);
bool validateSpatialChunkSet(
	const std::vector<std::string>& source_directories,
	const std::vector<std::string>& trained_directories,
	std::vector<SpatialChunkInfo>& infos, bool& spatial, std::string& error);
int spatialOwner(const SpatialChunkInfo& info, const Eigen::Vector2d& position);

void mergeSpatialPlys(const std::vector<std::string>& trained_directories,
	const std::vector<SpatialChunkInfo>& infos, const std::string& output,
	int sh_degree, int iteration = -1);
